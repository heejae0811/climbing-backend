import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    roc_curve, auc
)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from mrmr import mrmr_classif
from lime.lime_tabular import LimeTabularExplainer
import shap
import joblib


# =====================================================
# 전역 설정
# =====================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

RESULT_DIR = "./result"
os.makedirs(RESULT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

plt.rcParams["font.family"]        = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =====================================================
# scikit-learn tag compatibility for CatBoost
# =====================================================
class CatBoostClassifierSklearn(CatBoostClassifier, ClassifierMixin, BaseEstimator):
    pass


# =====================================================
# 1단계: 데이터 불러오기
# =====================================================
def data_loading():
    print("\n[1단계] Data Loading")

    files = glob.glob("./features_output/*.csv")
    files = [f for f in files if os.path.basename(f) != "all_global_features.csv"]
    print(f"찾은 파일: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError("폴더에 CSV 파일이 없습니다.")

    dfs = []
    common_numeric_cols = None
    nan_cols_union = set()
    inf_cols_union = set()

    for f in files:
        basename = os.path.basename(f)
        try:
            tmp = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            try:
                tmp = pd.read_csv(f, encoding="utf-8")
            except Exception:
                tmp = pd.read_csv(f, encoding="cp949")

        person_id = basename.split("_")[0]
        tmp["person_id"] = person_id

        if "label" not in tmp.columns:
            raise KeyError(f"{basename} 파일에 label 컬럼이 없습니다.")

        numeric_cols = set(tmp.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist())
        candidate_cols = numeric_cols - {"label"}

        if common_numeric_cols is None:
            common_numeric_cols = candidate_cols
        else:
            common_numeric_cols = common_numeric_cols & candidate_cols

        for col in candidate_cols:
            series = pd.to_numeric(tmp[col], errors="coerce")
            if series.isna().any():
                nan_cols_union.add(col)
            arr = series.to_numpy(dtype=float, copy=False)
            if np.isinf(arr).any():
                inf_cols_union.add(col)

        dfs.append(tmp)

    if not common_numeric_cols:
        raise ValueError("모든 파일에 공통으로 존재하는 numeric feature가 없습니다.")

    common_feature_cols = sorted(list(common_numeric_cols))
    exclude_features = ["body_size_median"]

    valid_feature_cols = [
        col for col in common_feature_cols
        if col not in nan_cols_union
        and col not in inf_cols_union
        and col not in exclude_features
    ]

    removed_nan_cols = sorted(list(set(common_feature_cols) & nan_cols_union))
    removed_inf_cols = sorted(list(set(common_feature_cols) & inf_cols_union))

    if len(valid_feature_cols) == 0:
        raise ValueError("NaN/Inf 제거 후 남은 공통 feature가 없습니다.")

    df = pd.concat(dfs, ignore_index=True)
    X = df[valid_feature_cols].apply(pd.to_numeric, errors="coerce")

    final_nan_cols = X.columns[X.isna().any()].tolist()
    final_inf_cols = X.columns[np.isinf(X.to_numpy(dtype=float)).any(axis=0)].tolist()

    if final_nan_cols or final_inf_cols:
        extra_remove = sorted(list(set(final_nan_cols + final_inf_cols)))
        print(f"\n[추가 안전 제거 변수] {extra_remove}")
        valid_feature_cols = [c for c in valid_feature_cols if c not in extra_remove]
        X = df[valid_feature_cols].apply(pd.to_numeric, errors="coerce")

    y      = df["label"].astype(int).values
    groups = df["person_id"].astype(str).values
    class_names  = ["Advanced", "Intermediate"]
    label_counts = np.bincount(y)

    print(f"공통 numeric feature: {len(common_feature_cols)}")
    print(f"NaN 때문에 제거된 feature: {len(removed_nan_cols)}")
    print(f"Inf 때문에 제거된 feature: {len(removed_inf_cols)}")
    print(f"최종 사용 feature: {len(valid_feature_cols)}")
    print(f"전체 participant 수: {len(np.unique(groups))}명")
    print(f"클래스 분포: Advanced(0)={label_counts[0]}개, Intermediate(1)={label_counts[1]}개")

    if removed_nan_cols:
        print(f"\n[NaN 포함으로 제거된 feature]\n{removed_nan_cols}")
    if removed_inf_cols:
        print(f"\n[Inf 포함으로 제거된 feature]\n{removed_inf_cols}")

    return X, y, groups, valid_feature_cols, class_names


# =====================================================
# 2단계: Train / Test 분리
# =====================================================
def data_split(X, y, groups, test_size=0.2, random_state=RANDOM_STATE, verbose=True):
    if verbose:
        print("\n[2단계] Train/Test Split (Group 기반, 8:2)")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train      = X.iloc[train_idx].reset_index(drop=True)
    X_test       = X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test     = y[train_idx], y[test_idx]
    groups_train        = groups[train_idx]
    groups_test         = groups[test_idx]

    overlap = set(np.unique(groups_train)) & set(np.unique(groups_test))
    if overlap:
        raise ValueError(f"동일 인물이 Train/Test 양쪽에 존재: {overlap}")

    if verbose:
        print(f"Train: {len(X_train)}개 ({len(np.unique(groups_train))}명) | "
              f"Test: {len(X_test)}개 ({len(np.unique(groups_test))}명)")
        print(f"Train 클래스 분포: {np.bincount(y_train)}")
        print(f"Test  클래스 분포: {np.bincount(y_test)}")
        print("✓ Train/Test 간 인물 중복 없음 확인")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


# =====================================================
# 3단계: Feature Selection — mRMR
# Minimum Redundancy Maximum Relevance
# - 클래스와의 관련성(Mutual Information) 최대화
# - 선택된 변수 간 중복성 최소화
# - 논문과 동일한 방법 (MIQ 기준)
# =====================================================
def feature_selection_mrmr(X_train, y_train, feature_cols,
                            n_features_to_select=None, min_features=5,
                            verbose=True):
    """
    mRMR (Minimum Redundancy Maximum Relevance) feature selection

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : array-like
    feature_cols : list of str
    n_features_to_select : int or None
        선택할 변수 수. None이면 전체 변수에 대해 mRMR 점수 계산 후
        점수가 양수인 변수만 선택 (자동 결정)
    min_features : int
        최소 유지 변수 수
    """
    if verbose:
        print("\n[3단계] Feature Selection — mRMR")
        print(f"  입력 feature: {len(feature_cols)}개")

    X_fs = X_train[feature_cols].copy()
    y_fs = pd.Series(y_train)

    # n_features_to_select가 없으면 전체 변수 수로 설정 후 점수 기반 자동 결정
    K = n_features_to_select if n_features_to_select is not None else len(feature_cols)
    K = min(K, len(feature_cols))

    try:
        selected = mrmr_classif(
            X=X_fs,
            y=y_fs,
            K=K,
        )
    except Exception as e:
        if verbose:
            print(f"  mRMR 실패: {e} → 전체 feature 사용")
        return feature_cols

    # n_features_to_select가 None이면 자동으로 상위 변수만 유지
    # mrmr_classif는 중요도 순으로 정렬된 리스트를 반환하므로
    # 전체를 넣었을 경우 min_features 이상을 유지하되 적절히 잘라냄
    if n_features_to_select is None:
        # 변수 수를 sqrt(전체 변수 수)로 자동 결정 (일반적 휴리스틱)
        auto_k = max(min_features, int(np.sqrt(len(feature_cols))))
        selected = selected[:auto_k]

    # min_features 보장
    if len(selected) < min_features:
        selected = selected[:min_features] if len(selected) >= min_features else feature_cols[:min_features]

    if verbose:
        print(f"  선택된 feature: {len(selected)}개")
        print(f"  Selected Features: {selected}")

    return selected


# =====================================================
# 4단계: 모델 정의
# =====================================================
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
        "AdaBoost": Pipeline([
            ("model", AdaBoostClassifier(random_state=RANDOM_STATE))
        ]),
        "LightGBM": Pipeline([
            ("model", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
        ]),
        "XGBoost": Pipeline([
            ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "CatBoost": Pipeline([
            ("model", CatBoostClassifierSklearn(random_state=RANDOM_STATE, verbose=False))
        ])
    }


# =====================================================
# 평가 지표
# =====================================================
def compute_metrics(model_label, y_test, y_pred, y_proba):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        full_cm = np.zeros((2, 2), dtype=int)
        full_cm[:cm.shape[0], :cm.shape[1]] = cm
        cm = full_cm

    tn, fp, fn, tp = cm.ravel()
    accuracy     = (y_pred == y_test).mean()
    precision    = precision_score(y_test, y_pred, zero_division=0)
    recall       = recall_score(y_test, y_pred, zero_division=0)
    f1           = f1_score(y_test, y_pred, zero_division=0)
    specificity  = (tn / (tn + fp)) if (tn + fp) > 0 else 0
    sensitivity  = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc          = matthews_corrcoef(y_test, y_pred)
    try:
        auc_value = roc_auc_score(y_test, y_proba)
    except Exception:
        auc_value = np.nan

    return {
        "Model":                 model_label,
        "Accuracy_raw":          accuracy,
        "Precision_raw":         precision,
        "Recall_raw":            recall,
        "F1_raw":                f1,
        "Specificity_raw":       specificity,
        "Sensitivity_raw":       sensitivity,
        "Balanced_Accuracy_raw": balanced_acc,
        "MCC_raw":               mcc,
        "AUC_raw":               auc_value,
        "Accuracy":              round(accuracy, 2),
        "Precision":             round(precision, 2),
        "Recall":                round(recall, 2),
        "F1":                    round(f1, 2),
        "Specificity":           round(specificity, 2),
        "Sensitivity":           round(sensitivity, 2),
        "Balanced_Accuracy":     round(balanced_acc, 2),
        "MCC":                   round(mcc, 2),
        "AUC":                   round(auc_value, 2) if pd.notnull(auc_value) else np.nan,
    }


# =====================================================
# 모델 학습 및 평가
# =====================================================
def evaluate_model(pipeline, model_name, X_train, y_train, X_test, y_test, verbose=True):
    model = clone(pipeline)
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") \
              else model.decision_function(X_test)
    metrics = compute_metrics(model_name, y_test, y_pred, y_proba)

    if verbose:
        print(f"  Acc={metrics['Accuracy']:.2f} | F1={metrics['F1']:.2f} | "
              f"MCC={metrics['MCC']:.2f} | "
              f"AUC={metrics['AUC'] if pd.notnull(metrics['AUC']) else 'nan'}")

    return model, y_proba, metrics


# =====================================================
# Excel 저장 헬퍼
# =====================================================
NUMERIC_METRIC_COLS = [
    "Accuracy", "Precision", "Recall", "F1",
    "Specificity", "Sensitivity", "Balanced_Accuracy", "MCC", "AUC"
]

def _apply_number_format(writer, sheet_name):
    ws = writer.sheets[sheet_name]
    header = [cell.value for cell in ws[1]]
    for col_idx, col_name in enumerate(header, start=1):
        if col_name in NUMERIC_METRIC_COLS:
            for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    cell.number_format = "0.00"


# =====================================================
# ① 전체 데이터 Feature Selection 비교 결과 저장
# =====================================================
def save_fs_comparison(results_with_fs, results_without_fs,
                       y_test, y_proba_dict_with, y_proba_dict_without):
    print("\n[결과 저장] Feature Selection 비교")

    metric_cols = ["Model"] + NUMERIC_METRIC_COLS
    df_with    = pd.DataFrame(results_with_fs)[metric_cols].copy()
    df_without = pd.DataFrame(results_without_fs)[metric_cols].copy()
    df_with["Feature_Selection"]    = "With FS"
    df_without["Feature_Selection"] = "Without FS"
    df_combined = pd.concat([df_with, df_without], ignore_index=True)

    excel_path = os.path.join(RESULT_DIR, "fs_comparison_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_with.drop(columns=["Feature_Selection"]).sort_values("F1", ascending=False)\
            .to_excel(writer, sheet_name="With_FeatureSelection", index=False)
        df_without.drop(columns=["Feature_Selection"]).sort_values("F1", ascending=False)\
            .to_excel(writer, sheet_name="Without_FeatureSelection", index=False)
        df_combined.sort_values(["Model", "Feature_Selection"])\
            .to_excel(writer, sheet_name="Combined", index=False)

        _apply_number_format(writer, "With_FeatureSelection")
        _apply_number_format(writer, "Without_FeatureSelection")
        _apply_number_format(writer, "Combined")

    print(f"  저장: {excel_path}")

    model_names = df_with["Model"].tolist()
    vals_w  = df_with.set_index("Model").reindex(model_names)["Accuracy"].values.astype(float)
    vals_wo = df_without.set_index("Model").reindex(model_names)["Accuracy"].values.astype(float)
    x = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(model_names, vals_w,  marker="o", linewidth=2, color="#2196F3", label="With Feature Selection")
    ax.plot(model_names, vals_wo, marker="o", linewidth=2, color="#FF5722", label="Without Feature Selection")
    for i, (vw, vwo) in enumerate(zip(vals_w, vals_wo)):
        ax.text(i, vw  + 0.012, f"{vw:.2f}",  ha="center", fontsize=8, color="#1565C0")
        ax.text(i, vwo - 0.022, f"{vwo:.2f}", ha="center", fontsize=8, color="#BF360C")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy: With Feature Selection vs Without Feature Selection",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "fs_comparison_accuracy_line.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  저장: fs_comparison_accuracy_line.png")

    _plot_roc(y_test, y_proba_dict_with,
              "ROC Curve — With Feature Selection", "roc_with_fs.png")
    _plot_roc(y_test, y_proba_dict_without,
              "ROC Curve — Without Feature Selection", "roc_without_fs.png")
    _print_fs_summary(df_with, df_without)


def _plot_roc(y_test, y_proba_dict, title, filename):
    plt.figure(figsize=(8, 6))
    for name, y_proba in y_proba_dict.items():
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=300)
    plt.close()


def _print_fs_summary(df_with, df_without):
    key = ["Accuracy", "F1", "Balanced_Accuracy", "MCC", "AUC"]
    df_w  = df_with.set_index("Model")[key].astype(float)
    df_wo = df_without.set_index("Model")[key].astype(float)
    df_d  = df_w - df_wo

    print("\n" + "=" * 80)
    print("  Feature Selection 유무 성능 비교 요약")
    print("=" * 80)
    print("\n[Δ = With FS − Without FS  (양수: FS 유리, 음수: FS 불리)]")
    print(df_d.round(2).to_string())
    print(f"\n[With FS 평균]    {df_w.mean().round(2).to_dict()}")
    print(f"[Without FS 평균] {df_wo.mean().round(2).to_dict()}")
    print(f"[Δ 평균]          {df_d.mean().round(2).to_dict()}")
    best_w  = df_w["F1"].idxmax()
    best_wo = df_wo["F1"].idxmax()
    print(f"\n최고 F1 (With FS):    {best_w}  ({df_w.loc[best_w, 'F1']:.2f})")
    print(f"최고 F1 (Without FS): {best_wo}  ({df_wo.loc[best_wo, 'F1']:.2f})")
    print("=" * 80)


# =====================================================
# ② Sample Size Analysis
# =====================================================
def is_valid_subset(y_sub, groups_sub):
    return len(np.unique(y_sub)) >= 2 and len(np.unique(groups_sub)) >= 4


def subset_by_group_fraction_stratified(X, y, groups, fraction, random_state=42):
    df_groups      = pd.DataFrame({"group": groups, "label": y})
    group_label_df = df_groups.groupby("group")["label"].first().reset_index()
    rng            = np.random.RandomState(random_state)
    selected_groups = []

    for label in sorted(group_label_df["label"].unique()):
        label_groups = group_label_df[group_label_df["label"] == label]["group"].values
        n_select     = max(1, min(int(round(len(label_groups) * fraction)), len(label_groups)))
        selected_groups.extend(rng.choice(label_groups, size=n_select, replace=False).tolist())

    mask = np.isin(groups, sorted(selected_groups))
    return X.loc[mask].reset_index(drop=True), y[mask], groups[mask]


def sample_size_analysis(X, y, groups, feature_cols,
                         fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
                         test_size=0.2, min_features=5, n_repeats=100):
    print("\n" + "=" * 70)
    print("  [Sample Size Analysis]  With FS (mRMR)")
    print("=" * 70)

    all_rows    = []
    models      = build_models()
    model_names = list(models.keys())

    for frac in fractions:
        pct = int(frac * 100)
        print(f"\n{'─'*70}\n  샘플 비율 {pct}%\n{'─'*70}")

        for rep in range(n_repeats):
            seed = RANDOM_STATE + rep
            print(f"  반복 {rep+1}/{n_repeats}")

            X_sub, y_sub, groups_sub = subset_by_group_fraction_stratified(
                X, y, groups, fraction=frac, random_state=seed)

            if not is_valid_subset(y_sub, groups_sub):
                print("    subset invalid → skip")
                continue

            try:
                X_tr, X_te, y_tr, y_te, g_tr, _ = data_split(
                    X_sub, y_sub, groups_sub,
                    test_size=test_size, random_state=seed, verbose=False)
            except Exception as e:
                print(f"    split 실패: {e}")
                continue

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                print("    class 부족 → skip")
                continue

            try:
                sel_feats = feature_selection_mrmr(
                    X_tr, y_tr, feature_cols,
                    min_features=min_features, verbose=False)
            except Exception as e:
                print(f"    FS 실패: {e}")
                sel_feats = feature_cols

            n_participants = len(np.unique(groups_sub))
            n_videos       = len(X_sub)

            for model_name in model_names:
                try:
                    _, _, metrics = evaluate_model(
                        models[model_name], model_name,
                        X_tr[sel_feats], y_tr,
                        X_te[sel_feats], y_te,
                        verbose=False)

                    all_rows.append({
                        "sample_size_percent": pct,
                        "feature_selection":   "With Feature Selection (mRMR)",
                        "repeat":              rep + 1,
                        "participant_count":   n_participants,
                        "video_count":         n_videos,
                        "model":               model_name,
                        "Accuracy":            metrics["Accuracy_raw"],
                        "F1":                  metrics["F1_raw"],
                        "MCC":                 metrics["MCC_raw"],
                        "AUC":                 metrics["AUC_raw"] if pd.notnull(metrics["AUC_raw"]) else np.nan,
                    })
                except Exception as e:
                    print(f"    {model_name} 실패: {e}")

    df_raw = pd.DataFrame(all_rows)
    if df_raw.empty:
        print("sample size 분석 결과 없음")
        return df_raw

    raw_path = os.path.join(RESULT_DIR, "sample_size_raw.xlsx")
    df_raw.to_excel(raw_path, index=False)
    print(f"\n  raw 결과 저장: {raw_path}")

    df_summary = (
        df_raw
        .groupby(["sample_size_percent", "feature_selection", "model"], as_index=False)
        .agg(
            Accuracy_mean=("Accuracy", "mean"), Accuracy_sd=("Accuracy", "std"),
            F1_mean=("F1", "mean"),             F1_sd=("F1", "std"),
            MCC_mean=("MCC", "mean"),           MCC_sd=("MCC", "std"),
            AUC_mean=("AUC", "mean"),           AUC_sd=("AUC", "std"),
            participant_count=("participant_count", "mean"),
            video_count=("video_count", "mean"),
        )
    )

    summary_path = os.path.join(RESULT_DIR, "sample_size_summary.xlsx")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        ws = writer.sheets["Summary"]
        mean_cols = ["Accuracy_mean", "Accuracy_sd", "F1_mean", "F1_sd",
                     "MCC_mean", "MCC_sd", "AUC_mean", "AUC_sd"]
        header = [c.value for c in ws[1]]
        for ci, cn in enumerate(header, 1):
            if cn in mean_cols:
                for row in ws.iter_rows(min_row=2, min_col=ci, max_col=ci):
                    for cell in row:
                        cell.number_format = "0.00"
    print(f"  요약 결과 저장: {summary_path}")

    _plot_sample_size(df_summary, model_names, fractions)
    return df_summary


def _plot_sample_size(df_summary, model_names, fractions):
    pct_list = sorted([int(f * 100) for f in fractions])
    cmap     = plt.get_cmap("tab10")
    colors   = {m: cmap(i) for i, m in enumerate(model_names)}

    fig, ax = plt.subplots(figsize=(10, 6))
    df_fs   = df_summary[df_summary["feature_selection"].str.contains("With Feature Selection")]

    for model_name in model_names:
        tmp  = df_fs[df_fs["model"] == model_name].sort_values("sample_size_percent")
        if tmp.empty:
            continue
        yerr = tmp["Accuracy_sd"].fillna(0).values
        ax.errorbar(tmp["sample_size_percent"], tmp["Accuracy_mean"],
                    yerr=yerr, marker="o", linewidth=1.8, capsize=3,
                    color=colors[model_name], label=model_name)

    ax.set_xlabel("Sample size (%)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy across Sample Sizes (mRMR Feature Selection)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(pct_list)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "sample_size_accuracy.png"), dpi=300)
    plt.close()
    print("  저장: sample_size_accuracy.png")


# =====================================================
# ③ 최종 모델 부가 결과
# =====================================================
def save_best_model_extras(best_model, best_model_name,
                           X_test_fs, y_test, selected_features, class_names):
    y_pred = best_model.predict(X_test_fs)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "best_confusion_matrix.png"), dpi=300)
    plt.close()

    perm = permutation_importance(
        best_model, X_test_fs, y_test,
        scoring="f1", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1)

    df_fi = pd.DataFrame({
        "Feature":    selected_features,
        "Importance": np.round(perm.importances_mean, 4),
        "Std":        np.round(perm.importances_std, 4),
    }).sort_values("Importance", ascending=False)

    df_fi.to_excel(os.path.join(RESULT_DIR, "feature_importance.xlsx"), index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(df_fi["Feature"], df_fi["Importance"], xerr=df_fi["Std"], align="center")
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance (Permutation, F1) — {best_model_name}")
    plt.xlabel("Mean Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "feature_importance_plot.png"), dpi=300)
    plt.close()


# =====================================================
# XAI: LIME
# =====================================================
def xai_lime(best_model, X_train, X_test, y_test, selected_features, class_names):
    print("\n[XAI] LIME")
    try:
        if not hasattr(best_model, "predict_proba"):
            print("  LIME 생략: predict_proba 미지원")
            return

        FEATURE_NAME_MAP = {
            "hip_center_velocity_mean": "Hip speed",
            "shoulder_y_symmetry_mean": "Shoulder height symmetry",
            "hip_center_jerk_sd": "Hip movement smoothness",
            "hip_center_velocity_sd": "Hip speed consistency",
            "left_hip_velocity_mean": "Left hip speed",
            "shoulder_center_vertical_sway": "Shoulder vertical sway",
            "hip_center_acceleration_sd": "Hip acceleration consistency",
            "right_hip_acceleration_mean": "Right hip acceleration",
        }
        pretty = [FEATURE_NAME_MAP.get(f, f) for f in selected_features]

        has_scaler = "scaler" in best_model.named_steps
        if has_scaler:
            sc     = best_model.named_steps["scaler"]
            X_tr_l = pd.DataFrame(sc.transform(X_train), columns=selected_features)
            X_te_l = pd.DataFrame(sc.transform(X_test),  columns=selected_features)
        else:
            X_tr_l, X_te_l = X_train.copy(), X_test.copy()

        explainer = LimeTabularExplainer(
            training_data=X_tr_l.values,
            feature_names=pretty,
            class_names=class_names,
            mode="classification",
            random_state=RANDOM_STATE)

        y_pred = best_model.predict(X_test)
        mis    = np.where(y_pred != y_test)[0]
        idx    = mis[0] if len(mis) > 0 else 0

        exp = explainer.explain_instance(
            X_te_l.iloc[idx].values,
            best_model.predict_proba,
            num_features=min(10, len(selected_features)),
            num_samples=5000)

        exp.save_to_file(os.path.join(RESULT_DIR, "lime_explanation.html"))
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(8, 6)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULT_DIR, "lime_explanation.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"]).to_excel(
            os.path.join(RESULT_DIR, "lime_explanation.xlsx"), index=False)
        print(f"  LIME 완료 (sample idx={idx})")

    except Exception as e:
        print(f"  LIME 실패: {e}")


# =====================================================
# XAI: SHAP
# =====================================================
def xai_shap(best_model, X_train, X_test, selected_features):
    print("\n[XAI] SHAP")
    try:
        FEATURE_NAME_MAP = {
            "hip_center_velocity_mean": "Hip speed",
            "shoulder_y_symmetry_mean": "Shoulder height symmetry",
            "hip_center_jerk_sd": "Hip movement smoothness",
            "hip_center_velocity_sd": "Hip speed consistency",
            "left_hip_velocity_mean": "Left hip speed",
            "shoulder_center_vertical_sway": "Shoulder vertical sway",
            "hip_center_acceleration_sd": "Hip acceleration consistency",
            "right_hip_acceleration_mean": "Right hip acceleration",
        }
        pretty = [FEATURE_NAME_MAP.get(f, f) for f in selected_features]

        has_scaler  = "scaler" in best_model.named_steps
        model_step  = best_model.named_steps["model"]
        tree_models = {"DecisionTreeClassifier", "RandomForestClassifier",
                       "GradientBoostingClassifier", "LGBMClassifier",
                       "XGBClassifier", "CatBoostClassifier", "CatBoostClassifierSklearn"}
        is_tree = model_step.__class__.__name__ in tree_models

        X_bg  = X_train.sample(n=min(50, len(X_train)), random_state=RANDOM_STATE)
        X_exp = X_test.iloc[:min(50, len(X_test))].copy().reset_index(drop=True)

        if is_tree:
            X_exp_in = pd.DataFrame(
                best_model.named_steps["scaler"].transform(X_exp), columns=selected_features
            ) if has_scaler else X_exp.copy()
            explainer = shap.TreeExplainer(model_step)
            shap_obj  = explainer(X_exp_in)
            raw, base = shap_obj.values, shap_obj.base_values
            data_plot = shap_obj.data if shap_obj.data is not None else X_exp_in.values
            if raw.ndim == 3:
                values      = raw[:, :, 1]
                base_values = base[:, 1] if np.ndim(base) > 1 else base
            else:
                values, base_values = raw, base
            expl = shap.Explanation(values=values, base_values=base_values,
                                    data=data_plot, feature_names=selected_features)
        else:
            sc = best_model.named_steps["scaler"] if has_scaler else None
            X_bg_t  = pd.DataFrame(sc.transform(X_bg),  columns=selected_features) if sc else X_bg.copy()
            X_exp_t = pd.DataFrame(sc.transform(X_exp), columns=selected_features) if sc else X_exp.copy()
            ke = shap.KernelExplainer(model_step.predict_proba, X_bg_t)
            sv = ke.shap_values(X_exp_t, nsamples=100)
            if isinstance(sv, list):
                values   = sv[1]
                base_val = ke.expected_value[1] if isinstance(ke.expected_value, (list, np.ndarray)) \
                           else ke.expected_value
            else:
                sv_np    = np.array(sv)
                values   = sv_np[:, :, 1] if sv_np.ndim == 3 else sv_np
                base_val = ke.expected_value[1] if isinstance(ke.expected_value, (list, np.ndarray)) \
                           else ke.expected_value
            expl = shap.Explanation(values=values,
                                    base_values=np.repeat(base_val, X_exp_t.shape[0]),
                                    data=X_exp_t.values, feature_names=pretty)

        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(expl, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "shap_beeswarm.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("  SHAP 완료")

    except Exception as e:
        print(f"  SHAP 실패: {e}")


# =====================================================
# MAIN
# =====================================================
def main():
    print("\n" + "=" * 70)
    print("  머신러닝 파이프라인")
    print("=" * 70)

    # ── 1. 데이터 로드 ──
    X, y, groups, feature_cols, class_names = data_loading()

    # ── 2. Train/Test 분리 ──
    X_train, X_test, y_train, y_test, groups_train, groups_test = data_split(
        X, y, groups, test_size=0.2, random_state=RANDOM_STATE, verbose=True)

    # ── 3. mRMR Feature Selection ──
    # n_features_to_select=None → sqrt(전체 변수 수) 기준 자동 결정
    # 직접 지정하려면 n_features_to_select=10 처럼 넣으면 됨
    selected_features = feature_selection_mrmr(
        X_train, y_train, feature_cols,
        n_features_to_select=None, min_features=5, verbose=True)

    print(f"\n  전체 feature 수:   {len(feature_cols)}개")
    print(f"  제거된 feature 수: {len(feature_cols) - len(selected_features)}개")
    print(f"  선택된 feature 수: {len(selected_features)}개")

    X_train_fs  = X_train[selected_features]
    X_test_fs   = X_test[selected_features]
    X_train_all = X_train[feature_cols]
    X_test_all  = X_test[feature_cols]

    # ── 4. With FS / Without FS 평가 ──
    models = build_models()
    results_with_fs, results_without_fs         = [], []
    y_proba_dict_with, y_proba_dict_without      = {}, {}
    best_model, best_model_name, best_f1_raw     = None, None, -999

    print("\n" + "=" * 70)
    print("  [전체 데이터 — With Feature Selection (mRMR)]")
    print("=" * 70)
    for model_name, pipeline in models.items():
        print(f"\n{'─'*55}\n  {model_name}\n{'─'*55}")
        try:
            trained, y_proba, metrics = evaluate_model(
                pipeline, model_name,
                X_train_fs, y_train, X_test_fs, y_test, verbose=True)
            results_with_fs.append(metrics)
            y_proba_dict_with[model_name] = y_proba
            if metrics["F1_raw"] > best_f1_raw:
                best_f1_raw, best_model, best_model_name = metrics["F1_raw"], trained, model_name
        except Exception as e:
            print(f"  실패: {e}")

    print("\n" + "=" * 70)
    print("  [전체 데이터 — Without Feature Selection]")
    print("=" * 70)
    for model_name, pipeline in models.items():
        print(f"\n{'─'*55}\n  {model_name}\n{'─'*55}")
        try:
            _, y_proba, metrics = evaluate_model(
                pipeline, model_name,
                X_train_all, y_train, X_test_all, y_test, verbose=True)
            results_without_fs.append(metrics)
            y_proba_dict_without[model_name] = y_proba
        except Exception as e:
            print(f"  실패: {e}")

    # ── 5. FS 비교 결과 저장 ──
    save_fs_comparison(
        results_with_fs, results_without_fs,
        y_test, y_proba_dict_with, y_proba_dict_without)

    # ── 6. Sample Size Analysis ──
    sample_size_analysis(
        X, y, groups, feature_cols,
        fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
        test_size=0.2, min_features=5, n_repeats=100)

    # ── 7. Best Model 부가 결과 ──
    if best_model is not None:
        print(f"\n{'='*70}")
        print(f"  Best Model (With FS mRMR, F1 기준): {best_model_name}  F1={best_f1_raw:.2f}")
        print(f"{'='*70}")
        save_best_model_extras(
            best_model, best_model_name,
            X_test_fs, y_test, selected_features, class_names)
        xai_lime(best_model, X_train_fs, X_test_fs, y_test, selected_features, class_names)
        xai_shap(best_model, X_train_fs, X_test_fs, selected_features)

        joblib.dump(best_model,        os.path.join(RESULT_DIR, "best_model.pkl"))
        joblib.dump(selected_features, os.path.join(RESULT_DIR, "best_features.pkl"))
        print("  모델 저장 완료: best_model.pkl / best_features.pkl")

    print("\n전체 파이프라인 완료!")


if __name__ == "__main__":
    main()