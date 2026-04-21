import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
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
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from mrmr import mrmr_classif
from lime.lime_tabular import LimeTabularExplainer
import shap
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

RESULT_DIR = "./result"
os.makedirs(RESULT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")
plt.rcParams["font.family"]        = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

INSTABILITY_STABLE     = 0.10
INSTABILITY_ACCEPTABLE = 0.15
INSTABILITY_UNSTABLE   = 0.20


class CatBoostClassifierSklearn(CatBoostClassifier, ClassifierMixin, BaseEstimator):
    pass


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

        numeric_cols   = set(tmp.select_dtypes(include=["float64","int64","float32","int32"]).columns.tolist())
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
        raise ValueError("공통 numeric feature 없음")

    common_feature_cols = sorted(list(common_numeric_cols))
    exclude_features    = ["body_size_median"]
    valid_feature_cols  = [
        col for col in common_feature_cols
        if col not in nan_cols_union
        and col not in inf_cols_union
        and col not in exclude_features
    ]

    removed_nan_cols = sorted(list(set(common_feature_cols) & nan_cols_union))
    removed_inf_cols = sorted(list(set(common_feature_cols) & inf_cols_union))
    if len(valid_feature_cols) == 0:
        raise ValueError("NaN/Inf 제거 후 남은 feature 없음")

    df = pd.concat(dfs, ignore_index=True)
    X  = df[valid_feature_cols].apply(pd.to_numeric, errors="coerce")

    final_nan_cols = X.columns[X.isna().any()].tolist()
    final_inf_cols = X.columns[np.isinf(X.to_numpy(dtype=float)).any(axis=0)].tolist()
    if final_nan_cols or final_inf_cols:
        extra_remove       = sorted(list(set(final_nan_cols + final_inf_cols)))
        print(f"\n[추가 안전 제거 변수] {extra_remove}")
        valid_feature_cols = [c for c in valid_feature_cols if c not in extra_remove]
        X                  = df[valid_feature_cols].apply(pd.to_numeric, errors="coerce")

    y            = df["label"].astype(int).values
    groups       = df["person_id"].astype(str).values
    class_names  = ["Advanced", "Intermediate"]
    label_counts = np.bincount(y)

    print(f"공통 numeric feature: {len(common_feature_cols)}")
    print(f"NaN 제거: {len(removed_nan_cols)}개  |  Inf 제거: {len(removed_inf_cols)}개")
    print(f"최종 사용 feature: {len(valid_feature_cols)}")
    print(f"전체 participant 수: {len(np.unique(groups))}명")
    print(f"클래스 분포: Advanced(0)={label_counts[0]}개, Intermediate(1)={label_counts[1]}개")
    if removed_nan_cols:
        print(f"\n[NaN 포함 제거 feature]\n{removed_nan_cols}")
    if removed_inf_cols:
        print(f"\n[Inf 포함 제거 feature]\n{removed_inf_cols}")

    return X, y, groups, valid_feature_cols, class_names


def data_split(X, y, groups, test_size=0.2, random_state=RANDOM_STATE, verbose=True):
    if verbose:
        print("\n[2단계] Train/Test Split (Group 기반, 8:2)")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train      = X.iloc[train_idx].reset_index(drop=True)
    X_test       = X.iloc[test_idx].reset_index(drop=True)
    y_train      = y[train_idx]
    y_test       = y[test_idx]
    groups_train = groups[train_idx]
    groups_test  = groups[test_idx]

    overlap = set(np.unique(groups_train)) & set(np.unique(groups_test))
    if overlap:
        raise ValueError(f"동일 인물이 Train/Test 양쪽에 존재: {overlap}")

    if verbose:
        print(f"Train: {len(X_train)}개 ({len(np.unique(groups_train))}명) | "
              f"Test: {len(X_test)}개 ({len(np.unique(groups_test))}명)")
        print(f"Train 클래스 분포: {np.bincount(y_train)}")
        print(f"Test  클래스 분포: {np.bincount(y_test)}")
        print("✓ Train/Test 간 인물 중복 없음 확인")

    return X_train, X_test, y_train, y_test, groups_train, groups_test, test_idx


def feature_selection(X_train, y_train, feature_cols, min_features=5, verbose=True):
    """
    Hybrid Feature Selection: mRMR → RFECV

    Stage 1. mRMR (Minimum Redundancy Maximum Relevance)
      - MIQ 기준으로 K = sqrt(전체 feature 수) 개 선택
      - 관련성 최대화 + 중복성 최소화

    Stage 2. RFECV (Recursive Feature Elimination with Cross-Validation)
      - mRMR 결과를 입력으로 사용
      - Random Forest 기반 feature importance로 반복 제거
      - StratifiedKFold 5-fold CV, 최적화 지표: F1-score
      - 최적 feature 수 자동 결정
    """
    if verbose:
        print("\n[3단계] Feature Selection — Hybrid (mRMR → RFECV)")
        print(f"  입력 feature: {len(feature_cols)}개")

    # ── Stage 1: mRMR ──
    if verbose:
        print("\n  [Stage 1] mRMR")

    K = max(min_features, int(np.sqrt(len(feature_cols))))
    K = min(K, len(feature_cols))

    try:
        mrmr_selected = mrmr_classif(
            X=X_train[feature_cols].copy(),
            y=pd.Series(y_train),
            K=K
        )
        mrmr_selected = list(mrmr_selected)
    except Exception as e:
        if verbose:
            print(f"  mRMR 실패: {e} → 전체 feature로 RFECV 진행")
        mrmr_selected = list(feature_cols)

    if verbose:
        print(f"  mRMR 선택 feature: {len(mrmr_selected)}개")
        print(f"  {mrmr_selected}")

    # ── Stage 2: RFECV ──
    if verbose:
        print("\n  [Stage 2] RFECV")

    rfecv = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
        min_features_to_select=min_features,
        n_jobs=-1
    )

    try:
        rfecv.fit(X_train[mrmr_selected], y_train)
        final_selected = [f for f, m in zip(mrmr_selected, rfecv.support_) if m]
        if verbose:
            print(f"  RFECV 최적 feature 수: {rfecv.n_features_}")
            print(f"  RFECV 선택 feature: {len(final_selected)}개")
            print(f"  {final_selected}")
    except Exception as e:
        if verbose:
            print(f"  RFECV 실패: {e} → mRMR 결과 그대로 사용")
        final_selected = mrmr_selected

    if len(final_selected) < min_features:
        final_selected = mrmr_selected[:min_features]

    if verbose:
        print(f"\n  ✓ 최종 선택 feature: {len(final_selected)}개")
        print(f"    전체 {len(feature_cols)}개 → mRMR {len(mrmr_selected)}개 → RFECV {len(final_selected)}개")

    return final_selected


def build_models():
    return {
        "Logistic regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
        ]),
        "K-nearest neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "Support vector machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=RANDOM_STATE))
        ]),
        "Decision tree": Pipeline([
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "Random forest": Pipeline([
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "Gradient boosting": Pipeline([
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


def evaluate_model(pipeline, model_name, X_train, y_train, X_test, y_test, verbose=True):
    model   = clone(pipeline)
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = (model.predict_proba(X_test)[:, 1]
               if hasattr(model, "predict_proba")
               else model.decision_function(X_test))
    metrics = compute_metrics(model_name, y_test, y_pred, y_proba)

    if verbose:
        print(f"  Acc={metrics['Accuracy']:.2f} | F1={metrics['F1']:.2f} | "
              f"MCC={metrics['MCC']:.2f} | "
              f"AUC={metrics['AUC'] if pd.notnull(metrics['AUC']) else 'nan'}")

    return model, y_proba, metrics


NUMERIC_METRIC_COLS = [
    "Accuracy", "Precision", "Recall", "F1",
    "Specificity", "Sensitivity", "Balanced_Accuracy", "MCC", "AUC"
]

def _apply_number_format(writer, sheet_name):
    ws     = writer.sheets[sheet_name]
    header = [cell.value for cell in ws[1]]
    for col_idx, col_name in enumerate(header, start=1):
        if col_name in NUMERIC_METRIC_COLS:
            for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    cell.number_format = "0.00"


def save_results(results, y_test, y_proba_dict):
    print("\n[결과 저장] 모델 평가 결과")
    metric_cols = ["Model"] + NUMERIC_METRIC_COLS
    df_results  = pd.DataFrame(results)[metric_cols].copy()

    excel_path = os.path.join(RESULT_DIR, "model_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_results.sort_values("F1", ascending=False).to_excel(
            writer, sheet_name="Results", index=False
        )
        _apply_number_format(writer, "Results")
    print(f"  저장: {excel_path}")

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
    plt.title("ROC Curve", fontsize=13)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"), dpi=300)
    plt.close()
    print("  저장: roc_curve.png")

    df_show = df_results.sort_values("F1", ascending=False)
    print("\n" + "=" * 70)
    print("  모델 성능 요약 (F1 기준 정렬)")
    print("=" * 70)
    print(df_show[["Model","Accuracy","F1","MCC","AUC","Sensitivity","Specificity"]].to_string(index=False))
    print("=" * 70)
    best = df_show.iloc[0]
    print(f"\n  Best Model: {best['Model']}  F1={best['F1']:.2f}  AUC={best['AUC']}")


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
    print("  [Sample Size Analysis]  Prediction Instability")
    print("  방법: Kim & Park (2026) Figure 2")
    print(f"  각 표본 비율에서 {n_repeats}회 반복 → F1의 SD = Instability")
    print("=" * 70)

    models      = build_models()
    model_names = list(models.keys())
    all_rows    = []

    for frac in fractions:
        pct = int(frac * 100)
        print(f"\n{'─'*70}\n  샘플 비율 {pct}%\n{'─'*70}")

        for rep in range(n_repeats):
            seed = RANDOM_STATE + rep
            X_sub, y_sub, groups_sub = subset_by_group_fraction_stratified(
                X, y, groups, fraction=frac, random_state=seed)

            if not is_valid_subset(y_sub, groups_sub):
                continue

            try:
                X_tr, X_te, y_tr, y_te, _, _, _ = data_split(
                    X_sub, y_sub, groups_sub,
                    test_size=test_size, random_state=seed, verbose=False)
            except Exception:
                continue

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            # 반복 속도를 위해 mRMR만 적용 (RFECV는 계산 비용 높음)
            try:
                K         = max(min_features, int(np.sqrt(len(feature_cols))))
                K         = min(K, len(feature_cols))
                sel_feats = list(mrmr_classif(
                    X=X_tr[feature_cols].copy(), y=pd.Series(y_tr), K=K
                ))
            except Exception:
                sel_feats = list(feature_cols)

            for model_name in model_names:
                try:
                    _, _, metrics = evaluate_model(
                        models[model_name], model_name,
                        X_tr[sel_feats], y_tr,
                        X_te[sel_feats], y_te, verbose=False)
                    all_rows.append({
                        "sample_size_percent": pct,
                        "model":               model_name,
                        "repeat":              rep + 1,
                        "F1":                  metrics["F1_raw"],
                    })
                except Exception:
                    pass

        print(f"  {pct}% 완료")

    df_raw = pd.DataFrame(all_rows)
    if df_raw.empty:
        print("  결과 없음")
        return df_raw

    df_instability = (
        df_raw
        .groupby(["sample_size_percent", "model"], as_index=False)
        .agg(
            instability=("F1", "std"),
            F1_mean=("F1", "mean"),
            n_valid_repeats=("F1", "count"),
        )
    )

    df_instability.to_excel(
        os.path.join(RESULT_DIR, "sample_size_instability.xlsx"), index=False
    )
    print(f"\n  결과 저장: sample_size_instability.xlsx")

    _plot_prediction_instability(df_instability, model_names, fractions)
    _print_instability_summary(df_instability, fractions)

    return df_instability


def _plot_prediction_instability(df_instability, model_names, fractions):
    pct_list = sorted([int(f * 100) for f in fractions])
    cmap     = plt.get_cmap("tab10")
    colors   = {m: cmap(i) for i, m in enumerate(model_names)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name in model_names:
        tmp = (df_instability[df_instability["model"] == model_name]
               .sort_values("sample_size_percent"))
        if tmp.empty:
            continue
        ax.plot(tmp["sample_size_percent"], tmp["instability"], marker="o", linewidth=1.8, color=colors[model_name], label=model_name)

    ax.axhline(y=INSTABILITY_STABLE,     color="black", linestyle="-",  linewidth=1.0, label=f"Stable ({INSTABILITY_STABLE})")
    ax.axhline(y=INSTABILITY_ACCEPTABLE, color="black", linestyle="--", linewidth=1.0, label=f"Acceptable ({INSTABILITY_ACCEPTABLE})")
    ax.axhline(y=INSTABILITY_UNSTABLE,   color="black", linestyle=":",  linewidth=1.0, label=f"Unstable ({INSTABILITY_UNSTABLE})")

    ax.set_xlabel("Sample size (%)", fontsize=12)
    ax.set_ylabel("Average instability (SD)", fontsize=12)
    ax.set_xticks(pct_list)
    ax.set_ylim(0, max(0.35, df_instability["instability"].max() + 0.05))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "sample_size_instability.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  저장: sample_size_instability.png")


def _print_instability_summary(df_instability, fractions):
    print("\n" + "=" * 70)
    print("  Prediction Instability 요약")
    print(f"  기준: Stable ≤ {INSTABILITY_STABLE} | Acceptable ≤ {INSTABILITY_ACCEPTABLE} | Unstable > {INSTABILITY_UNSTABLE}")
    print("=" * 70)

    for frac in fractions:
        pct = int(frac * 100)
        sub = df_instability[df_instability["sample_size_percent"] == pct]
        if sub.empty:
            continue
        avg = sub["instability"].mean()
        if avg <= INSTABILITY_STABLE:
            status = "✅ Stable"
        elif avg <= INSTABILITY_ACCEPTABLE:
            status = "⚠️  Acceptable"
        elif avg <= INSTABILITY_UNSTABLE:
            status = "❌ Unstable"
        else:
            status = "❌ Very Unstable"
        print(f"  {pct:3d}%  평균 instability={avg:.3f}  {status}")

    pivot = (df_instability.groupby("sample_size_percent")["instability"]
             .mean().reset_index().sort_values("sample_size_percent"))
    stable_pct = None
    for _, row in pivot.iterrows():
        if row["instability"] <= INSTABILITY_ACCEPTABLE:
            stable_pct = int(row["sample_size_percent"])
            break

    if stable_pct is not None:
        print(f"\n  → 모델 성능이 {stable_pct}% 이상에서 안정화됨 (instability ≤ {INSTABILITY_ACCEPTABLE})")
    else:
        print("\n  → 분석된 범위 내에서 안정화 구간 미달성")
    print("=" * 70)


def save_best_model_extras(best_model, best_model_name, X_test_fs, y_test, selected_features, class_names):
    y_pred = best_model.predict(X_test_fs)
    cm     = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "best_confusion_matrix.png"), dpi=300)
    plt.close()

    perm = permutation_importance(best_model, X_test_fs, y_test,
                                  scoring="f1", n_repeats=100,
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
# LIME 샘플 모드 설정
# "all"     : 전체 test sample
# "correct" : 정분류만
# "wrong"   : 오분류만
# "n"       : 랜덤 n개 (LIME_SAMPLE_N 설정)
# =====================================================
LIME_SAMPLE_MODE = "all"
LIME_SAMPLE_N    = 10   # LIME_SAMPLE_MODE = "n" 일 때만 사용

FEATURE_NAME_MAP = {
    "hip_center_velocity_mean":      "Hip velocity",
    "shoulder_y_symmetry_mean":      "Shoulder vertical symmetry",
    "hip_center_jerk_sd":            "Hip smoothness",
    "hip_center_velocity_sd":        "Hip velocity consistency",
    "left_shoulder_vertical_sway":   "Left shoulder vertical sway",
    "left_hip_velocity_mean":        "Left hip velocity",
    "hip_center_acceleration_sd":    "Hip acceleration consistency",
    "shoulder_center_velocity_mean": "Shoulder velocity",
}


def _lime_select_indices(y_pred, y_test):
    """LIME_SAMPLE_MODE에 따라 대상 샘플 인덱스 반환"""
    correct_idx = np.where(y_pred == y_test)[0]
    wrong_idx   = np.where(y_pred != y_test)[0]

    print(f"  전체 Test 샘플: {len(y_test)}개")
    print(f"  정분류: {len(correct_idx)}개 | 오분류: {len(wrong_idx)}개")

    if LIME_SAMPLE_MODE == "all":
        indices = list(range(len(y_test)))
    elif LIME_SAMPLE_MODE == "correct":
        indices = list(correct_idx)
    elif LIME_SAMPLE_MODE == "wrong":
        indices = list(wrong_idx)
        if len(indices) == 0:
            print("  오분류 샘플 없음 → 전체로 대체")
            indices = list(range(len(y_test)))
    elif LIME_SAMPLE_MODE == "n":
        all_idx = list(range(len(y_test)))
        n       = min(LIME_SAMPLE_N, len(all_idx))
        indices = random.sample(all_idx, n)
    else:
        raise ValueError(f"LIME_SAMPLE_MODE 값이 잘못됨: {LIME_SAMPLE_MODE}")

    print(f"  → LIME 생성 대상: {len(indices)}개 (mode='{LIME_SAMPLE_MODE}')")
    return indices


def xai_lime(best_model, X_train, X_test, y_test, selected_features, class_names,
             sample_ids=None):
    """
    Test data 샘플에 대해 LIME HTML 파일 생성
    LIME_SAMPLE_MODE로 대상 샘플 범위 조정 가능

    저장 위치: result/lime/
    파일명 형식: {sample_id}_true-{true_label}_pred-{pred_label}_{O/X}.html
    """
    print("\n[XAI] LIME")
    try:
        if not hasattr(best_model, "predict_proba"):
            print("  LIME 생략: predict_proba 미지원")
            return

        pretty = [FEATURE_NAME_MAP.get(f, f) for f in selected_features]

        # 스케일링
        has_scaler = "scaler" in best_model.named_steps
        if has_scaler:
            sc     = best_model.named_steps["scaler"]
            X_tr_l = pd.DataFrame(sc.transform(X_train), columns=selected_features)
            X_te_l = pd.DataFrame(sc.transform(X_test),  columns=selected_features)
        else:
            X_tr_l = X_train.copy()
            X_te_l = X_test.copy()

        # sample_ids 설정
        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(X_te_l))]

        # LIME explainer 생성
        explainer = LimeTabularExplainer(
            training_data=X_tr_l.values,
            feature_names=pretty,
            class_names=class_names,
            mode="classification",
            random_state=RANDOM_STATE
        )

        y_pred  = best_model.predict(X_test)
        indices = _lime_select_indices(y_pred, y_test)

        # 저장 폴더
        lime_dir = os.path.join(RESULT_DIR, "lime")
        os.makedirs(lime_dir, exist_ok=True)

        print(f"  처리 중...")
        for rank, idx in enumerate(indices, start=1):
            sid          = sample_ids[idx]
            true_label   = class_names[y_test[idx]]
            pred_label   = class_names[y_pred[idx]]
            correct_mark = "O" if y_test[idx] == y_pred[idx] else "X"

            exp = explainer.explain_instance(
                X_te_l.iloc[idx].values,
                best_model.predict_proba,
                num_features=len(selected_features),
                num_samples=5000
            )

            filename = f"{sid}_true-{true_label}_pred-{pred_label}_{correct_mark}.html"
            exp.save_to_file(os.path.join(lime_dir, filename))
            print(f"  [{rank}/{len(indices)}] {filename}")

        print(f"  저장 완료: result/lime/ ({len(indices)}개 HTML)")

    except Exception as e:
        print(f"  LIME 실패: {e}")


def xai_shap(best_model, X_train, X_test, selected_features):
    print("\n[XAI] SHAP")
    try:
        pretty      = [FEATURE_NAME_MAP.get(f, f) for f in selected_features]
        has_scaler  = "scaler" in best_model.named_steps
        model_step  = best_model.named_steps["model"]
        tree_models = {"DecisionTreeClassifier", "RandomForestClassifier",
                       "GradientBoostingClassifier", "LGBMClassifier",
                       "XGBClassifier", "CatBoostClassifier", "CatBoostClassifierSklearn"}
        is_tree = model_step.__class__.__name__ in tree_models

        X_bg  = X_train.sample(n=min(50, len(X_train)), random_state=RANDOM_STATE)
        X_exp = X_test.iloc[:min(50, len(X_test))].copy().reset_index(drop=True)

        if is_tree:
            X_exp_in = (pd.DataFrame(
                best_model.named_steps["scaler"].transform(X_exp), columns=selected_features
            ) if has_scaler else X_exp.copy())
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
                                    data=data_plot, feature_names=pretty)
        else:
            sc      = best_model.named_steps["scaler"] if has_scaler else None
            X_bg_t  = pd.DataFrame(sc.transform(X_bg),  columns=selected_features) if sc else X_bg.copy()
            X_exp_t = pd.DataFrame(sc.transform(X_exp), columns=selected_features) if sc else X_exp.copy()
            ke = shap.KernelExplainer(model_step.predict_proba, X_bg_t)
            sv = ke.shap_values(X_exp_t, nsamples=100)
            if isinstance(sv, list):
                values   = sv[1]
                base_val = ke.expected_value[1] if isinstance(ke.expected_value, (list, np.ndarray)) else ke.expected_value
            else:
                sv_np    = np.array(sv)
                values   = sv_np[:, :, 1] if sv_np.ndim == 3 else sv_np
                base_val = ke.expected_value[1] if isinstance(ke.expected_value, (list, np.ndarray)) else ke.expected_value
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


def main():
    print("\n" + "=" * 70)
    print("  머신러닝 파이프라인")
    print("=" * 70)

    # 1. 데이터 로드
    X, y, groups, feature_cols, class_names = data_loading()

    # 2. Train/Test 분리
    X_train, X_test, y_train, y_test, groups_train, groups_test, test_idx = data_split(
        X, y, groups, test_size=0.2, random_state=RANDOM_STATE, verbose=True)

    # video id 추출 (파일명 기반, LIME 파일명에 사용)
    # data_loading에서 X의 인덱스는 원본 df 기준이므로 X_test 인덱스로 추출
    video_ids_test = [
        f"{groups[i]}_{y[i]}" for i in test_idx
    ]

    # 3. Feature Selection — Hybrid (mRMR → RFECV)
    selected_features = feature_selection(
        X_train, y_train, feature_cols, min_features=5, verbose=True)

    X_train_fs = X_train[selected_features]
    X_test_fs  = X_test[selected_features]

    # 4. 모델 평가
    models       = build_models()
    results      = []
    y_proba_dict = {}
    best_model, best_model_name, best_f1 = None, None, -999

    for model_name, pipeline in models.items():
        print(f"\n{'─'*55}\n  {model_name}\n{'─'*55}")
        try:
            trained, y_proba, metrics = evaluate_model(
                pipeline, model_name,
                X_train_fs, y_train, X_test_fs, y_test, verbose=True)
            results.append(metrics)
            y_proba_dict[model_name] = y_proba
            if metrics["F1_raw"] > best_f1:
                best_f1, best_model, best_model_name = metrics["F1_raw"], trained, model_name
        except Exception as e:
            print(f"  실패: {e}")

    # 5. 결과 저장
    save_results(results, y_test, y_proba_dict)

    # 6. Sample Size Analysis (Prediction Instability)
    sample_size_analysis(
        X, y, groups, feature_cols,
        fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
        test_size=0.2, min_features=5, n_repeats=100)

    # 7. Best Model 부가 결과
    if best_model is not None:
        print(f"\n{'='*70}")
        print(f"  Best Model: {best_model_name}  F1={best_f1:.2f}")
        print(f"{'='*70}")
        save_best_model_extras(
            best_model, best_model_name,
            X_test_fs, y_test, selected_features, class_names)
        xai_lime(best_model, X_train_fs, X_test_fs, y_test, selected_features, class_names,
                 sample_ids=video_ids_test)
        xai_shap(best_model, X_train_fs, X_test_fs, selected_features)

        joblib.dump(best_model,        os.path.join(RESULT_DIR, "best_model.pkl"))
        joblib.dump(selected_features, os.path.join(RESULT_DIR, "best_features.pkl"))
        print("  모델 저장 완료: best_model.pkl / best_features.pkl")

    print("\n전체 파이프라인 완료!")


if __name__ == "__main__":
    main()