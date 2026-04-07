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
from sklearn.feature_selection import RFECV, f_classif
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
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from lime.lime_tabular import LimeTabularExplainer
from statsmodels.stats.multitest import multipletests
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


# =====================================================
# scikit-learn tag compatibility for CatBoost
# =====================================================
class CatBoostClassifierSklearn(CatBoostClassifier, ClassifierMixin, BaseEstimator):
    pass


# =====================================================
# 1단계: 데이터 불러오기 (CSV)
# 조건:
# 1) 모든 파일에 공통으로 존재하는 numeric feature만 사용
# 2) NaN이 단 하나의 파일에라도 있으면 해당 변수는 전체에서 제외
# 3) Inf가 단 하나의 파일에라도 있으면 해당 변수는 전체에서 제외
# 4) body_size_median 제외
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
        print("\n[추가 안전 제거 변수]")
        print(extra_remove)
        valid_feature_cols = [c for c in valid_feature_cols if c not in extra_remove]
        X = df[valid_feature_cols].apply(pd.to_numeric, errors="coerce")

    y = df["label"].astype(int).values
    groups = df["person_id"].astype(str).values

    class_names = ["Advanced", "Intermediate"]
    label_counts = np.bincount(y)

    print(f"공통 numeric feature: {len(common_feature_cols)}")
    print(f"NaN 때문에 제거된 feature: {len(removed_nan_cols)}")
    print(f"Inf 때문에 제거된 feature: {len(removed_inf_cols)}")
    print(f"최종 사용 feature: {len(valid_feature_cols)}")
    print(f"전체 participant 수: {len(np.unique(groups))}명")
    print(f"클래스 분포: Advanced(0)={label_counts[0]}개, Intermediate(1)={label_counts[1]}개")

    if removed_nan_cols:
        print("\n[NaN 포함으로 제거된 feature]")
        print(removed_nan_cols)

    if removed_inf_cols:
        print("\n[Inf 포함으로 제거된 feature]")
        print(removed_inf_cols)

    return X, y, groups, valid_feature_cols, class_names


# =====================================================
# 2단계: Train / Test 분리 (Group 기반)
# =====================================================
def data_split(X, y, groups, test_size=0.2, random_state=RANDOM_STATE, verbose=True):
    if verbose:
        print("\n[2단계] Train/Test Split (Group 기반, 8:2)")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

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
# 3단계: Feature Selection
# Stage 1: Correlation filtering
# Stage 2: RFECV + RandomForest
# =====================================================
def feature_selection_hybrid(X_train, y_train, groups_train, feature_cols, min_features=5, verbose=True):
    """
    mHealth 기반 Feature Selection
    Step 1: Correlation filtering (|r| > 0.80)
    Step 2: RFECV + RandomForest (F1 기준)
    """

    if verbose:
        print("\n[3단계] Feature Selection")
        print("  [Stage 1] Correlation filtering (threshold=0.80)")

    # -------------------------------------------------
    # Stage 1: Correlation filtering
    # -------------------------------------------------
    X_stage1 = X_train[feature_cols].copy()

    corr_matrix = X_stage1.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop_cols = [col for col in upper.columns if any(upper[col] > 0.80)]
    stage1_features = [f for f in feature_cols if f not in drop_cols]

    # 최소 feature 보장
    if len(stage1_features) < min_features:
        if verbose:
            print("  feature 수가 부족하여 일부 변수 유지")
        stage1_features = feature_cols

    if verbose:
        print(f"  입력 feature: {len(feature_cols)}개")
        print(f"  Stage 1 selected: {len(stage1_features)}개")
        print(f"  제거된 feature: {drop_cols}")

    # -------------------------------------------------
    # Stage 2: RFECV + RandomForest
    # -------------------------------------------------
    if verbose:
        print(f"\n  [Stage 2] RFECV + RandomForest (입력: {len(stage1_features)}개, F1 기준)")

    unique_groups = np.unique(groups_train)
    n_groups = len(unique_groups)
    n_splits = min(5, n_groups)

    # 그룹 수 부족하면 RFECV 생략
    if n_splits < 2:
        if verbose:
            print("  그룹 수 부족 → RFECV 생략, Stage 1 결과 사용")
        return stage1_features[:max(min_features, len(stage1_features))]

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="f1",
        min_features_to_select=min_features,
        n_jobs=-1
    )

    rfecv.fit(X_train[stage1_features], y_train, groups=groups_train)

    final_features = [
        f for f, s in zip(stage1_features, rfecv.support_) if s
    ]

    # 최소 feature 보장
    if len(final_features) < min_features:
        final_features = stage1_features[:max(min_features, len(stage1_features))]

    if verbose:
        print(f"  최종 선택: {len(final_features)}개")
        print(f"  Selected Features: {final_features}")

    return final_features


# =====================================================
# 4단계: 모델 정의
# =====================================================
def build_models():
    models = {
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
    return models


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

    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        auc_value = roc_auc_score(y_test, y_proba)
    except Exception:
        auc_value = np.nan

    return {
        "Model": model_label,
        "Accuracy_raw": accuracy,
        "Precision_raw": precision,
        "Recall_raw": recall,
        "F1_raw": f1,
        "Specificity_raw": specificity,
        "Sensitivity_raw": sensitivity,
        "Balanced_Accuracy_raw": balanced_acc,
        "MCC_raw": mcc,
        "AUC_raw": auc_value,

        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2),
        "Specificity": round(specificity, 2),
        "Sensitivity": round(sensitivity, 2),
        "Balanced_Accuracy": round(balanced_acc, 2),
        "MCC": round(mcc, 2),
        "AUC": round(auc_value, 2) if pd.notnull(auc_value) else np.nan
    }


# =====================================================
# 5단계: 모델 학습 및 평가
# =====================================================
def evaluate_model(pipeline, model_name, X_train, y_train, X_test, y_test, verbose=True):
    model = clone(pipeline)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    metrics = compute_metrics(model_name, y_test, y_pred, y_proba)

    if verbose:
        print(f"  Acc={metrics['Accuracy']:.2f} | "
              f"F1={metrics['F1']:.2f} | "
              f"MCC={metrics['MCC']:.2f} | "
              f"AUC={metrics['AUC'] if pd.notnull(metrics['AUC']) else 'nan'}")

    return model, y_proba, metrics


# =====================================================
# 6단계: 최종 결과 저장
# =====================================================
def save_final_results(results_list, y_test, y_proba_dict,
                       best_model_name, best_model, X_test, selected_features, class_names):
    print("\n[6단계] 최종 결과 저장 및 시각화")

    excel_columns = [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "Specificity", "Sensitivity", "Balanced_Accuracy", "MCC", "AUC"
    ]
    df_results = pd.DataFrame(results_list)
    df_results_excel = df_results[excel_columns].sort_values("F1", ascending=False)
    df_results_excel.to_excel(os.path.join(RESULT_DIR, "final_results.xlsx"), index=False)

    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "best_confusion_matrix.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, y_proba in y_proba_dict.items():
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    perm = permutation_importance(
        best_model, X_test, y_test,
        scoring="f1", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    df_fi = pd.DataFrame({
        "Feature": selected_features,
        "Importance": np.round(perm.importances_mean, 4),
        "Std": np.round(perm.importances_std, 4)
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

    print("최종 결과 저장 완료")


# =====================================================
# XAI: LIME
# =====================================================
def xai_lime(best_model, X_train, X_test, y_test, selected_features, class_names):
    print("\n[7단계] XAI (LIME)")

    try:
        if not hasattr(best_model, "predict_proba"):
            print("LIME 생략: predict_proba 미지원 모델")
            return

        # ---------------------------------------------
        # 1. 사용자 친화적 변수명 매핑
        # ---------------------------------------------
        feature_name_map = {
            "com_acceleration_sd": "Hip center acceleration",
            "shoulder_y_symmetry_mean": "Shoulder vertical symmetry",
            "com_gie": "Hip center efficiency",
            "com_acceleration_max": "Hip center maximum acceleration",
            "hip_x_symmetry_max": "Hip horizontal symmetry",
            "shoulder_y_symmetry_max": "Shoulder maximum vertical symmetry",
            "right_wrist_acceleration_max": "Right wrist maximum acceleration",
            "shoulder_x_symmetry_mean": "Shoulder horizontal symmetry",
            "shoulder_x_symmetry_max": "Shoulder maximum horizontal symmetry"
        }

        pretty_features = [feature_name_map.get(f, f) for f in selected_features]

        # ---------------------------------------------
        # 2. scaler 유무 확인
        #    pipeline 안에 scaler가 있으면 변환된 값으로 LIME 수행
        # ---------------------------------------------
        has_scaler = "scaler" in best_model.named_steps

        if has_scaler:
            scaler = best_model.named_steps["scaler"]
            X_train_lime = pd.DataFrame(
                scaler.transform(X_train),
                columns=selected_features
            )
            X_test_lime = pd.DataFrame(
                scaler.transform(X_test),
                columns=selected_features
            )
        else:
            X_train_lime = X_train.copy()
            X_test_lime = X_test.copy()

        # ---------------------------------------------
        # 3. LIME explainer 생성
        # ---------------------------------------------
        explainer = LimeTabularExplainer(
            training_data=X_train_lime.values,
            feature_names=pretty_features,
            class_names=class_names,
            mode="classification",
            random_state=RANDOM_STATE
        )

        # ---------------------------------------------
        # 4. 샘플 선택
        #    오분류 샘플 우선, 없으면 첫 번째 샘플
        # ---------------------------------------------
        y_pred = best_model.predict(X_test)
        misclassified = np.where(y_pred != y_test)[0]
        sample_idx = misclassified[0] if len(misclassified) > 0 else 0

        # ---------------------------------------------
        # 5. LIME 설명 생성
        # ---------------------------------------------
        exp = explainer.explain_instance(
            X_test_lime.iloc[sample_idx].values,
            best_model.predict_proba,
            num_features=min(10, len(selected_features)),
            num_samples=5000
        )

        # ---------------------------------------------
        # 6. HTML 저장
        # ---------------------------------------------
        exp.save_to_file(os.path.join(RESULT_DIR, "lime_explanation.html"))

        # ---------------------------------------------
        # 7. PNG 저장
        # ---------------------------------------------
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(8, 6)
        fig.tight_layout()
        fig.savefig(
            os.path.join(RESULT_DIR, "lime_explanation.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

        # ---------------------------------------------
        # 8. 텍스트 결과 저장
        # ---------------------------------------------
        lime_list = exp.as_list()
        lime_df = pd.DataFrame(lime_list, columns=["Feature", "Contribution"])
        lime_df.to_excel(os.path.join(RESULT_DIR, "lime_explanation.xlsx"), index=False)

        print(f"LIME sample index: {sample_idx}")
        print("LIME HTML + PNG + XLSX 저장 완료")

    except Exception as e:
        print(f"LIME 실행 실패: {e}")

# =====================================================
# XAI: SHAP
# =====================================================
def xai_shap(best_model, best_model_name, X_train, X_test, selected_features):
    print("\n[7-추가] XAI (SHAP)")

    try:
        feature_name_map = {
            "com_acceleration_sd": "Hip center acceleration",
            "shoulder_y_symmetry_mean": "Shoulder vertical symmetry",
            "com_gie": "Hip center efficiency",
            "com_acceleration_max": "Hip center maximum acceleration",
            "hip_x_symmetry_max": "Hip horizontal symmetry",
            "shoulder_y_symmetry_max": "Shoulder maximum vertical symmetry",
            "right_wrist_acceleration_max": "Right wrist maximum acceleration",
            "shoulder_x_symmetry_mean": "Shoulder horizontal symmetry",
            "shoulder_x_symmetry_max": "Shoulder maximum horizontal symmetry"
        }
        pretty_features = [feature_name_map.get(f, f) for f in selected_features]

        X_background = X_train.sample(n=min(50, len(X_train)), random_state=RANDOM_STATE)
        X_explain = X_test.iloc[:min(50, len(X_test))].copy().reset_index(drop=True)

        has_scaler = "scaler" in best_model.named_steps
        model_step = best_model.named_steps["model"]

        tree_model_names = {
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LGBMClassifier",
            "XGBClassifier",
            "CatBoostClassifier",
            "CatBoostClassifierSklearn"
        }
        is_tree_model = model_step.__class__.__name__ in tree_model_names

        if is_tree_model:
            if has_scaler:
                scaler = best_model.named_steps["scaler"]
                X_explain_input = pd.DataFrame(
                    scaler.transform(X_explain),
                    columns=selected_features
                )
            else:
                X_explain_input = X_explain.copy()

            explainer = shap.TreeExplainer(model_step)
            shap_obj = explainer(X_explain_input)

            raw = shap_obj.values
            base = shap_obj.base_values
            data_for_plot = shap_obj.data if shap_obj.data is not None else X_explain_input.values

            if raw.ndim == 3:
                values = raw[:, :, 1]
                if np.ndim(base) > 1:
                    base_values = base[:, 1]
                else:
                    base_values = base
            else:
                values = raw
                base_values = base

            values_for_plot = shap.Explanation(
                values=values,
                base_values=base_values,
                data=data_for_plot,
                feature_names=pretty_features
            )

        else:
            if has_scaler:
                scaler = best_model.named_steps["scaler"]
                X_background_trans = pd.DataFrame(
                    scaler.transform(X_background),
                    columns=selected_features
                )
                X_explain_trans = pd.DataFrame(
                    scaler.transform(X_explain),
                    columns=selected_features
                )
            else:
                X_background_trans = X_background.copy()
                X_explain_trans = X_explain.copy()

            explainer = shap.KernelExplainer(model_step.predict_proba, X_background_trans)
            raw_sv = explainer.shap_values(X_explain_trans, nsamples=100)

            if isinstance(raw_sv, list):
                values = raw_sv[1]
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                sv_np = np.array(raw_sv)
                if sv_np.ndim == 3:
                    values = sv_np[:, :, 1]
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    values = sv_np
                    base_value = explainer.expected_value

            values_for_plot = shap.Explanation(
                values=values,
                base_values=np.repeat(base_value, X_explain_trans.shape[0]),
                data=X_explain_trans.values,
                feature_names=pretty_features
            )

        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(values_for_plot, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "shap_beeswarm.png"), dpi=300, bbox_inches="tight")
        plt.close()

        print("shap_beeswarm.png 저장 완료")

    except Exception as e:
        print(f"SHAP 실행 실패: {e}")


# =====================================================
# 모델 저장
# =====================================================
def save_model(best_model, selected_features):
    print("\n[8단계] 모델 저장")
    joblib.dump(best_model, os.path.join(RESULT_DIR, "best_model.pkl"))
    joblib.dump(selected_features, os.path.join(RESULT_DIR, "best_features.pkl"))
    print("저장 완료: best_model.pkl, best_features.pkl")


# =====================================================
# subset 유효성 확인
# =====================================================
def is_valid_subset(y_sub, groups_sub):
    if len(np.unique(y_sub)) < 2:
        return False
    if len(np.unique(groups_sub)) < 4:
        return False
    return True


# =====================================================
# group 단위 stratified subset 생성
# =====================================================
def subset_by_group_fraction_stratified(X, y, groups, fraction, random_state=42):
    df_groups = pd.DataFrame({
        "group": groups,
        "label": y
    })

    group_label_df = df_groups.groupby("group")["label"].first().reset_index()
    unique_labels = sorted(group_label_df["label"].unique())

    selected_groups = []
    rng = np.random.RandomState(random_state)

    for label in unique_labels:
        label_groups = group_label_df[group_label_df["label"] == label]["group"].values
        n_total = len(label_groups)
        n_select = max(1, int(round(n_total * fraction)))
        n_select = min(n_select, n_total)

        chosen = rng.choice(label_groups, size=n_select, replace=False)
        selected_groups.extend(chosen.tolist())

    selected_groups = np.array(sorted(selected_groups))
    mask = np.isin(groups, selected_groups)

    X_sub = X.loc[mask].reset_index(drop=True)
    y_sub = y[mask]
    groups_sub = groups[mask]

    return X_sub, y_sub, groups_sub


# =====================================================
# mHealth 방식 pilot
# - 10/30/50/70/90%
# - 각 비율당 1회
# - subset 생성 -> split -> feature selection -> 10개 모델 평가
# - sample size figure는 딱 2개만 저장
# =====================================================
def sample_size_analysis_pilot(
    X, y, groups, feature_cols,
    fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
    test_size=0.2,
    min_features=5,
    n_repeats=1
):
    all_rows = []
    models = build_models()
    model_names = list(models.keys())

    for frac in fractions:
        print("\n" + "-" * 70)
        print(f"[샘플 비율 {int(frac * 100)}%]")
        print("-" * 70)

        for repeat_idx in range(n_repeats):
            seed = RANDOM_STATE + repeat_idx

            print(f"  반복 {repeat_idx+1}/{n_repeats}")

            # 1) subset 생성
            X_sub, y_sub, groups_sub = subset_by_group_fraction_stratified(
                X, y, groups, fraction=frac, random_state=seed
            )

            if not is_valid_subset(y_sub, groups_sub):
                print("    subset invalid → skip")
                continue

            # 2) subset 내부 train/test split
            X_train_sub, X_test_sub, y_train_sub, y_test_sub, groups_train_sub, groups_test_sub = data_split(
                X_sub, y_sub, groups_sub, test_size=test_size, random_state=seed, verbose=False
            )

            if len(np.unique(y_train_sub)) < 2 or len(np.unique(y_test_sub)) < 2:
                print("    train/test class 부족 → skip")
                continue

            # 3) feature selection 다시 수행
            selected_features_sub = feature_selection_hybrid(
                X_train_sub, y_train_sub, groups_train_sub,
                feature_cols, min_features=min_features, verbose=False
            )

            X_train_fs = X_train_sub[selected_features_sub]
            X_test_fs = X_test_sub[selected_features_sub]

            # 4) 모든 모델 평가
            for model_name in model_names:
                pipeline = models[model_name]

                try:
                    _, _, metrics = evaluate_model(
                        pipeline, model_name,
                        X_train_fs, y_train_sub,
                        X_test_fs, y_test_sub,
                        verbose=False
                    )

                    all_rows.append({
                        "sample_size_percent": int(frac * 100),
                        "repeat": repeat_idx + 1,
                        "participant_count": len(np.unique(groups_sub)),
                        "video_count": len(X_sub),
                        "model": model_name,
                        "accuracy": metrics["Accuracy_raw"],
                        "f1": metrics["F1_raw"],
                        "auc": metrics["AUC_raw"] if pd.notnull(metrics["AUC_raw"]) else np.nan
                    })

                except Exception as e:
                    print(f"    {model_name} 실패: {e}")

    df_sample = pd.DataFrame(all_rows)

    if df_sample.empty:
        print("sample size 분석 결과가 없습니다.")
        return df_sample

    # raw 저장
    df_sample.to_excel(os.path.join(RESULT_DIR, "sample_size_plot_raw.xlsx"), index=False)

    # 평균/표준편차 집계
    df_summary = (
        df_sample
        .groupby(["sample_size_percent", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
            participant_count_mean=("participant_count", "mean"),
            video_count_mean=("video_count", "mean")
        )
    )

    df_summary.to_excel(os.path.join(RESULT_DIR, "sample_size_plot_summary.xlsx"), index=False)

    # Figure 1: Accuracy mean ± sd
    plt.figure(figsize=(12, 8))
    for model_name in model_names:
        tmp = df_summary[df_summary["model"] == model_name].sort_values("sample_size_percent")
        if len(tmp) == 0:
            continue

        plt.errorbar(
            tmp["sample_size_percent"],
            tmp["accuracy_mean"],
            yerr=tmp["accuracy_sd"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=model_name
        )

    plt.xlabel("Sample size (%)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Model accuracy across increasing sample sizes", fontsize=16)
    plt.xticks([10, 30, 50, 70, 90])
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "sample_size_accuracy_plot.png"), dpi=300)
    plt.close()

    # Figure 2: F1 mean ± sd
    plt.figure(figsize=(12, 8))
    for model_name in model_names:
        tmp = df_summary[df_summary["model"] == model_name].sort_values("sample_size_percent")
        if len(tmp) == 0:
            continue

        plt.errorbar(
            tmp["sample_size_percent"],
            tmp["f1_mean"],
            yerr=tmp["f1_sd"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=model_name
        )

    plt.xlabel("Sample size (%)", fontsize=14)
    plt.ylabel("F1-score", fontsize=14)
    plt.title("Model F1-score across increasing sample sizes", fontsize=16)
    plt.xticks([10, 30, 50, 70, 90])
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "sample_size_f1_plot.png"), dpi=300)
    plt.close()

    print("\n[sample size 반복 분석 요약]")
    print(df_summary)

    return df_summary


# =====================================================
# MAIN
# =====================================================
def main():
    print("\n" + "=" * 60)
    print("  머신러닝 파이프라인 시작")
    print("=" * 60)

    # 전체 데이터 로드
    X, y, groups, feature_cols, class_names = data_loading()

    # -------------------------------------------------
    # 추가 분석: mHealth 방식 pilot sample size analysis
    # -------------------------------------------------
    df_sample = sample_size_analysis_pilot(
        X=X,
        y=y,
        groups=groups,
        feature_cols=feature_cols,
        fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
        test_size=0.2,
        min_features=5,
        n_repeats=1
    )

    print("\n[샘플 크기 분석 결과 요약]")
    print(df_sample)

    # -------------------------------------------------
    # 기존 전체 데이터 최종 모델링
    # -------------------------------------------------
    X_train, X_test, y_train, y_test, groups_train, groups_test = data_split(
        X, y, groups, test_size=0.2, random_state=RANDOM_STATE, verbose=True
    )

    selected_features = feature_selection_hybrid(
        X_train, y_train, groups_train, feature_cols, min_features=5, verbose=True
    )
    X_train_fs = X_train[selected_features]
    X_test_fs = X_test[selected_features]

    models = build_models()

    print("\n[5단계] 모델 학습 및 평가")
    print(f"  사용 Feature: {len(selected_features)}개")

    results_list = []
    y_proba_dict = {}
    best_model = None
    best_model_name = None
    best_f1_raw = -999

    for model_name, pipeline in models.items():
        print(f"\n{'─'*50}\n  {model_name}\n{'─'*50}")

        try:
            trained_model, y_proba, metrics = evaluate_model(
                pipeline, model_name,
                X_train_fs, y_train,
                X_test_fs, y_test,
                verbose=True
            )
            results_list.append(metrics)
            y_proba_dict[model_name] = y_proba

            if metrics["F1_raw"] > best_f1_raw:
                best_f1_raw = metrics["F1_raw"]
                best_model = trained_model
                best_model_name = model_name

        except Exception as e:
            print(f"{model_name} 실패: {e}")

    if best_model is None:
        raise RuntimeError("모든 모델 학습이 실패했습니다.")

    print("\n" + "=" * 60)
    print(f"  Best Model: {best_model_name} | F1(raw)={best_f1_raw:.4f}")
    print("=" * 60)

    save_final_results(
        results_list, y_test, y_proba_dict,
        best_model_name, best_model, X_test_fs, selected_features, class_names
    )

    xai_lime(best_model, X_train_fs, X_test_fs, y_test, selected_features, class_names)
    xai_shap(best_model, best_model_name, X_train_fs, X_test_fs, selected_features)
    save_model(best_model, selected_features)

    print("\n전체 파이프라인 완료!")


if __name__ == "__main__":
    main()