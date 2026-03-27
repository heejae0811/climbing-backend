import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
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
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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

    files = glob.glob("./features_xlsx/*.xlsx")
    print(f"찾은 파일 수: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError("폴더에 엑셀 파일이 없습니다.")

    dfs = []
    for f in files:
        tmp = pd.read_excel(f)
        basename = os.path.basename(f)
        person_id = basename.split("_")[0]
        tmp["person_id"] = person_id
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    if "label" not in df.columns:
        raise KeyError("label 컬럼이 없습니다.")

    y = df["label"].astype(int).values
    groups = df["person_id"].astype(str).values

    class_names = ["Advanced", "Intermediate"]
    label_counts = np.bincount(y)

    exclude_features = ["id", "label", "total_time", "body_size_median", "person_id"]
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude_features]

    X = df[feature_cols].copy()

    print(f"Feature 개수: {len(feature_cols)}")
    print(f"클래스 분포: Advanced(0)={label_counts[0]}개, Intermediate(1)={label_counts[1]}개")

    return X, y, groups, feature_cols, class_names


# =====================================================
# 2단계: Train / Test 분리 (Group 기반)
# =====================================================
def data_split(X, y, groups, test_size=0.2):
    print("\n[2단계] Train/Test Split (Group 기반, 8:2)")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    overlap = set(np.unique(groups_train)) & set(np.unique(groups_test))
    if overlap:
        raise ValueError(f"동일 인물이 Train/Test 양쪽에 존재: {overlap}")

    print(f"Train: {len(X_train)}개 ({len(np.unique(groups_train))}명) | "
          f"Test: {len(X_test)}개 ({len(np.unique(groups_test))}명)")
    print(f"Train 클래스 분포: {np.bincount(y_train)}")
    print(f"Test  클래스 분포: {np.bincount(y_test)}")
    print("✓ Train/Test 간 인물 중복 없음 확인")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


# =====================================================
# 3단계: Feature Selection
#   Stage 1: 상관관계 기반 Redundancy 제거 (threshold=0.80)
#   Stage 2: RFECV + RandomForest (F1 기준)
# =====================================================
def feature_selection_hybrid(X_train, y_train, groups_train, feature_cols, min_features=5):
    print("\n[3단계] Feature Selection")

    # ── Stage 1: 상관관계 기반 Redundancy 제거 ───────────────────
    print("  [Stage 1] 상관관계 제거 (threshold=0.80)")

    corr_matrix = X_train[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.80)]
    stage1_features = [f for f in feature_cols if f not in drop_cols]

    print(f"  {len(feature_cols)}개 → 상관관계 제거 후: {len(stage1_features)}개")
    print(f"  제거된 Feature: {drop_cols}")

    if len(stage1_features) < min_features:
        stage1_features = feature_cols

    # ── Stage 2: RFECV + RandomForest (F1 기준) ──────────────────
    print(f"\n  [Stage 2] RFECV + RandomForest (입력: {len(stage1_features)}개, F1 기준)")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

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

    selected_features = [f for f, s in zip(stage1_features, rfecv.support_) if s]

    print(f"  최종 선택: {len(selected_features)}개")
    print(f"  Selected Features: {selected_features}")

    return selected_features


# =====================================================
# 4단계: 모델 정의
# =====================================================
def build_models():
    print("\n[4단계] 모델 정의 (10개, Pipeline 기반)")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=RANDOM_STATE))
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
            ("model", RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=-1))
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

    for name, pipe in models.items():
        has_scaler = "scaler" in pipe.named_steps
        print(f"  {name}: {'스케일링 O' if has_scaler else '스케일링 X'}")

    return models


# =====================================================
# 평가 지표
# =====================================================
def compute_metrics(model_label, y_test, y_pred, y_proba):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
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
def evaluate_model(pipeline, model_name, X_train, y_train, X_test, y_test):
    model = clone(pipeline)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    metrics = compute_metrics(model_name, y_test, y_pred, y_proba)

    print(f"  Acc={metrics['Accuracy']:.2f} | "
          f"F1={metrics['F1']:.2f} | MCC={metrics['MCC']:.2f} | "
          f"AUC={metrics['AUC'] if pd.notnull(metrics['AUC']) else 'nan'}")

    return model, y_proba, metrics


# =====================================================
# 6단계: 결과 저장
# =====================================================
def save_results(results_list, y_test, y_proba_dict,
                 best_model_name, best_model, X_test, selected_features, class_names):
    print("\n[6단계] 결과 저장 및 시각화")

    # 엑셀 저장용: raw 제거하고 반올림된 지표만 저장
    excel_columns = [
        "Model", "Accuracy", "Precision", "Recall", "F1",
        "Specificity", "Sensitivity", "Balanced_Accuracy", "MCC", "AUC"
    ]
    df_results = pd.DataFrame(results_list)
    df_results_excel = df_results[excel_columns].sort_values("F1", ascending=False)
    df_results_excel.to_excel(os.path.join(RESULT_DIR, "final_results.xlsx"), index=False)

    # 이미지: Best model Confusion Matrix
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "best_confusion_matrix.png"), dpi=300)
    plt.close()

    # 이미지: ROC Curve
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

    # 엑셀 + 이미지: Permutation Feature Importance (F1 기준)
    perm = permutation_importance(
        best_model, X_test, y_test,
        scoring="f1", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    df_fi = pd.DataFrame({
        "Feature": selected_features,
        "Importance": np.round(perm.importances_mean, 2),
        "Std": np.round(perm.importances_std, 2)
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

    print("결과 저장 완료")


# =====================================================
# 7단계: LIME
# =====================================================
def xai_lime(best_model, X_train, X_test, y_test, selected_features, class_names):
    print("\n[7단계] XAI (LIME)")

    try:
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=selected_features,
            class_names=class_names,
            mode="classification",
            random_state=RANDOM_STATE
        )

        y_pred = best_model.predict(X_test)
        misclassified = np.where(y_pred != y_test)[0]
        sample_idx = misclassified[0] if len(misclassified) > 0 else 0

        exp = explainer.explain_instance(
            X_test.iloc[sample_idx].values,
            best_model.predict_proba,
            num_samples=5000
        )

        exp.save_to_file(os.path.join(RESULT_DIR, "lime_explanation.html"))

        fig = exp.as_pyplot_figure()
        fig.set_size_inches(8, 6)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULT_DIR, "lime_explanation.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("LIME HTML + PNG 저장 완료")

    except Exception as e:
        print(f"LIME 실행 실패: {e}")


# =====================================================
# 7-추가: SHAP
# =====================================================
def xai_shap(best_model, best_model_name, X_train, X_test, selected_features, class_names):
    print("\n[7-추가] XAI (SHAP)")

    try:
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
            print("  SHAP mode: TreeExplainer")

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

            if raw.ndim == 3:
                values_for_plot = shap.Explanation(
                    values=raw[:, :, 1],
                    base_values=shap_obj.base_values[:, 1] if np.ndim(shap_obj.base_values) > 1 else shap_obj.base_values,
                    data=shap_obj.data,
                    feature_names=selected_features
                )
            else:
                values_for_plot = shap_obj

        else:
            print("  SHAP mode: KernelExplainer")

            if has_scaler:
                scaler = best_model.named_steps["scaler"]
                X_background_trans = pd.DataFrame(scaler.transform(X_background), columns=selected_features)
                X_explain_trans = pd.DataFrame(scaler.transform(X_explain), columns=selected_features)
            else:
                X_background_trans = X_background.copy()
                X_explain_trans = X_explain.copy()

            explainer = shap.KernelExplainer(model_step.predict_proba, X_background_trans)
            raw_sv = explainer.shap_values(X_explain_trans, nsamples=100)

            if isinstance(raw_sv, list):
                values = raw_sv[1]
                base_value = explainer.expected_value[1] \
                    if isinstance(explainer.expected_value, (list, np.ndarray)) \
                    else explainer.expected_value
            else:
                sv_np = np.array(raw_sv)
                if sv_np.ndim == 3:
                    values = sv_np[:, :, 1]
                    base_value = explainer.expected_value[1] \
                        if isinstance(explainer.expected_value, (list, np.ndarray)) \
                        else explainer.expected_value
                else:
                    values = sv_np
                    base_value = explainer.expected_value

            values_for_plot = shap.Explanation(
                values=values,
                base_values=np.repeat(base_value, X_explain_trans.shape[0]),
                data=X_explain_trans.values,
                feature_names=selected_features
            )

        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(values_for_plot, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "shap_beeswarm.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        print("  shap_beeswarm.png 저장 완료")

    except Exception as e:
        print(f"SHAP 실행 실패: {e}")


# =====================================================
# 8단계: 모델 저장
# =====================================================
def save_model(best_model, selected_features):
    print("\n[8단계] 모델 저장")
    joblib.dump(best_model, os.path.join(RESULT_DIR, "best_model.pkl"))
    joblib.dump(selected_features, os.path.join(RESULT_DIR, "best_features.pkl"))
    print("저장 완료: best_model.pkl, best_features.pkl")


# =====================================================
# MAIN
# =====================================================
def main():
    print("\n" + "=" * 60)
    print("  머신러닝 파이프라인 시작")
    print("=" * 60)

    X, y, groups, feature_cols, class_names = data_loading()

    X_train, X_test, y_train, y_test, groups_train, groups_test = data_split(X, y, groups)

    selected_features = feature_selection_hybrid(X_train, y_train, groups_train, feature_cols, min_features=5)
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

        trained_model, y_proba, metrics = evaluate_model(
            pipeline, model_name, X_train_fs, y_train, X_test_fs, y_test
        )
        results_list.append(metrics)
        y_proba_dict[model_name] = y_proba

        if metrics["F1_raw"] > best_f1_raw:
            best_f1_raw = metrics["F1_raw"]
            best_model = trained_model
            best_model_name = model_name

    print("\n" + "=" * 60)
    print(f"  Best Model: {best_model_name} | F1(raw)={best_f1_raw:.4f}")
    print("=" * 60)

    save_results(results_list, y_test, y_proba_dict, best_model_name, best_model, X_test_fs, selected_features, class_names)

    xai_lime(best_model, X_train_fs, X_test_fs, y_test, selected_features, class_names)
    xai_shap(best_model, best_model_name, X_train_fs, X_test_fs, selected_features, class_names)

    save_model(best_model, selected_features)

    print("\n전체 파이프라인 완료!")


if __name__ == "__main__":
    main()