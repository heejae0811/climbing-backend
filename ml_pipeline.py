import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    roc_curve, auc, make_scorer
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer
import joblib


# =====================================================
# 전역 설정
# =====================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

RESULT_DIR = "./result"
os.makedirs(RESULT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


# =====================================================
# scikit-learn 1.8+ tag compatibility for CatBoost
# =====================================================
class CatBoostClassifierSklearn(CatBoostClassifier, ClassifierMixin, BaseEstimator):
    """CatBoost wrapper to provide __sklearn_tags__ via BaseEstimator."""
    pass


# =====================================================
# 1단계: 데이터 불러오기
#   - id, label, total_time은 ML 분석에서 제외
# =====================================================
def data_loading():
    print("\n[1단계] Data Loading")

    files = glob.glob("./features_xlsx_all/*.xlsx")
    print(f"찾은 파일 수: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError("❌ 폴더에 엑셀 파일이 없습니다. ./features_xlsx 폴더 확인 필요")

    df_list = [pd.read_excel(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    if "label" not in df.columns:
        raise KeyError("❌ label 컬럼이 없습니다. 데이터 파일을 확인하세요.")

    # y 라벨
    y = df["label"].astype(int).values
    class_names = ["Advanced", "Intermediate"]
    label_counts = np.bincount(y)

    # 수치형 feature 자동 선택
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # ML에서 제외할 변수
    exclude_features = ["id", "label", "total_time", "body_size_median"]
    feature_cols = [c for c in feature_cols if c not in exclude_features]

    X = df[feature_cols]

    # 결측치 확인
    total_missing = X.isnull().sum().sum()
    print(f"결측치 개수: {total_missing}")

    if total_missing > 0:
        print("⚠️ 경고: 결측치가 존재합니다. (현재 코드는 별도 처리 없이 진행)")

    print(f"최종 Feature 개수: {len(feature_cols)}")
    print(f"클래스 분포: Advanced(0)- {label_counts[0]}개, Intermediate(1) - {label_counts[1]}개")

    return X, y, feature_cols, class_names


# =====================================================
# 2단계: Data Split (Train/Test)
#   - Test 데이터는 이후 모든 과정에서 완전 분리
# =====================================================
def data_split(X, y, test_size=0.2):
    print("\n[2단계] Data Split (Train/Test)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Train 샘플 수: {len(X_train)}")
    print(f"Test  샘플 수: {len(X_test)}")
    print(f"Train 클래스 분포: {np.bincount(y_train)}")
    print(f"Test  클래스 분포: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


# =====================================================
# 3단계: Feature Selection (RFECV)
#   - Train 데이터만 사용
#   - Logistic Regression + MCC 기반
# =====================================================
def feature_selection_rfecv(X_train, y_train, min_features=5):
    print("\n[3단계] Feature Selection (RFECV - Logistic Regression 기반)")

    # 여기서 사용하는 스케일러는 feature 선택용 전용 (ML 스케일러와 별개)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    estimator = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mcc_scorer = make_scorer(matthews_corrcoef)

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring=mcc_scorer,
        min_features_to_select=min_features,
        n_jobs=-1
    )

    rfecv.fit(X_train_scaled, y_train)

    selected_mask = rfecv.support_
    selected_features = X_train.columns[selected_mask].tolist()

    df_sel = pd.DataFrame({"Selected_Features": selected_features})
    selected_path = os.path.join(RESULT_DIR, "selected_features.xlsx")
    df_sel.to_excel(selected_path, index=False)

    print(f"Selected Features: {len(selected_features)}개 \n", selected_features)

    return selected_features


# =====================================================
# Feature Scaling (모델별)
#   - Logistic, KNN, SVM만 스케일링
# =====================================================
def feature_scaling(X_train, X_test, selected_features, model_name):
    scaling_required = ["Logistic Regression", "KNN", "SVM"]

    X_train_fs = X_train[selected_features]
    X_test_fs = X_test[selected_features]

    if model_name not in scaling_required:
        print(f"⚠ 스케일링 생략: {model_name}")
        return X_train_fs, X_test_fs, None

    print(f"✔ 스케일링 적용: {model_name}")
    scaler = StandardScaler()
    scaler.fit(X_train_fs)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_fs),
        columns=selected_features,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_fs),
        columns=selected_features,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


# =====================================================
# 4단계: ML 모델 정의
#   - Base 모델들 정의
# =====================================================
def model_development():
    print("\n[4단계] ML 모델 생성")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            use_label_encoder=False
        ),
        "CatBoost": CatBoostClassifierSklearn(
            random_state=RANDOM_STATE,
            verbose=False
        )
    }

    print(f"모델 개수: {len(models)}개")

    return models


# =====================================================
# 5단계: 모델 학습 및 평가 (Base)
# =====================================================
def model_evaluation(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    metrics = {
        "Model": model_name + " (Base)",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "Balanced_Accuracy": bal_acc,
        "MCC": mcc,
        "AUC": auc_score
    }

    print(f"기본 모델 결과 [{model_name}] | "f"Acc={accuracy:.3f}, F1={f1:.3f}, MCC={mcc:.3f}, AUC={auc_score:.3f}")

    return model, y_pred, y_proba, mcc, metrics


# =====================================================
# GridSearchCV 설정 (모델별)
# =====================================================
def get_param_grid(model_name):
    grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "poly"]
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        },
        "LightGBM": {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.001, 0.01, 0.1]
        },
        "XGBoost": {
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "CatBoost": {
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1]
        }
    }
    return grids.get(model_name, None)


# =====================================================
# GridSearchCV 기반 평가 (Tuned)
#   - scoring = MCC
# =====================================================
def evaluate_with_gridsearch(model, X_train, y_train, X_test, y_test, model_name):
    param_grid = get_param_grid(model_name)

    if param_grid is None:
        print(f"❌ GridSearch 미지원 모델: {model_name}")
        return None, None, None, None, None

    print(f"✔ GridSearchCV 실행: {model_name} (scoring = MCC)")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="matthews_corrcoef",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"{model_name} Best Params: {grid.best_params_}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    metrics = {
        "Model": model_name + " (Tuned)",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "Balanced_Accuracy": bal_acc,
        "MCC": mcc,
        "AUC": auc_score
    }

    print(f"GridSearch 튜닝 결과 [{model_name}] | " f"Acc={accuracy:.3f}, F1={f1:.3f}, MCC={mcc:.3f}, AUC={auc_score:.3f} \n")

    return best_model, y_pred, y_proba, mcc, metrics


# =====================================================
# 6단계: 엑셀 저장 및 시각화
#   - final_results.xlsx
#   - Confusion Matrix (Best Model)
#   - ROC Curve (Base / Tuned 별도 그래프)
#   - Feature Importance 엑셀 + Plot
# =====================================================
def save_results(results_list, y_test, y_proba_base, y_proba_tuned, best_model_name, best_model, X_test_fs, selected_features, class_names):
    print("\n[6단계] 엑셀 저장 및 시각화")

    # 결과 엑셀
    df_results = pd.DataFrame(results_list)
    df_results_sorted = df_results.sort_values("MCC", ascending=False)
    excel_path = os.path.join(RESULT_DIR, "final_results.xlsx")
    df_results_sorted.to_excel(excel_path, index=False)
    print(f"성능 지표 엑셀 저장 완료: {excel_path}")

    # Confusion Matrix (Best)
    y_pred_best = best_model.predict(X_test_fs)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(RESULT_DIR, "best_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix 저장: {cm_path}")

    # ROC Curve (Base Models)
    if len(y_proba_base) > 0:
        plt.figure(figsize=(10, 7))
        for name, y_proba in y_proba_base.items():
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Base Models")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_base_path = os.path.join(RESULT_DIR, "roc_base_models.png")
        plt.savefig(roc_base_path, dpi=300)
        plt.close()
        print(f"ROC(Base) 저장: {roc_base_path}")

    # ROC Curve (Tuned Models)
    if len(y_proba_tuned) > 0:
        plt.figure(figsize=(10, 7))
        for name, y_proba in y_proba_tuned.items():
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Tuned Models (GridSearch)")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_tuned_path = os.path.join(RESULT_DIR, "roc_tuned_models.png")
        plt.savefig(roc_tuned_path, dpi=300)
        plt.close()
        print(f"ROC(Tuned) 저장: {roc_tuned_path}")

    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            best_model,
            X_test_fs,
            y_test,
            scoring="matthews_corrcoef",
            n_repeats=10,
            random_state=RANDOM_STATE
        )
        importances = perm.importances_mean

    df_fi = pd.DataFrame({
        "Feature": selected_features,
        "Importance": importances
    })
    fi_excel_path = os.path.join(RESULT_DIR, "feature_importance.xlsx")
    df_fi.to_excel(fi_excel_path, index=False)
    print(f"Feature Importance 엑셀 저장: {fi_excel_path}")

    plt.figure(figsize=(10, 7))
    df_fi_sorted = df_fi.sort_values("Importance", ascending=False)
    plt.barh(df_fi_sorted["Feature"], df_fi_sorted["Importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    fi_plot_path = os.path.join(RESULT_DIR, "feature_importance_plot.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Feature Importance Plot 저장: {fi_plot_path}")


# =====================================================
# 7단계: XAI (LIME)
# =====================================================
def xai_lime(best_model, X_train_fs, X_test_fs, selected_features, class_names):
    print("\n[7단계] XAI (LIME)")

    try:
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train_fs),
            feature_names=selected_features,
            class_names=class_names,
            mode="classification"
        )

        sample = X_test_fs.iloc[3].values # LIME 예시

        def predict_fn(x):
            return best_model.predict_proba(x)

        exp = explainer.explain_instance(sample, predict_fn)
        lime_path = os.path.join(RESULT_DIR, "best_lime_explanation.html")
        exp.save_to_file(lime_path)
        print(f"LIME 결과 저장: {lime_path}")

    except Exception as e:
        print(f"❌ LIME 실행 실패: {e}")


# =====================================================
# 8단계: 알고리즘 저장 (pkl)
# =====================================================
def save_algorithm(best_model, scaler, selected_features):
    print("\n[8단계] 알고리즘 저장")

    model_path = os.path.join(RESULT_DIR, "best_model.pkl")
    scaler_path = os.path.join(RESULT_DIR, "best_scaler.pkl")
    features_path = os.path.join(RESULT_DIR, "best_features.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(selected_features, features_path)

    print(f"모델 저장 완료: {model_path}")
    print(f"스케일러 저장 완료: {scaler_path}")
    print(f"피처 목록 저장 완료: {features_path}")


# =====================================================
# MAIN: 전체 파이프라인 실행
# =====================================================
def main():
    print("\n============================================")
    print("🚀 머신러닝 파이프라인 시작")
    print("============================================")

    # 1. 데이터 불러오기
    X, y, feature_names, class_names = data_loading()

    # 2. Train/Test 분리 (Test는 이후 과정에서 완전 분리)
    X_train, X_test, y_train, y_test = data_split(X, y)

    # 3. Feature Selection (Train 데이터만 사용)
    selected_features = feature_selection_rfecv(X_train, y_train, min_features=5)

    # 4. 모델 정의
    models = model_development()

    best_model = None
    best_model_name = None
    best_mcc = -999
    best_scaler = None
    best_X_train_fs = None
    best_X_test_fs = None

    results_list = []
    y_proba_base = {}
    y_proba_tuned = {}

    print("\n[5단계] 기본 모델 학습 및 평가 + GridSearch 튜닝")

    for model_name, model in models.items():

        # 4-1. Feature Subset + Scaling (모델별)
        X_train_fs, X_test_fs, scaler = feature_scaling(
            X_train, X_test, selected_features, model_name
        )

        # 5-1. 기본 모델 평가
        base_model, y_pred_base, y_proba_base_model, mcc_base, metrics_base = model_evaluation(
            model, X_train_fs, y_train, X_test_fs, y_test, model_name
        )

        results_list.append(metrics_base)
        y_proba_base[model_name] = y_proba_base_model

        # Best Model 갱신 (Base 기준)
        if mcc_base > best_mcc:
            best_mcc = mcc_base
            best_model = base_model
            best_model_name = model_name + " (Base)"
            best_scaler = scaler
            best_X_train_fs = X_train_fs
            best_X_test_fs = X_test_fs

        # 5-2. GridSearch 튜닝 모델 평가
        tuned_model, _, y_proba_tuned_model, mcc_tuned, metrics_tuned = evaluate_with_gridsearch(
            model, X_train_fs, y_train, X_test_fs, y_test, model_name
        )

        if metrics_tuned is not None:
            results_list.append(metrics_tuned)
            y_proba_tuned[model_name] = y_proba_tuned_model

            if mcc_tuned > best_mcc:
                best_mcc = mcc_tuned
                best_model = tuned_model
                best_model_name = model_name + " (Tuned)"
                best_scaler = scaler
                best_X_train_fs = X_train_fs
                best_X_test_fs = X_test_fs

    print("\n============================================")
    print(f"🏆 최종 Best Model: {best_model_name} | MCC={best_mcc:.3f}")
    print("============================================")

    # 6. 결과 저장 및 시각화
    save_results(
        results_list=results_list,
        y_test=y_test,
        y_proba_base=y_proba_base,
        y_proba_tuned=y_proba_tuned,
        best_model_name=best_model_name,
        best_model=best_model,
        X_test_fs=best_X_test_fs,
        selected_features=selected_features,
        class_names=class_names
    )

    # 7. XAI (LIME)
    xai_lime(best_model, best_X_train_fs, best_X_test_fs, selected_features, class_names)

    # 8. 알고리즘 저장
    save_algorithm(best_model, best_scaler, selected_features)

    print("\n🎉 전체 파이프라인 완료!")


# =====================================================
# MAIN 실행
# =====================================================
if __name__ == "__main__":
    main()
