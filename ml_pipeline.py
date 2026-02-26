import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
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
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
#   - id, label, total_time, body_size_median 제외
#   - 수치형 feature만 자동 선택
# =====================================================
def data_loading():
    print("\n[1단계] Data Loading")

    files = glob.glob("./features_xlsx/*.xlsx")
    print(f"찾은 파일 수: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError("폴더에 엑셀 파일이 없습니다.")

    df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)

    if "label" not in df.columns:
        raise KeyError("label 컬럼이 없습니다.")

    y = df["label"].astype(int).values
    class_names = ["Advanced", "Intermediate"]
    label_counts = np.bincount(y)

    exclude_features = ["id", "label", "total_time", "body_size_median"]
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude_features]

    X = df[feature_cols]

    total_missing = X.isnull().sum().sum()
    print(f"결측치 개수: {total_missing}")
    if total_missing > 0:
        print("경고: 결측치가 존재합니다.")

    print(f"Feature 개수: {len(feature_cols)}")
    print(f"클래스 분포: Advanced(0)={label_counts[0]}개, Intermediate(1)={label_counts[1]}개")

    return X, y, feature_cols, class_names


# =====================================================
# 2단계: Train / Test 분리
#   - stratify로 클래스 비율 유지
#   - Test는 최종 평가에만 사용, 학습/튜닝에 절대 미사용
# =====================================================
def data_split(X, y, test_size=0.2):
    print("\n[2단계] Train/Test Split (80:20, Stratified)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Train: {len(X_train)}개  |  Test: {len(X_test)}개")
    print(f"Train 클래스 분포: {np.bincount(y_train)}")
    print(f"Test  클래스 분포: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


# =====================================================
# 3단계: Feature Selection
#   [방법] RF Embedded + Permutation Importance
#   [CV]   Train 내 5-fold Stratified CV
#   [기준] MCC (모델 학습/튜닝 기준과 통일)
#   [선택] 5-fold importance 평균 >= 전체 평균 (threshold)
#   [보장] Train 데이터만 사용, Test leakage 없음
# =====================================================
def feature_selection(X_train, y_train, min_features=5):
    print("\n[3단계] Feature Selection (RF + Permutation Importance + 5-Fold CV + MCC)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    importance_matrix = np.zeros((5, X_train.shape[1]))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_val   = y_train[val_idx]

        # RF를 Train fold로 학습
        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X_fold_train, y_fold_train)

        # Validation fold에서 MCC 기준 permutation importance 계산
        perm = permutation_importance(
            rf, X_fold_val, y_fold_val,
            scoring="matthews_corrcoef",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        importance_matrix[fold_idx] = perm.importances_mean

    # 5-fold importance 평균
    mean_importances = importance_matrix.mean(axis=0)

    df_importance = pd.DataFrame({
        "Feature":    X_train.columns,
        "Importance": mean_importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # threshold: 전체 importance 평균 이상인 feature 선택
    threshold = mean_importances.mean()
    selected  = df_importance[df_importance["Importance"] >= threshold]
    if len(selected) < min_features:
        selected = df_importance.head(min_features)

    selected_features = selected["Feature"].tolist()

    print(f"Importance threshold: {threshold:.4f}")
    print(f"Selected Features ({len(selected_features)}개): {selected_features}")
    print(df_importance.to_string(index=False))

    df_importance.to_excel(
        os.path.join(RESULT_DIR, "feature_selection_importance.xlsx"), index=False)
    pd.DataFrame({"Selected_Features": selected_features}).to_excel(
        os.path.join(RESULT_DIR, "selected_features.xlsx"), index=False)

    return selected_features


# =====================================================
# 4단계: 모델 정의 (8개)
#   - 스케일링 필요(LR, KNN, SVM): Pipeline에 StandardScaler 포함
#   - 스케일링 불필요(DT, RF, LGBM, XGB, CAT): StandardScaler 미포함
#   - Pipeline으로 통일 → scaler와 model 항상 세트
#   - clone()으로 Base/GridSearch/Optuna 독립 학습 보장
# =====================================================
def build_models():
    print("\n[4단계] 모델 정의 (8개, Pipeline 기반)")

    models = {
        # ── 스케일링 필요 모델 (거리/경계 기반) ───────────
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs",
                random_state=RANDOM_STATE
            ))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=7, weights="distance", metric="euclidean"
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                C=1.0, kernel="rbf", gamma="scale",
                probability=True, random_state=RANDOM_STATE
            ))
        ]),
        # ── 스케일링 불필요 모델 (트리 기반) ─────────────
        "Decision Tree": Pipeline([
            ("model", DecisionTreeClassifier(
                max_depth=5, min_samples_split=10, min_samples_leaf=5,
                random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=10,
                min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "LightGBM": Pipeline([
            ("model", LGBMClassifier(
                n_estimators=200, num_leaves=15, learning_rate=0.05,
                min_child_samples=20, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("model", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "CatBoost": Pipeline([
            ("model", CatBoostClassifierSklearn(
                iterations=200, depth=4, learning_rate=0.05,
                l2_leaf_reg=5, random_state=RANDOM_STATE, verbose=False
            ))
        ])
    }

    for name, pipe in models.items():
        has_scaler = "scaler" in pipe.named_steps
        print(f"  {name}: {'스케일링 O' if has_scaler else '스케일링 X'}")

    return models


# =====================================================
# GridSearch 파라미터 그리드
#   - 200-300샘플 소규모 균형 이진분류 기준
#   - 과적합 방지 파라미터 위주
#   - model__ prefix: Pipeline 내부 step 지정
# =====================================================
def get_gridsearch_params(model_name):
    grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10]
        },
        "KNN": {
            "model__n_neighbors": [5, 7, 9, 11, 13],
            "model__weights":     ["uniform", "distance"]
        },
        "SVM": {
            "model__C":     [0.1, 1, 10],
            "model__gamma": ["scale", "auto"]
        },
        "Decision Tree": {
            "model__max_depth":        [3, 4, 5],
            "model__min_samples_leaf": [5, 10, 15]
        },
        "Random Forest": {
            "model__n_estimators":     [200, 300],
            "model__max_depth":        [5, 7, 10],
            "model__min_samples_leaf": [5, 10]
        },
        "LightGBM": {
            "model__num_leaves":        [10, 15, 20],
            "model__learning_rate":     [0.01, 0.05, 0.1],
            "model__min_child_samples": [20, 30]
        },
        "XGBoost": {
            "model__max_depth":     [3, 4, 5],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample":     [0.8, 1.0]
        },
        "CatBoost": {
            "model__depth":         [3, 4, 5],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__l2_leaf_reg":   [3, 5, 10]
        }
    }
    return grids.get(model_name, None)


# =====================================================
# Optuna 파라미터 탐색 범위
#   - GridSearch와 동일 범위를 연속적으로 탐색
#   - 200-300샘플 기준, 과적합 방지 위주
# =====================================================
def get_optuna_params(trial, model_name):
    if model_name == "Logistic Regression":
        return {"model__C": trial.suggest_float("C", 0.01, 10, log=True)}
    elif model_name == "KNN":
        return {
            "model__n_neighbors": trial.suggest_int("n_neighbors", 5, 13),
            "model__weights":     trial.suggest_categorical("weights", ["uniform", "distance"])
        }
    elif model_name == "SVM":
        return {
            "model__C":     trial.suggest_float("C", 0.1, 10, log=True),
            "model__gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
        }
    elif model_name == "Decision Tree":
        return {
            "model__max_depth":        trial.suggest_int("max_depth", 3, 5),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 15)
        }
    elif model_name == "Random Forest":
        return {
            "model__n_estimators":     trial.suggest_int("n_estimators", 200, 300),
            "model__max_depth":        trial.suggest_int("max_depth", 5, 10),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 10)
        }
    elif model_name == "LightGBM":
        return {
            "model__num_leaves":        trial.suggest_int("num_leaves", 10, 20),
            "model__learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "model__min_child_samples": trial.suggest_int("min_child_samples", 20, 30)
        }
    elif model_name == "XGBoost":
        return {
            "model__max_depth":     trial.suggest_int("max_depth", 3, 5),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "model__subsample":     trial.suggest_float("subsample", 0.8, 1.0)
        }
    elif model_name == "CatBoost":
        return {
            "model__depth":         trial.suggest_int("depth", 3, 5),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "model__l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 3, 10)
        }
    return {}


# =====================================================
# 공통 평가 지표 계산
# =====================================================
def compute_metrics(model_label, y_test, y_pred, y_proba):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Model":             model_label,
        "Accuracy":          (y_pred == y_test).mean(),
        "Precision":         precision_score(y_test, y_pred, zero_division=0),
        "Recall":            recall_score(y_test, y_pred, zero_division=0),
        "F1":                f1_score(y_test, y_pred, zero_division=0),
        "Specificity":       tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Sensitivity":       tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Balanced_Accuracy": balanced_accuracy_score(y_test, y_pred),
        "MCC":               matthews_corrcoef(y_test, y_pred),
        "AUC":               roc_auc_score(y_test, y_proba)
    }


# =====================================================
# 5-1: Base 모델 학습 및 평가
#   - clone(): 초기화된 복사본 → GridSearch/Optuna와 완전 독립
#   - X_train으로만 학습
#   - X_test로 최종 1회 평가
# =====================================================
def evaluate_base(pipeline, model_name, X_train, y_train, X_test, y_test):
    model = clone(pipeline)             # 초기 상태 보장
    model.fit(X_train, y_train)         # Train만 사용

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(f"{model_name} (Base)", y_test, y_pred, y_proba)

    print(f"  [Base]       Acc={metrics['Accuracy']:.3f} | "
          f"MCC={metrics['MCC']:.3f} | AUC={metrics['AUC']:.3f}")

    return model, y_proba, metrics


# =====================================================
# 5-2: GridSearch 학습 및 평가
#   - clone(): Base와 완전 독립된 초기 상태
#   - Train 내 5-fold CV로 파라미터 탐색 (MCC 기준)
#   - refit=True: best params로 전체 Train 재학습
#   - X_test로 최종 1회 평가
# =====================================================
def evaluate_gridsearch(pipeline, model_name, X_train, y_train, X_test, y_test):
    param_grid = get_gridsearch_params(model_name)
    if param_grid is None:
        return None, None, None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=clone(pipeline),      # 초기 상태 보장
        param_grid=param_grid,
        cv=cv,                          # Train 내 5-fold CV
        scoring="matthews_corrcoef",    # MCC 기준
        refit=True,                     # best params로 전체 Train 재학습
        n_jobs=-1
    )
    grid.fit(X_train, y_train)          # Train만 사용

    best_model = grid.best_estimator_
    print(f"  [GridSearch] Best: {grid.best_params_}")

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(f"{model_name} (GridSearch)", y_test, y_pred, y_proba)

    print(f"  [GridSearch] Acc={metrics['Accuracy']:.3f} | "
          f"MCC={metrics['MCC']:.3f} | AUC={metrics['AUC']:.3f}")

    return best_model, y_proba, metrics


# =====================================================
# 5-3: Optuna 학습 및 평가
#   - clone(): Base/GridSearch와 완전 독립된 초기 상태
#   - Objective: Train 내 5-fold CV MCC 평균 최대화
#   - Best params로 전체 Train 재학습
#   - X_test로 최종 1회 평가
# =====================================================
def evaluate_optuna(pipeline, model_name, X_train, y_train, X_test, y_test, n_trials=50):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = get_optuna_params(trial, model_name)
        if not params:
            return 0.0
        pipe = clone(pipeline)          # 매 trial마다 초기 상태
        pipe.set_params(**params)
        # Train 내 5-fold CV MCC 평균 최대화
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring="matthews_corrcoef"
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)

    # Best params로 전체 Train 재학습
    best_params = get_optuna_params(study.best_trial, model_name)
    best_model  = clone(pipeline)       # 초기 상태 보장
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)    # Train만 사용

    print(f"  [Optuna]     Best: {study.best_params}")

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(f"{model_name} (Optuna)", y_test, y_pred, y_proba)

    print(f"  [Optuna]     Acc={metrics['Accuracy']:.3f} | "
          f"MCC={metrics['MCC']:.3f} | AUC={metrics['AUC']:.3f}")

    return best_model, y_proba, metrics


# =====================================================
# 6단계: 결과 저장 및 시각화
# =====================================================
def save_results(results_list, y_test,
                 y_proba_base, y_proba_gs, y_proba_optuna,
                 best_model_name, best_model,
                 X_test_fs, selected_features, class_names):
    print("\n[6단계] 결과 저장 및 시각화")

    # 성능 지표 엑셀 (MCC 기준 내림차순)
    df_results = pd.DataFrame(results_list).sort_values("MCC", ascending=False)
    df_results.to_excel(os.path.join(RESULT_DIR, "final_results.xlsx"), index=False)
    print("성능 지표 저장: final_results.xlsx")

    # Confusion Matrix (Best Model)
    y_pred_best = best_model.predict(X_test_fs)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "best_confusion_matrix.png"), dpi=300)
    plt.close()
    print("Confusion Matrix 저장 완료")

    # ROC Curve (Base / GridSearch / Optuna 각각)
    for label, y_proba_dict, fname in [
        ("Base Models",       y_proba_base,   "roc_base.png"),
        ("GridSearch Models", y_proba_gs,     "roc_gridsearch.png"),
        ("Optuna Models",     y_proba_optuna, "roc_optuna.png"),
    ]:
        if not y_proba_dict:
            continue
        plt.figure(figsize=(8, 6))
        for name, y_proba in y_proba_dict.items():
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {label}")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, fname), dpi=300)
        plt.close()
        print(f"ROC Curve 저장: {fname}")

    # Feature Importance (MCC 기준 permutation importance)
    # → 모든 모델에 동일 기준 적용 (논문 일관성)
    perm = permutation_importance(
        best_model, X_test_fs, y_test,
        scoring="matthews_corrcoef",
        n_repeats=10,
        random_state=RANDOM_STATE
    )
    df_fi = pd.DataFrame({
        "Feature":    selected_features,
        "Importance": perm.importances_mean,
        "Std":        perm.importances_std
    }).sort_values("Importance", ascending=False)

    df_fi.to_excel(os.path.join(RESULT_DIR, "feature_importance.xlsx"), index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(df_fi["Feature"], df_fi["Importance"],
             xerr=df_fi["Std"], align="center")
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance (Permutation, MCC) - {best_model_name}")
    plt.xlabel("Mean Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "feature_importance_plot.png"), dpi=300)
    plt.close()
    print("Feature Importance 저장 완료")


# =====================================================
# 7단계: XAI (LIME)
#   - 오분류 샘플 자동 선택
#   - Pipeline이 스케일링 처리 → 원본 데이터 전달
# =====================================================
def xai_lime(best_model, X_train_fs, X_test_fs, y_test, selected_features, class_names):
    print("\n[7단계] XAI (LIME)")

    try:
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train_fs),
            feature_names=selected_features,
            class_names=class_names,
            mode="classification",
            random_state=RANDOM_STATE
        )

        y_pred = best_model.predict(X_test_fs)
        misclassified = np.where(y_pred != y_test)[0]

        if len(misclassified) > 0:
            sample_idx = misclassified[0]
            print(f"오분류 샘플 선택 (index={sample_idx})")
        else:
            sample_idx = 0
            print("오분류 샘플 없음 → 첫 번째 샘플 사용")

        exp = explainer.explain_instance(
            X_test_fs.iloc[sample_idx].values,
            best_model.predict_proba
        )
        lime_path = os.path.join(RESULT_DIR, "lime_explanation.html")
        exp.save_to_file(lime_path)
        print(f"LIME 저장 완료: {lime_path}")

    except Exception as e:
        print(f"LIME 실행 실패: {e}")


# =====================================================
# 8단계: 모델 저장 (Pipeline 1개 = scaler + model)
# =====================================================
def save_model(best_model, selected_features):
    print("\n[8단계] 모델 저장")
    joblib.dump(best_model,        os.path.join(RESULT_DIR, "best_model.pkl"))
    joblib.dump(selected_features, os.path.join(RESULT_DIR, "best_features.pkl"))
    print("저장 완료: best_model.pkl, best_features.pkl")


# =====================================================
# MAIN
# =====================================================
def main():
    print("\n" + "=" * 60)
    print("  머신러닝 파이프라인 시작")
    print("=" * 60)

    # 1. 데이터 로딩
    X, y, feature_cols, class_names = data_loading()

    # 2. Train / Test 분리
    #    Test는 5, 6단계 최종 평가에만 사용
    X_train, X_test, y_train, y_test = data_split(X, y)

    # 3. Feature Selection (Train만 사용)
    selected_features = feature_selection(X_train, y_train, min_features=5)
    X_train_fs = X_train[selected_features]
    X_test_fs  = X_test[selected_features]

    # 4. 모델 정의
    models = build_models()

    # 5. 학습 및 평가
    print("\n[5단계] 모델 학습 및 평가")
    print("  학습/튜닝: X_train (5-fold CV, MCC 기준)")
    print("  평가:      X_test (최종 1회)")

    results_list   = []
    y_proba_base   = {}
    y_proba_gs     = {}
    y_proba_optuna = {}
    best_model      = None
    best_model_name = None
    best_mcc        = -999

    for model_name, pipeline in models.items():
        print(f"\n{'─'*50}")
        print(f"  {model_name}")
        print(f"{'─'*50}")

        # Base
        b_model, b_proba, b_metrics = evaluate_base(
            pipeline, model_name, X_train_fs, y_train, X_test_fs, y_test
        )
        results_list.append(b_metrics)
        y_proba_base[model_name] = b_proba
        if b_metrics["MCC"] > best_mcc:
            best_mcc, best_model, best_model_name = (
                b_metrics["MCC"], b_model, f"{model_name} (Base)"
            )

        # GridSearch
        g_model, g_proba, g_metrics = evaluate_gridsearch(
            pipeline, model_name, X_train_fs, y_train, X_test_fs, y_test
        )
        if g_metrics is not None:
            results_list.append(g_metrics)
            y_proba_gs[model_name] = g_proba
            if g_metrics["MCC"] > best_mcc:
                best_mcc, best_model, best_model_name = (
                    g_metrics["MCC"], g_model, f"{model_name} (GridSearch)"
                )

        # Optuna
        o_model, o_proba, o_metrics = evaluate_optuna(
            pipeline, model_name, X_train_fs, y_train, X_test_fs, y_test, n_trials=50
        )
        if o_metrics is not None:
            results_list.append(o_metrics)
            y_proba_optuna[model_name] = o_proba
            if o_metrics["MCC"] > best_mcc:
                best_mcc, best_model, best_model_name = (
                    o_metrics["MCC"], o_model, f"{model_name} (Optuna)"
                )

    print("\n" + "=" * 60)
    print(f"  Best Model: {best_model_name} | MCC={best_mcc:.3f}")
    print("=" * 60)

    # 6. 결과 저장
    save_results(
        results_list=results_list,
        y_test=y_test,
        y_proba_base=y_proba_base,
        y_proba_gs=y_proba_gs,
        y_proba_optuna=y_proba_optuna,
        best_model_name=best_model_name,
        best_model=best_model,
        X_test_fs=X_test_fs,
        selected_features=selected_features,
        class_names=class_names
    )

    # 7. XAI
    xai_lime(best_model, X_train_fs, X_test_fs, y_test, selected_features, class_names)

    # 8. 모델 저장
    save_model(best_model, selected_features)

    print("\n전체 파이프라인 완료!")


if __name__ == "__main__":
    main()