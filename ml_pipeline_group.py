import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, GroupShuffleSplit
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
#   - id, label, total_time, body_size_median 제외
#   - 수치형 feature만 자동 선택
#   - 파일명에서 person_id 추출 (그룹 분할용)
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
        # 파일명에서 앞 숫자(person_id) 추출: e.g. "150_정유라_1_230315.xlsx" → "150"
        basename = os.path.basename(f)
        person_id = basename.split("_")[0]
        tmp["person_id"] = person_id
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    if "label" not in df.columns:
        raise KeyError("label 컬럼이 없습니다.")

    y      = df["label"].astype(int).values
    groups = df["person_id"].values
    class_names  = ["Advanced", "Intermediate"]
    label_counts = np.bincount(y)

    exclude_features = ["id", "label", "total_time", "body_size_median", "person_id"]
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude_features]

    X = df[feature_cols]

    unique_persons = np.unique(groups)
    print(f"Feature 개수: {len(feature_cols)}")
    print(f"총 인원 수: {len(unique_persons)}명 → {list(unique_persons)}")
    print(f"클래스 분포: Advanced(0)={label_counts[0]}개, Intermediate(1)={label_counts[1]}개")

    return X, y, groups, feature_cols, class_names


# =====================================================
# 2단계: Train / Test 분리 (Group 기반)
#   - GroupShuffleSplit: 같은 person_id는 반드시 같은 세트에
#   - Test는 최종 평가에만 사용, 학습/튜닝에 절대 미사용
# =====================================================
def data_split(X, y, groups, test_size=0.2):
    print("\n[2단계] Train/Test Split (Group 기반, 80:20)")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx],      y[test_idx]

    train_persons = np.unique(groups[train_idx])
    test_persons  = np.unique(groups[test_idx])

    print(f"Train: {len(X_train)}개  |  Test: {len(X_test)}개")
    print(f"Train 사람: {list(train_persons)} ({len(train_persons)}명)")
    print(f"Test  사람: {list(test_persons)}  ({len(test_persons)}명)")
    print(f"Train 클래스 분포: {np.bincount(y_train)}")
    print(f"Test  클래스 분포: {np.bincount(y_test)}")

    overlap = set(train_persons) & set(test_persons)
    if overlap:
        raise ValueError(f"동일 인물이 Train/Test 양쪽에 존재: {overlap}")
    print("✓ Train/Test 간 인물 중복 없음 확인")

    return X_train, X_test, y_train, y_test


# =====================================================
# 3단계: Feature Selection (RFECV)
#   [방법] RFECV (Recursive Feature Elimination with CV)
#   [추정기] RandomForest (feature_importances_ 제공)
#   [CV]   Train 내 5-fold Stratified CV
#   [기준] MCC (모델 학습/튜닝 기준과 통일)
#   [선택] CV 성능 최대화하는 최적 변수 개수 자동 결정
#   [보장] Train 데이터만 사용, Test leakage 없음
# =====================================================
def feature_selection_rfecv(X_train, y_train, feature_cols, min_features=3):
    print("\n[3단계] Feature Selection (RFECV + RandomForest + 5-Fold CV + MCC)")

    from sklearn.metrics import make_scorer

    def specificity_mcc_scorer(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)

    mcc_scorer = make_scorer(specificity_mcc_scorer)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # RFECV의 estimator: feature_importances_를 가진 트리 기반 모델 사용
    estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rfecv = RFECV(
        estimator=estimator,
        step=1,                  # 매 iteration마다 변수 1개씩 제거
        cv=cv,
        scoring=mcc_scorer,      # MCC 기준으로 최적 변수 수 결정
        min_features_to_select=min_features,
        n_jobs=-1
    )
    rfecv.fit(X_train, y_train)

    selected_mask     = rfecv.support_
    selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
    n_selected        = rfecv.n_features_

    print(f"전체 변수 수: {len(feature_cols)}개")
    print(f"최적 변수 수 (RFECV 자동 결정): {n_selected}개")
    print(f"Selected Features: {selected_features}")

    # 변수별 ranking 저장 (1 = 선택됨)
    df_ranking = pd.DataFrame({
        "Feature":  feature_cols,
        "Ranking":  rfecv.ranking_,
        "Selected": selected_mask
    }).sort_values("Ranking")
    df_ranking.to_excel(
        os.path.join(RESULT_DIR, "rfecv_feature_ranking.xlsx"), index=False)

    # RFECV CV 점수 시각화 (변수 개수별 MCC)
    cv_results = rfecv.cv_results_
    n_features_range = range(min_features, len(feature_cols) + 1)
    mean_scores = cv_results["mean_test_score"]
    std_scores  = cv_results["std_test_score"]

    plt.figure(figsize=(10, 5))
    plt.plot(n_features_range, mean_scores, marker="o", lw=2, label="Mean MCC")
    plt.fill_between(
        n_features_range,
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.2, label="±1 Std"
    )
    plt.axvline(x=n_selected, color="red", linestyle="--",
                label=f"Optimal: {n_selected} features")
    plt.xlabel("Number of Features")
    plt.ylabel("Cross-Validated MCC")
    plt.title("RFECV: MCC vs Number of Features")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "rfecv_score_plot.png"), dpi=300)
    plt.close()
    print("RFECV 결과 저장 완료: rfecv_feature_ranking.xlsx, rfecv_score_plot.png")

    return selected_features


# =====================================================
# 4단계: 모델 정의 (8개)
# =====================================================
def build_models():
    print("\n[4단계] 모델 정의 (8개, Pipeline 기반)")

    models = {
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
# =====================================================
def evaluate_base(pipeline, model_name, X_train, y_train, X_test, y_test):
    model = clone(pipeline)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(f"{model_name} (Base)", y_test, y_pred, y_proba)

    print(f"  [Base]       Acc={metrics['Accuracy']:.3f} | "
          f"MCC={metrics['MCC']:.3f} | AUC={metrics['AUC']:.3f}")

    return model, y_proba, metrics


# =====================================================
# 5-2: GridSearch 학습 및 평가
# =====================================================
def evaluate_gridsearch(pipeline, model_name, X_train, y_train, X_test, y_test):
    param_grid = get_gridsearch_params(model_name)
    if param_grid is None:
        return None, None, None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        cv=cv,
        scoring="matthews_corrcoef",
        refit=True,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"  [GridSearch] Best: {grid.best_params_}")

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(f"{model_name} (GridSearch)", y_test, y_pred, y_proba)

    print(f"  [GridSearch] Acc={metrics['Accuracy']:.3f} | "
          f"MCC={metrics['MCC']:.3f} | AUC={metrics['AUC']:.3f}")

    return best_model, y_proba, metrics


# =====================================================
# 6단계: 결과 저장 및 시각화
# =====================================================
def save_results(results_list, y_test,
                 y_proba_base, y_proba_gs,
                 best_model_name, best_model,
                 X_test, selected_features, class_names):
    print("\n[6단계] 결과 저장 및 시각화")

    df_results = pd.DataFrame(results_list).sort_values("MCC", ascending=False)
    df_results.to_excel(os.path.join(RESULT_DIR, "final_results.xlsx"), index=False)
    print("성능 지표 저장: final_results.xlsx")

    y_pred_best = best_model.predict(X_test)
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

    for label, y_proba_dict, fname in [
        ("Base Models",       y_proba_base, "roc_base.png"),
        ("GridSearch Models", y_proba_gs,   "roc_gridsearch.png"),
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

    perm = permutation_importance(
        best_model, X_test, y_test,
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

    plt.figure(figsize=(10, max(6, len(selected_features) * 0.3)))
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

        if len(misclassified) > 0:
            sample_idx = misclassified[0]
            print(f"오분류 샘플 선택 (index={sample_idx})")
        else:
            sample_idx = 0
            print("오분류 샘플 없음 → 첫 번째 샘플 사용")

        exp = explainer.explain_instance(
            X_test.iloc[sample_idx].values,
            best_model.predict_proba
        )
        lime_path = os.path.join(RESULT_DIR, "lime_explanation.html")
        exp.save_to_file(lime_path)
        print(f"LIME 저장 완료: {lime_path}")

    except Exception as e:
        print(f"LIME 실행 실패: {e}")


# =====================================================
# 8단계: 모델 저장
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
    X, y, groups, feature_cols, class_names = data_loading()

    # 2. Train / Test 분리 (Group 기반: 같은 사람은 같은 세트)
    X_train, X_test, y_train, y_test = data_split(X, y, groups)

    # 3. Feature Selection - RFECV (Train만 사용)
    selected_features = feature_selection_rfecv(X_train, y_train, feature_cols, min_features=3)
    X_train_fs = X_train[selected_features]
    X_test_fs  = X_test[selected_features]

    # 4. 모델 정의
    models = build_models()

    # 5. 학습 및 평가
    print("\n[5단계] 모델 학습 및 평가")
    print(f"  사용 Feature: RFECV 선택 {len(selected_features)}개")
    print("  학습/튜닝: X_train (5-fold CV, MCC 기준)")
    print("  평가:      X_test (최종 1회)")

    results_list = []
    y_proba_base = {}
    y_proba_gs   = {}
    best_model      = None
    best_model_name = None
    best_mcc        = -999

    for model_name, pipeline in models.items():
        print(f"\n{'─'*50}")
        print(f"  {model_name}")
        print(f"{'─'*50}")

        b_model, b_proba, b_metrics = evaluate_base(
            pipeline, model_name, X_train_fs, y_train, X_test_fs, y_test
        )
        results_list.append(b_metrics)
        y_proba_base[model_name] = b_proba
        if b_metrics["MCC"] > best_mcc:
            best_mcc, best_model, best_model_name = (
                b_metrics["MCC"], b_model, f"{model_name} (Base)"
            )

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

    print("\n" + "=" * 60)
    print(f"  Best Model: {best_model_name} | MCC={best_mcc:.3f}")
    print("=" * 60)

    # 6. 결과 저장
    save_results(
        results_list=results_list,
        y_test=y_test,
        y_proba_base=y_proba_base,
        y_proba_gs=y_proba_gs,
        best_model_name=best_model_name,
        best_model=best_model,
        X_test=X_test_fs,
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