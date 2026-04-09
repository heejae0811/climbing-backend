import os
import glob
import random
import numpy as np
import pandas as pd
import joblib

from lime.lime_tabular import LimeTabularExplainer

# =====================================================
# 설정
# =====================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

RESULT_DIR = "./result"
LIME_DIR = "./result/lime_html"
os.makedirs(LIME_DIR, exist_ok=True)

# 샘플 모드 선택
# "all"        : 전체 test sample
# "correct"    : 정분류만
# "wrong"      : 오분류만
# "n"          : 랜덤 n개 (아래 SAMPLE_N 설정)
SAMPLE_MODE = "all"
SAMPLE_N = 10  # SAMPLE_MODE = "n" 일 때만 사용

CLASS_NAMES = ["Advanced", "Intermediate"]


# =====================================================
# 1. 데이터 불러오기
# =====================================================
def load_data():
    files = glob.glob("./features_output/*.csv")
    files = [f for f in files if os.path.basename(f) != "all_global_features.csv"]
    if not files:
        raise FileNotFoundError("features_output 폴더에 CSV 파일이 없습니다.")

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            try:
                tmp = pd.read_csv(f, encoding="utf-8")
            except Exception:
                tmp = pd.read_csv(f, encoding="cp949")

        basename = os.path.basename(f)
        person_id = basename.split("_")[0]
        tmp["person_id"] = person_id
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    return df


# =====================================================
# 2. Train/Test 재분리 (기존 파이프라인과 동일 조건)
# =====================================================
def rebuild_split(df, selected_features):
    from sklearn.model_selection import GroupShuffleSplit

    exclude_features = ["id", "label", "body_size_median", "person_id"]
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude_features]

    X = df[feature_cols].copy()
    y = df["label"].astype(int).values
    groups = df["person_id"].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx][selected_features].reset_index(drop=True)
    X_test  = X.iloc[test_idx][selected_features].reset_index(drop=True)
    y_train = y[train_idx]
    y_test  = y[test_idx]

    # 원본 영상 id 추출 (파일명 추적용)
    video_ids_test = df["id"].iloc[test_idx].reset_index(drop=True).values \
        if "id" in df.columns else np.array([f"sample{i}" for i in range(len(test_idx))])

    return X_train, X_test, y_train, y_test, video_ids_test


# =====================================================
# 3. 샘플 인덱스 선택
# =====================================================
def select_samples(model, X_test, y_test):
    y_pred = model.predict(X_test)
    correct_idx   = np.where(y_pred == y_test)[0]
    wrong_idx     = np.where(y_pred != y_test)[0]

    print(f"\n전체 Test 샘플: {len(y_test)}개")
    print(f"  정분류: {len(correct_idx)}개 | 오분류: {len(wrong_idx)}개")

    if SAMPLE_MODE == "all":
        indices = list(range(len(y_test)))
    elif SAMPLE_MODE == "correct":
        indices = list(correct_idx)
    elif SAMPLE_MODE == "wrong":
        indices = list(wrong_idx)
        if len(indices) == 0:
            print("  오분류 샘플이 없습니다. 전체로 대체합니다.")
            indices = list(range(len(y_test)))
    elif SAMPLE_MODE == "n":
        all_idx = list(range(len(y_test)))
        n = min(SAMPLE_N, len(all_idx))
        indices = random.sample(all_idx, n)
    else:
        raise ValueError(f"SAMPLE_MODE 값이 잘못됨: {SAMPLE_MODE}")

    print(f"  → LIME 생성 대상: {len(indices)}개 (mode='{SAMPLE_MODE}')\n")
    return indices, y_pred


# =====================================================
# 4. LIME HTML 생성
# =====================================================
def generate_lime_htmls(model, X_train, X_test, y_test, y_pred, indices, selected_features, video_ids):
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=selected_features,
        class_names=CLASS_NAMES,
        mode="classification",
        random_state=RANDOM_STATE
    )

    for i, idx in enumerate(indices):
        true_label  = CLASS_NAMES[y_test[idx]]
        pred_label  = CLASS_NAMES[y_pred[idx]]
        is_correct  = "O" if y_test[idx] == y_pred[idx] else "X"
        video_id    = str(video_ids[idx])

        exp = explainer.explain_instance(
            X_test.iloc[idx].values,
            model.predict_proba,
            num_samples=5000
        )

        filename = (
            f"{video_id}"
            f"_true-{true_label}"
            f"_pred-{pred_label}"
            f"_{is_correct}.html"
        )
        save_path = os.path.join(LIME_DIR, filename)
        exp.save_to_file(save_path)

        print(f"  [{i+1}/{len(indices)}] {filename}")

    print(f"\n✅ 저장 완료: {LIME_DIR}")


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 55)
    print("  LIME HTML 다중 생성기")
    print("=" * 55)

    # 모델 + feature 로드
    model_path    = os.path.join(RESULT_DIR, "best_model.pkl")
    feature_path  = os.path.join(RESULT_DIR, "best_features.pkl")

    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        raise FileNotFoundError(
            "best_model.pkl 또는 best_features.pkl이 없습니다.\n"
            "기존 파이프라인을 먼저 실행해 주세요."
        )

    model             = joblib.load(model_path)
    selected_features = joblib.load(feature_path)

    print(f"✅ 모델 로드 완료")
    print(f"✅ Feature {len(selected_features)}개 로드: {selected_features}")

    # 데이터 로드 및 분리
    df = load_data()
    X_train, X_test, y_train, y_test, video_ids_test = rebuild_split(df, selected_features)

    # 샘플 선택
    indices, y_pred = select_samples(model, X_test, y_test)

    # LIME HTML 생성
    generate_lime_htmls(model, X_train, X_test, y_test, y_pred, indices, selected_features, video_ids_test)


if __name__ == "__main__":
    main()