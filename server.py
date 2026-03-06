import os
from pathlib import Path

# ==========================================================
# (선택) matplotlib cache writable 설정
# - __file__ 없는 환경(노트북 등)에서도 안전하게 처리
# ==========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

_MPLCONFIGDIR = BASE_DIR / "temp" / "mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from mediapipe import tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image


# ==========================================================
# 0. 설정 및 모델 로드
# ==========================================================
FRAME_INTERVAL = 1
UPLOAD_DIR = BASE_DIR / "temp"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

VISIBILITY_MIN = 0.5
HIP_RECOGNITION_RATIO_MIN = 0.30
SMOOTH_WINDOW = 5

POSE_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_full.task"
MODEL_PATH = BASE_DIR / "result" / "best_model.pkl"
FEATURES_PATH = BASE_DIR / "result" / "best_features.pkl"

print("🔹 Loading ML artifacts...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"best_model.pkl not found: {MODEL_PATH}")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"best_features.pkl not found: {FEATURES_PATH}")

# ✅ best_model.pkl 자체가 Pipeline(스케일러 포함)일 수 있으므로 scaler를 따로 로드하지 않음
model = joblib.load(MODEL_PATH)
selected_features = joblib.load(FEATURES_PATH)

# selected_features가 numpy array로 저장된 케이스 대비
if isinstance(selected_features, np.ndarray):
    selected_features = selected_features.tolist()

print(f"✔ Loaded model: {type(model)}")
print(f"✔ Loaded. Features: {len(selected_features)}")


# ==========================================================
# 1. 유틸리티 함수
# ==========================================================
def fill_missing(arr):
    s = pd.Series(arr, dtype=float)
    s = s.interpolate(limit_direction="both").ffill().bfill()
    return s.to_numpy()


def center_point(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def velocity_series(pts, dt):
    v = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        v.append(np.sqrt(dx ** 2 + dy ** 2) / dt)
    return np.array(v)


def acc_series(v, dt):
    return np.gradient(v) / dt


def jerk_series(a, dt):
    return np.gradient(a) / dt


def moving_average(arr, window):
    if window <= 1:
        return np.asarray(arr, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.size < window:
        return arr
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def robust_body_size_from_landmarks(lm, visibility_min):
    def dist(i, j):
        dx = lm[i].x - lm[j].x
        dy = lm[i].y - lm[j].y
        return np.sqrt(dx * dx + dy * dy)

    pairs = [(11, 12), (23, 24)]
    dists = []
    for i, j in pairs:
        if lm[i].visibility >= visibility_min and lm[j].visibility >= visibility_min:
            d = dist(i, j)
            if np.isfinite(d) and d > 1e-6:
                dists.append(d)

    if len(dists) == 0:
        return np.nan
    return float(np.mean(dists))


# ==========================================================
# 2. 특징 추출 (mediapipe tasks 버전)
# ==========================================================
def extract_features(video_path: str):
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pose model not found: {POSE_MODEL_PATH}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if fps <= 0 else float(fps)
    dt = 1.0 / fps

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = frame_count / fps if fps > 0 else np.nan

    hip_pts, body_sizes, hip_visible_flags = [], [], []
    frame_idx = 0

    base_options = tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_INTERVAL != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_frame, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                left_ok = lm[23].visibility >= VISIBILITY_MIN
                right_ok = lm[24].visibility >= VISIBILITY_MIN

                if left_ok and right_ok:
                    hip_pts.append(center_point((lm[23].x, lm[23].y), (lm[24].x, lm[24].y)))
                    body_sizes.append(robust_body_size_from_landmarks(lm, VISIBILITY_MIN))
                    hip_visible_flags.append(True)
                else:
                    hip_pts.append((np.nan, np.nan))
                    body_sizes.append(np.nan)
                    hip_visible_flags.append(False)
            else:
                hip_pts.append((np.nan, np.nan))
                body_sizes.append(np.nan)
                hip_visible_flags.append(False)

            frame_idx += 1

    cap.release()

    hip_recognition_ratio = float(np.mean(hip_visible_flags)) if hip_visible_flags else 0.0
    if hip_recognition_ratio < HIP_RECOGNITION_RATIO_MIN:
        return None

    bs = np.asarray(body_sizes, dtype=float)
    valid_bs = bs[np.isfinite(bs)]
    if valid_bs.size == 0:
        return None

    bs_med = float(np.median(valid_bs))
    if not np.isfinite(bs_med) or bs_med <= 1e-6:
        return None

    hip_x = fill_missing([p[0] for p in hip_pts])
    hip_y = fill_missing([p[1] for p in hip_pts])
    hip_xy = list(zip(hip_x, hip_y))

    hip_v_pixel = velocity_series(hip_xy, dt)
    hip_v = hip_v_pixel / bs_med

    hip_v_for_diff = moving_average(hip_v, SMOOTH_WINDOW)
    hip_a = acc_series(hip_v_for_diff, dt)
    hip_j = jerk_series(hip_a, dt)

    path_pixel = np.sum(hip_v_pixel * dt)
    path = path_pixel / bs_med

    feats = {
        "total_time": total_time,
        "body_size_median": bs_med,
        "fluency_hip_velocity_mean": float(np.mean(hip_v)),
        "fluency_hip_velocity_max": float(np.max(hip_v)),
        "fluency_hip_acc_mean": float(np.mean(np.abs(hip_a))),
        "fluency_hip_acc_max": float(np.max(np.abs(hip_a))),
        "fluency_hip_jerk_mean": float(np.mean(np.abs(hip_j))),
        "fluency_hip_jerk_max": float(np.max(np.abs(hip_j))),
        "fluency_hip_jerk_rms": float(np.sqrt(np.mean(hip_j ** 2))),
        "fluency_hip_path_length": float(path),
        "stability_hip_velocity_sd": float(np.std(hip_v)),
        "stability_hip_acc_sd": float(np.std(hip_a)),
        "stability_hip_jerk_sd": float(np.std(hip_j)),
    }

    return feats


# ==========================================================
# 3. 한국어 피드백 생성
# ==========================================================
def generate_korean_feedback(feats):
    msg = []
    if feats.get("fluency_hip_jerk_mean", 0) > 0.05:
        msg.append("움직임이 다소 급합니다. 무게 중심을 더 천천히 이동시켜 보세요.")
    else:
        msg.append("중심 이동이 매우 부드럽고 안정적입니다.")

    if feats.get("stability_hip_velocity_sd", 0) > 0.08:
        msg.append("일정한 속도를 유지하기보다 끊기는 동작이 보입니다. 리듬감을 높여보세요.")

    return msg if msg else ["전반적으로 안정적인 등반입니다."]


# ==========================================================
# 4. Flask 서버 설정
# ==========================================================
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    temp_path = None
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video"}), 400

        video = request.files["video"]
        if not video.filename:
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(video.filename)
        temp_path = str(UPLOAD_DIR / filename)
        video.save(temp_path)

        feats = extract_features(temp_path)
        if feats is None:
            return jsonify({"error": "Landmark extraction failed"}), 422

        # ✅ 학습 때 쓴 feature 순서로 맞추고 결측은 0 처리
        X = pd.DataFrame([feats]).reindex(columns=selected_features).fillna(0)

        # ✅ model이 Pipeline이면 내부에서 scaler 포함되어 자동 처리됨
        # ✅ Pipeline이 아니어도 그대로 predict 가능
        pred = int(model.predict(X)[0])

        # predict_proba 없는 모델 대비(보통은 다 있음)
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0, 1])
        else:
            prob = None

        return jsonify({
            "prediction": {
                "label": "Advanced" if pred == 0 else "Intermediate",  # 0: Advanced, 1: Intermediate
                "probability": None if prob is None else round(prob, 3)
            },
            "feedback_features": {k: round(float(v), 4) for k, v in feats.items()},
            "feedback_messages": generate_korean_feedback(feats)
        })

    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    # 외부 접속 필요 없으면 host="127.0.0.1"로 바꿔도 됨
    app.run(host="0.0.0.0", port=5001, debug=False)