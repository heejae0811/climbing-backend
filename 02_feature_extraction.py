import re
import os
import glob
import cv2
import numpy as np
import pandas as pd
from mediapipe import tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image

# ==========================================================
# 0. 기본 설정
# ==========================================================
VIDEO_DIR = "./videos/"
OUTPUT_DIR = "./features_xlsx/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_INTERVAL = 1  # 모든 프레임 분석
VISIBILITY_MIN = 0.5  # landmark visibility threshold
JOINT_RECOGNITION_RATIO_MIN = 0.30  # 각 관절 인식률 30% 미만이면 제외
SMOOTH_WINDOW = 5  # acc/jerk 계산용 이동평균 창 (홀수 권장)
POSE_MODEL_PATH = "./models/pose_landmarker_full.task"

# 분석할 관절
TARGET_JOINTS = {
    "left_hip": 23,
    "right_hip": 24,
    "left_shoulder": 11,
    "right_shoulder": 12
}


# ==========================================================
# 1. id, label
# ==========================================================
def extract_id_and_label(video_path):
    fname = os.path.basename(video_path)
    stem = os.path.splitext(fname)[0]

    # Label 추출 (언더바 사이의 숫자)
    m = re.search(r'_(\d)_', stem)
    label = int(m.group(1)) if m else None

    # ID 생성: 앞 숫자 + "_" + label
    parts = stem.split('_')
    if len(parts) >= 1 and label is not None:
        video_id = f"{parts[0]}_{label}"
    else:
        video_id = stem

    return video_id, label


# ==========================================================
# 2. Missing 처리
# ==========================================================
def fill_missing(arr):
    """보간 후 forward/backward fill로 완전히 채우기"""
    s = pd.Series(arr, dtype=float)
    s = s.interpolate(limit_direction="both").ffill().bfill()
    return s.to_numpy()


def nan_ratio(arr):
    return float(np.mean(np.isnan(np.asarray(arr, dtype=float))))


# ==========================================================
# 3. Kinematics 계산 함수
# ==========================================================
def velocity_series(pts, dt):
    """속도 시계열 계산 (첫 프레임 속도는 0)"""
    v = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        v.append(np.sqrt(dx ** 2 + dy ** 2) / dt)
    return np.array(v, dtype=float)


def acc_series(v, dt):
    """가속도 계산"""
    return np.gradient(v) / dt


def jerk_series(a, dt):
    """저크 계산"""
    return np.gradient(a) / dt


def moving_average(arr, window):
    """간단 이동평균 (창이 크면 더 부드럽게)"""
    if window <= 1:
        return np.asarray(arr, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.size < window:
        return arr
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def robust_body_size_from_landmarks(lm, visibility_min):
    """Robust한 body size 계산 - 어깨폭/골반폭만 사용, visibility 기준 적용"""
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
# 4. Joint feature 계산 함수
# ==========================================================
def compute_joint_features(joint_xy, bs_med, dt, smooth_window, joint_name):
    """
    단일 관절의 x,y 좌표 시계열로부터 feature 계산
    모든 값은 body size median으로 정규화
    """
    joint_v_pixel = velocity_series(joint_xy, dt)

    # body size 정규화
    joint_v = joint_v_pixel / bs_med
    joint_v_for_diff = moving_average(joint_v, smooth_window)
    joint_a = acc_series(joint_v_for_diff, dt)
    joint_j = jerk_series(joint_a, dt)

    # path length (픽셀 총 이동거리 / body size)
    path_pixel = np.sum(joint_v_pixel * dt)
    path = path_pixel / bs_med

    feats = {
        f"{joint_name}_velocity_mean": float(np.mean(joint_v)),
        f"{joint_name}_velocity_max": float(np.max(joint_v)),
        f"{joint_name}_velocity_sd": float(np.std(joint_v)),

        f"{joint_name}_acc_mean": float(np.mean(np.abs(joint_a))),
        f"{joint_name}_acc_max": float(np.max(np.abs(joint_a))),
        f"{joint_name}_acc_sd": float(np.std(joint_a)),

        f"{joint_name}_jerk_mean": float(np.mean(np.abs(joint_j))),
        f"{joint_name}_jerk_max": float(np.max(np.abs(joint_j))),
        f"{joint_name}_jerk_sd": float(np.std(joint_j)),

        f"{joint_name}_path_length": float(path),
    }

    return feats


# ==========================================================
# 5. Feature Extraction
# ==========================================================
def extract_features(video_path):
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"❌ Pose 모델 파일 없음 → {POSE_MODEL_PATH}")
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if fps <= 0 else fps
    dt = 1.0 / fps

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = frame_count / fps if fps > 0 else np.nan

    frame_idx = 0

    # 각 관절별 좌표/가시성 저장
    joint_points = {joint_name: [] for joint_name in TARGET_JOINTS.keys()}
    joint_visible_flags = {joint_name: [] for joint_name in TARGET_JOINTS.keys()}
    body_sizes = []

    base_options = tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
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

                # body size
                body_sizes.append(robust_body_size_from_landmarks(lm, VISIBILITY_MIN))

                # 각 관절 좌표 추출
                for joint_name, joint_idx in TARGET_JOINTS.items():
                    if lm[joint_idx].visibility >= VISIBILITY_MIN:
                        joint_points[joint_name].append((lm[joint_idx].x, lm[joint_idx].y))
                        joint_visible_flags[joint_name].append(True)
                    else:
                        joint_points[joint_name].append((np.nan, np.nan))
                        joint_visible_flags[joint_name].append(False)

            else:
                body_sizes.append(np.nan)
                for joint_name in TARGET_JOINTS.keys():
                    joint_points[joint_name].append((np.nan, np.nan))
                    joint_visible_flags[joint_name].append(False)

            frame_idx += 1

    cap.release()

    # body size median 계산
    bs = np.array(body_sizes, dtype=float)
    valid_bs = bs[np.isfinite(bs)]

    if valid_bs.size == 0:
        print(f"❌ BodySize 유효값 없음 → {os.path.basename(video_path)}")
        return None

    bs_med = float(np.median(valid_bs))
    if not np.isfinite(bs_med) or bs_med <= 1e-6:
        print(f"❌ BodySize 비정상 → {os.path.basename(video_path)}")
        return None

    feats = {
        "id": extract_id_and_label(video_path)[0],
        "label": extract_id_and_label(video_path)[1],
        "total_time": total_time,
        "body_size_median": bs_med,
    }

    # 각 관절별 인식률 확인 및 feature 계산
    for joint_name in TARGET_JOINTS.keys():
        visible_flags = joint_visible_flags[joint_name]
        recognition_ratio = float(np.mean(visible_flags)) if visible_flags else 0.0

        if recognition_ratio < JOINT_RECOGNITION_RATIO_MIN:
            print(f"❌ {joint_name} 인식률 {recognition_ratio:.2f} → {os.path.basename(video_path)}")
            return None

        joint_x = np.array([p[0] for p in joint_points[joint_name]], dtype=float)
        joint_y = np.array([p[1] for p in joint_points[joint_name]], dtype=float)

        joint_x = fill_missing(joint_x)
        joint_y = fill_missing(joint_y)

        joint_xy = list(zip(joint_x, joint_y))

        joint_feats = compute_joint_features(
            joint_xy=joint_xy,
            bs_med=bs_med,
            dt=dt,
            smooth_window=SMOOTH_WINDOW,
            joint_name=joint_name
        )

        feats.update(joint_feats)

    return feats


# ==========================================================
# 6. MAIN
# ==========================================================
def main():
    files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + \
            glob.glob(os.path.join(VIDEO_DIR, "*.mov"))

    if not files:
        print("❌ 분석할 비디오가 없습니다.")
        return

    print(f"📁 총 {len(files)}개 비디오 분석 시작\n")

    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}.xlsx")

        print(f"[{idx}/{len(files)}] {os.path.basename(video_path)}")
        feats = extract_features(video_path)

        if feats is None:
            print(f"  ❌ Feature 추출 실패 → {video_path}\n")
            fail_count += 1
            continue

        df = pd.DataFrame([feats])
        df.to_excel(out_path, index=False)
        print(f"  ✅ 저장 완료: {base}.xlsx\n")
        success_count += 1

    print("=" * 60)
    print("🎉 분석 완료!")
    print(f"   성공: {success_count}개 | 실패: {fail_count}개")
    print("=" * 60)


if __name__ == "__main__":
    main()