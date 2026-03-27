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

FRAME_INTERVAL = 1
VISIBILITY_MIN = 0.5
JOINT_RECOGNITION_RATIO_MIN = 0.30
SMOOTH_WINDOW = 5
POSE_MODEL_PATH = "./models/pose_landmarker_full.task"

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

    m = re.search(r'_(\d)_', stem)
    label = int(m.group(1)) if m else None

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
    s = pd.Series(arr, dtype=float)
    s = s.interpolate(limit_direction="both").ffill().bfill()
    return s.to_numpy()


# ==========================================================
# 3. 기본 계산 함수
# ==========================================================
def moving_average(arr, window):
    if window <= 1:
        return np.asarray(arr, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.size < window:
        return arr
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def compute_diff(arr, dt):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return np.zeros_like(arr)
    return np.gradient(arr) / dt


def compute_abs_stats(arr, prefix):
    arr = np.asarray(arr, dtype=float)
    abs_arr = np.abs(arr)
    return {
        f"{prefix}_mean": float(np.mean(abs_arr)),
        f"{prefix}_max": float(np.max(abs_arr)),
        f"{prefix}_sd": float(np.std(abs_arr)),
    }


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


def compute_path_length(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))


def compute_displacement(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return 0.0
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    return float(np.sqrt(dx ** 2 + dy ** 2))


# ==========================================================
# 4. 개별 관절 feature 계산
# ==========================================================
def compute_joint_features(joint_x, joint_y, bs_med, dt, smooth_window, joint_name):
    """
    단일 관절의 x,y 좌표 시계열에서
    velocity / acc / jerk / path_efficiency / vertical_efficiency만 추출
    """

    joint_x = np.asarray(joint_x, dtype=float)
    joint_y = np.asarray(joint_y, dtype=float)

    # 시작점 기준 정렬 + body size 정규화
    x_rel = (joint_x - joint_x[0]) / bs_med
    y_rel = (joint_y - joint_y[0]) / bs_med

    dx = np.diff(x_rel, prepend=x_rel[0])
    dy = np.diff(y_rel, prepend=y_rel[0])

    dx_s = moving_average(dx, smooth_window)
    dy_s = moving_average(dy, smooth_window)

    velocity = np.sqrt(dx_s ** 2 + dy_s ** 2) / dt
    velocity_s = moving_average(velocity, smooth_window)

    acc = compute_diff(velocity_s, dt)
    jerk = compute_diff(acc, dt)

    path_length = compute_path_length(x_rel, y_rel)
    straight_dist = compute_displacement(x_rel, y_rel)
    path_efficiency = straight_dist / path_length if path_length > 1e-8 else 0.0

    vertical_gain = y_rel[-1] - y_rel[0]
    vertical_efficiency = np.abs(vertical_gain) / path_length if path_length > 1e-8 else 0.0

    feats = {}
    feats.update(compute_abs_stats(velocity, f"{joint_name}_velocity"))
    feats.update(compute_abs_stats(acc, f"{joint_name}_acc"))
    feats.update(compute_abs_stats(jerk, f"{joint_name}_jerk"))

    feats[f"{joint_name}_path_efficiency"] = float(path_efficiency)
    feats[f"{joint_name}_vertical_efficiency"] = float(vertical_efficiency)

    return feats


# ==========================================================
# 5. center series 계산
# ==========================================================
def compute_center_series(center_x, center_y, bs_med):
    center_x = np.asarray(center_x, dtype=float)
    center_y = np.asarray(center_y, dtype=float)

    x_rel = (center_x - center_x[0]) / bs_med
    y_rel = (center_y - center_y[0]) / bs_med

    return x_rel, y_rel


# ==========================================================
# 6. relation feature 계산
# ==========================================================
def compute_relation_features(
    left_hip_x, left_hip_y, right_hip_x, right_hip_y,
    left_sh_x, left_sh_y, right_sh_x, right_sh_y,
    hip_center_x_rel, hip_center_y_rel,
    shoulder_center_x_rel, shoulder_center_y_rel,
    bs_med
):
    feats = {}

    # ------------------------------------------------------
    # Symmetry
    # ------------------------------------------------------
    hip_y_diff = np.abs((left_hip_y - right_hip_y) / bs_med)
    hip_x_diff = np.abs((left_hip_x - right_hip_x) / bs_med)
    shoulder_y_diff = np.abs((left_sh_y - right_sh_y) / bs_med)
    shoulder_x_diff = np.abs((left_sh_x - right_sh_x) / bs_med)

    feats.update(compute_abs_stats(hip_y_diff, "hip_y_symmetry"))
    feats.update(compute_abs_stats(hip_x_diff, "hip_x_symmetry"))
    feats.update(compute_abs_stats(shoulder_y_diff, "shoulder_y_symmetry"))
    feats.update(compute_abs_stats(shoulder_x_diff, "shoulder_x_symmetry"))

    # ------------------------------------------------------
    # 중심 이동 안정성만 유지
    # ------------------------------------------------------
    feats["hip_center_horizontal_sway_sd"] = float(np.std(hip_center_x_rel))
    feats["hip_center_vertical_sway_sd"] = float(np.std(hip_center_y_rel))
    feats["shoulder_center_horizontal_sway_sd"] = float(np.std(shoulder_center_x_rel))
    feats["shoulder_center_vertical_sway_sd"] = float(np.std(shoulder_center_y_rel))

    return feats


# ==========================================================
# 7. Feature Extraction
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

                body_sizes.append(robust_body_size_from_landmarks(lm, VISIBILITY_MIN))

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

    bs = np.array(body_sizes, dtype=float)
    valid_bs = bs[np.isfinite(bs)]

    if valid_bs.size == 0:
        print(f"❌ BodySize 유효값 없음 → {os.path.basename(video_path)}")
        return None

    bs_med = float(np.median(valid_bs))
    if not np.isfinite(bs_med) or bs_med <= 1e-6:
        print(f"❌ BodySize 비정상 → {os.path.basename(video_path)}")
        return None

    video_id, label = extract_id_and_label(video_path)

    feats = {
        "id": video_id,
        "label": label,
        "total_time": total_time,
        "body_size_median": bs_med,
    }

    processed = {}

    # ------------------------------------------------------
    # recognition ratio는 저장하지 않고 품질 필터링만 수행
    # ------------------------------------------------------
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

        processed[joint_name] = {
            "x": joint_x,
            "y": joint_y
        }

    # ------------------------------------------------------
    # 개별 관절 feature
    # ------------------------------------------------------
    for joint_name in TARGET_JOINTS.keys():
        joint_feats = compute_joint_features(
            joint_x=processed[joint_name]["x"],
            joint_y=processed[joint_name]["y"],
            bs_med=bs_med,
            dt=dt,
            smooth_window=SMOOTH_WINDOW,
            joint_name=joint_name
        )
        feats.update(joint_feats)

    # ------------------------------------------------------
    # center 계산 (feature 추출은 하지 않고 sway 계산용으로만 사용)
    # ------------------------------------------------------
    left_hip_x = processed["left_hip"]["x"]
    left_hip_y = processed["left_hip"]["y"]
    right_hip_x = processed["right_hip"]["x"]
    right_hip_y = processed["right_hip"]["y"]

    left_sh_x = processed["left_shoulder"]["x"]
    left_sh_y = processed["left_shoulder"]["y"]
    right_sh_x = processed["right_shoulder"]["x"]
    right_sh_y = processed["right_shoulder"]["y"]

    hip_center_x = (left_hip_x + right_hip_x) / 2.0
    hip_center_y = (left_hip_y + right_hip_y) / 2.0
    shoulder_center_x = (left_sh_x + right_sh_x) / 2.0
    shoulder_center_y = (left_sh_y + right_sh_y) / 2.0

    hip_center_x_rel, hip_center_y_rel = compute_center_series(
        hip_center_x, hip_center_y, bs_med
    )
    shoulder_center_x_rel, shoulder_center_y_rel = compute_center_series(
        shoulder_center_x, shoulder_center_y, bs_med
    )

    # ------------------------------------------------------
    # relation feature
    # ------------------------------------------------------
    relation_feats = compute_relation_features(
        left_hip_x, left_hip_y, right_hip_x, right_hip_y,
        left_sh_x, left_sh_y, right_sh_x, right_sh_y,
        hip_center_x_rel, hip_center_y_rel,
        shoulder_center_x_rel, shoulder_center_y_rel,
        bs_med
    )
    feats.update(relation_feats)

    return feats


# ==========================================================
# 8. MAIN
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