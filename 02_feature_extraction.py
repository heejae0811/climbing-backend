"""
클라이밍 영상 MediaPipe 2D 기반 feature extraction
- 총 86개 변수 추출
  * 운동학 (kinematics)
      - 8개 관절 (left/right shoulder, hip, wrist, ankle) × 6개 = 48개
        변수: velocity(mean/max/sd), acceleration(mean/max/sd)
      - 2개 center (shoulder_center, hip_center) × 9개 = 18개
        변수: velocity(mean/max/sd), acceleration(mean/max/sd), jerk(mean/max/sd)
      → 소계: 66개
  * center 추가변수 : 8개
      hip_center_gie, shoulder_center_gie          (ConvexHull 기반 경로 효율성)
      hip_center_path_efficiency,                  (직선거리 / 누적거리)
      shoulder_center_path_efficiency,
      hip_center_horizontal_sway,                  (x좌표 std)
      hip_center_vertical_sway,                    (y좌표 std)
      shoulder_center_horizontal_sway,
      shoulder_center_vertical_sway
  * symmetry : 12개
      shoulder_x/y_symmetry(mean/max/sd), hip_x/y_symmetry(mean/max/sd)
- 출력 파일명: 01_0.csv 형식
"""

import os
import glob
import traceback
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import ConvexHull

# =========================================================
# 0. 사용자 설정
# =========================================================

INPUT_VIDEO_DIR = "./videos"
OUTPUT_ROOT_DIR = "./features_output"
POSE_MODEL_PATH = "./models/pose_landmarker_full.task"
VIDEO_EXTENSIONS = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV")

VISIBILITY_MIN = 0.5
MAX_INTERP_GAP = 5
FRAME_INTERVAL = 1

MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
NUM_POSES = 1

USE_PIXEL_COORDINATES = True
SAVE_VISIBILITY = True
AUTO_PARSE_LABEL_FROM_FILENAME = True

# =========================================================
# 1. Pose landmark index 정의
# =========================================================

POSE_IDX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# 실제 landmark에서 추출할 8개 관절
LANDMARK_MAP_RAW = {
    "left_shoulder":  POSE_IDX["left_shoulder"],
    "right_shoulder": POSE_IDX["right_shoulder"],
    "left_hip":       POSE_IDX["left_hip"],
    "right_hip":      POSE_IDX["right_hip"],
    "left_wrist":     POSE_IDX["left_wrist"],
    "right_wrist":    POSE_IDX["right_wrist"],
    "left_ankle":     POSE_IDX["left_ankle"],
    "right_ankle":    POSE_IDX["right_ankle"],
}

# velocity + acceleration만 계산할 8개 관절 (jerk 없음)
JOINTS_RAW = [
    "left_shoulder", "right_shoulder",
    "left_hip",      "right_hip",
    "left_wrist",    "right_wrist",
    "left_ankle",    "right_ankle",
]

# velocity + acceleration + jerk 모두 계산할 2개 center
JOINTS_CENTER = ["shoulder_center", "hip_center"]

# =========================================================
# 2. 폴더 생성
# =========================================================

def ensure_directories() -> None:
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

# =========================================================
# 3. 유틸 함수
# =========================================================

def safe_video_basename(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def get_video_file_list(input_dir: str) -> List[str]:
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(video_files)


def get_video_metadata(video_path: str) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, width, height


def parse_label_from_filename(video_name: str) -> Optional[int]:
    parts = video_name.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except ValueError:
            return np.nan
    return np.nan


def make_output_id(video_name: str) -> str:
    """예: 01_박민희_0_250110_v3 → 01_0"""
    parts = video_name.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[2]}"
    return video_name


def get_output_csv_path(video_path: str) -> str:
    video_name = safe_video_basename(video_path)
    output_id  = make_output_id(video_name)
    return os.path.join(OUTPUT_ROOT_DIR, f"{output_id}.csv")

# =========================================================
# 4. NaN 처리
# =========================================================

def interpolate_short_gaps(series: pd.Series, max_gap: int = 5) -> pd.Series:
    s = series.copy()
    if s.isna().sum() == 0:
        return s

    is_nan = s.isna().to_numpy()
    n = len(s)
    start = None
    gap_segments = []

    for i in range(n):
        if is_nan[i] and start is None:
            start = i
        elif not is_nan[i] and start is not None:
            gap_segments.append((start, i - 1))
            start = None
    if start is not None:
        gap_segments.append((start, n - 1))

    for gap_start, gap_end in gap_segments:
        gap_len   = gap_end - gap_start + 1
        left_idx  = gap_start - 1
        right_idx = gap_end + 1
        has_left  = left_idx >= 0 and pd.notna(s.iloc[left_idx])
        has_right = right_idx < n and pd.notna(s.iloc[right_idx])

        if gap_len <= max_gap and has_left and has_right:
            sub = s.iloc[left_idx:right_idx + 1].copy()
            sub = sub.interpolate(method="linear")
            s.iloc[left_idx:right_idx + 1] = sub

    return s


def apply_short_gap_interpolation(df: pd.DataFrame, columns: List[str], max_gap: int) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = interpolate_short_gaps(out[col], max_gap=max_gap)
    return out

# =========================================================
# 5. 운동학 계산 함수
# =========================================================

def compute_velocity(x: pd.Series, y: pd.Series, dt: float) -> pd.Series:
    dx = x.diff()
    dy = y.diff()
    return np.sqrt(dx ** 2 + dy ** 2) / dt


def compute_acceleration(v: pd.Series, dt: float) -> pd.Series:
    return v.diff() / dt


def compute_jerk(a: pd.Series, dt: float) -> pd.Series:
    return a.diff() / dt

# =========================================================
# 6. GIE (Global Index of Efficiency) — 구버전 com_gie와 동일 수식
# =========================================================

def compute_gie(x: pd.Series, y: pd.Series) -> float:
    """
    GIE = (log(2L) - log(c)) / log(2)
    L = 누적 이동 거리
    c = ConvexHull 넓이 (경로가 차지하는 공간)
    값이 클수록 좁은 공간에서 많이 이동 → 효율적인 경로
    """
    coords = np.column_stack([x.to_numpy(), y.to_numpy()])
    valid  = coords[~np.isnan(coords).any(axis=1)]

    if len(valid) < 3:
        return np.nan

    diffs        = np.diff(valid, axis=0)
    step_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    L            = np.sum(step_lengths)

    if L <= 0:
        return np.nan

    try:
        hull = ConvexHull(valid)
        c    = hull.area
    except Exception:
        return np.nan

    if c <= 0:
        return np.nan

    return float((np.log(2 * L) - np.log(c)) / np.log(2))

# =========================================================
# 7. Path Efficiency — 직선거리 / 누적거리
# =========================================================

def compute_path_efficiency(x: pd.Series, y: pd.Series) -> float:
    """
    path_efficiency = 직선 거리(시작→끝) / 누적 이동 거리
    1에 가까울수록 직선에 가깝게 이동
    """
    coords = np.column_stack([x.to_numpy(), y.to_numpy()])
    valid  = coords[~np.isnan(coords).any(axis=1)]

    if len(valid) < 2:
        return np.nan

    diffs        = np.diff(valid, axis=0)
    step_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    total_path   = np.sum(step_lengths)

    if total_path <= 0:
        return np.nan

    straight_line = np.sqrt(np.sum((valid[-1] - valid[0]) ** 2))
    return float(straight_line / total_path)

# =========================================================
# 8. 요약 통계
# =========================================================

def summarize_series(series: pd.Series, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(series.mean(skipna=True)),
        f"{prefix}_max":  float(series.max(skipna=True)),
        f"{prefix}_sd":   float(series.std(skipna=True)),
    }

# =========================================================
# 9. Raw landmark 추출
# =========================================================

def extract_raw_landmarks_from_video(video_path: str) -> pd.DataFrame:
    fps, frame_count, width, height = get_video_metadata(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 열기 실패: {video_path}")

    rows = []

    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        output_segmentation_masks=False,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx     = 0
        processed_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_INTERVAL != 0:
                frame_idx += 1
                continue

            rgb               = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image          = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms = int((frame_idx / fps) * 1000)
            result            = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            row = {
                "video_name":    safe_video_basename(video_path),
                "frame_idx":     frame_idx,
                "processed_idx": processed_idx,
                "timestamp_sec": frame_idx / fps if fps > 0 else np.nan,
                "fps":           fps,
                "frame_width":   width,
                "frame_height":  height,
            }

            if result.pose_landmarks is None or len(result.pose_landmarks) == 0:
                for joint_name in LANDMARK_MAP_RAW:
                    row[f"{joint_name}_x"] = np.nan
                    row[f"{joint_name}_y"] = np.nan
                    if SAVE_VISIBILITY:
                        row[f"{joint_name}_visibility"] = np.nan
            else:
                landmarks = result.pose_landmarks[0]
                for joint_name, idx in LANDMARK_MAP_RAW.items():
                    lm         = landmarks[idx]
                    visibility = float(getattr(lm, "visibility", np.nan))

                    if np.isnan(visibility) or visibility < VISIBILITY_MIN:
                        x, y = np.nan, np.nan
                    else:
                        if USE_PIXEL_COORDINATES:
                            x = lm.x * width
                            y = lm.y * height
                        else:
                            x = lm.x
                            y = lm.y

                    row[f"{joint_name}_x"] = x
                    row[f"{joint_name}_y"] = y
                    if SAVE_VISIBILITY:
                        row[f"{joint_name}_visibility"] = visibility

            rows.append(row)
            processed_idx += 1
            frame_idx     += 1

    cap.release()
    return pd.DataFrame(rows)

# =========================================================
# 10. 신체 정규화
# =========================================================

def add_body_size_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["shoulder_width"] = np.sqrt(
        (out["right_shoulder_x"] - out["left_shoulder_x"]) ** 2 +
        (out["right_shoulder_y"] - out["left_shoulder_y"]) ** 2
    )
    out["hip_width"] = np.sqrt(
        (out["right_hip_x"] - out["left_hip_x"]) ** 2 +
        (out["right_hip_y"] - out["left_hip_y"]) ** 2
    )
    out["body_size"] = (out["shoulder_width"] + out["hip_width"]) / 2.0
    return out


def compute_body_size_median(df: pd.DataFrame) -> float:
    if "body_size" not in df.columns:
        return np.nan
    valid = df["body_size"].dropna()
    valid = valid[valid > 0]
    if len(valid) == 0:
        return np.nan
    return float(valid.median())


def normalize_coordinates_inplace(df: pd.DataFrame, body_size_median: float) -> pd.DataFrame:
    out = df.copy()
    if pd.isna(body_size_median) or body_size_median <= 0:
        for joint in LANDMARK_MAP_RAW:
            out[f"{joint}_x"] = np.nan
            out[f"{joint}_y"] = np.nan
        return out

    for joint in LANDMARK_MAP_RAW:
        out[f"{joint}_x"] = out[f"{joint}_x"] / body_size_median
        out[f"{joint}_y"] = out[f"{joint}_y"] / body_size_median
    return out

# =========================================================
# 11. center (shoulder_center, hip_center) 좌표 추가
#     한쪽이라도 NaN이면 center도 NaN
# =========================================================

def add_center_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for left, right, center in [
        ("left_shoulder", "right_shoulder", "shoulder_center"),
        ("left_hip",      "right_hip",      "hip_center"),
    ]:
        both_valid = out[f"{left}_x"].notna() & out[f"{right}_x"].notna()

        out[f"{center}_x"] = np.where(
            both_valid,
            (out[f"{left}_x"] + out[f"{right}_x"]) / 2.0,
            np.nan
        )
        out[f"{center}_y"] = np.where(
            both_valid,
            (out[f"{left}_y"] + out[f"{right}_y"]) / 2.0,
            np.nan
        )

    return out

# =========================================================
# 12. 보간 대상 컬럼 목록
# =========================================================

def get_xy_columns_for_interpolation() -> List[str]:
    cols = []
    for joint in JOINTS_RAW:
        cols += [f"{joint}_x", f"{joint}_y"]
    for c in JOINTS_CENTER:
        cols += [f"{c}_x", f"{c}_y"]
    return cols

# =========================================================
# 13. time-series 변수 생성
# =========================================================

def build_timeseries_variables(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    if len(df_raw) == 0:
        return pd.DataFrame(), np.nan

    out = df_raw.copy()

    # 신체 크기 정규화
    out = add_body_size_columns(out)
    body_size_median = compute_body_size_median(out)
    out = normalize_coordinates_inplace(out, body_size_median)

    # center 좌표 추가 (한쪽 NaN → center NaN)
    out = add_center_columns(out)

    # 보간
    interp_cols = get_xy_columns_for_interpolation()
    out = apply_short_gap_interpolation(out, interp_cols, max_gap=MAX_INTERP_GAP)

    fps_values = out["fps"].dropna().unique()
    fps = fps_values[0] if len(fps_values) > 0 and fps_values[0] > 0 else 30.0
    dt  = 1.0 / fps

    # ── 8개 관절: velocity + acceleration만 (jerk 없음) ──
    for joint in JOINTS_RAW:
        v_col = f"{joint}_velocity"
        a_col = f"{joint}_acceleration"
        out[v_col] = compute_velocity(out[f"{joint}_x"], out[f"{joint}_y"], dt)
        out[a_col] = compute_acceleration(out[v_col], dt)

    # ── 2개 center: velocity + acceleration + jerk ──
    for joint in JOINTS_CENTER:
        v_col = f"{joint}_velocity"
        a_col = f"{joint}_acceleration"
        j_col = f"{joint}_jerk"
        out[v_col] = compute_velocity(out[f"{joint}_x"], out[f"{joint}_y"], dt)
        out[a_col] = compute_acceleration(out[v_col], dt)
        out[j_col] = compute_jerk(out[a_col], dt)

    # ── symmetry (shoulder, hip) ──
    out["shoulder_x_symmetry"] = (out["left_shoulder_x"] - out["right_shoulder_x"]).abs()
    out["shoulder_y_symmetry"] = (out["left_shoulder_y"] - out["right_shoulder_y"]).abs()
    out["hip_x_symmetry"]      = (out["left_hip_x"]      - out["right_hip_x"]).abs()
    out["hip_y_symmetry"]      = (out["left_hip_y"]      - out["right_hip_y"]).abs()

    return out, body_size_median

# =========================================================
# 14. summary feature 생성 (86개)
# =========================================================

def build_summary_features(df_ts: pd.DataFrame, body_size_median: float) -> pd.DataFrame:
    if len(df_ts) == 0:
        return pd.DataFrame()

    summary: Dict[str, float] = {}

    video_name = df_ts["video_name"].iloc[0]
    summary["id"]              = make_output_id(video_name)
    summary["label"]           = parse_label_from_filename(video_name) if AUTO_PARSE_LABEL_FROM_FILENAME else np.nan
    summary["body_size_median"] = body_size_median

    # ────────────────────────────────────────────────
    # A. 운동학 — 8개 관절: velocity + acceleration (각 mean/max/sd) = 48개
    # ────────────────────────────────────────────────
    for joint in JOINTS_RAW:
        summary.update(summarize_series(df_ts[f"{joint}_velocity"],     f"{joint}_velocity"))
        summary.update(summarize_series(df_ts[f"{joint}_acceleration"], f"{joint}_acceleration"))

    # ────────────────────────────────────────────────
    # B. 운동학 — 2개 center: velocity + acceleration + jerk (각 mean/max/sd) = 18개
    # ────────────────────────────────────────────────
    for joint in JOINTS_CENTER:
        summary.update(summarize_series(df_ts[f"{joint}_velocity"],     f"{joint}_velocity"))
        summary.update(summarize_series(df_ts[f"{joint}_acceleration"], f"{joint}_acceleration"))
        summary.update(summarize_series(df_ts[f"{joint}_jerk"],         f"{joint}_jerk"))

    # ────────────────────────────────────────────────
    # C. center 추가변수 8개
    # ────────────────────────────────────────────────
    # GIE (ConvexHull 기반 — 구버전 com_gie와 동일 수식)
    summary["hip_center_gie"]      = compute_gie(df_ts["hip_center_x"],      df_ts["hip_center_y"])
    summary["shoulder_center_gie"] = compute_gie(df_ts["shoulder_center_x"], df_ts["shoulder_center_y"])

    # Path Efficiency (직선거리 / 누적거리)
    summary["hip_center_path_efficiency"]      = compute_path_efficiency(df_ts["hip_center_x"],      df_ts["hip_center_y"])
    summary["shoulder_center_path_efficiency"] = compute_path_efficiency(df_ts["shoulder_center_x"], df_ts["shoulder_center_y"])

    # Sway (좌표 std)
    summary["hip_center_horizontal_sway"]      = float(df_ts["hip_center_x"].std(skipna=True))
    summary["hip_center_vertical_sway"]        = float(df_ts["hip_center_y"].std(skipna=True))
    summary["shoulder_center_horizontal_sway"] = float(df_ts["shoulder_center_x"].std(skipna=True))
    summary["shoulder_center_vertical_sway"]   = float(df_ts["shoulder_center_y"].std(skipna=True))

    # ────────────────────────────────────────────────
    # D. symmetry 12개
    # ────────────────────────────────────────────────
    for var in ["shoulder_x_symmetry", "shoulder_y_symmetry",
                "hip_x_symmetry",      "hip_y_symmetry"]:
        summary.update(summarize_series(df_ts[var], var))

    # 변수 수 검증
    feature_cols = [c for c in summary if c not in ("id", "label", "body_size_median")]
    assert len(feature_cols) == 86, f"변수 수 오류: {len(feature_cols)}개 (기대값: 86)"

    return pd.DataFrame([summary])

# =========================================================
# 15. 저장 함수
# =========================================================

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")

# =========================================================
# 16. 영상 1개 처리
# =========================================================

def process_single_video(video_path: str) -> Optional[pd.DataFrame]:
    video_name = safe_video_basename(video_path)
    print("=" * 80)
    print(f"[처리 시작] {video_name}")

    try:
        df_raw = extract_raw_landmarks_from_video(video_path)
        if df_raw.empty:
            print(f"[경고] raw landmark 추출 결과가 비어 있습니다: {video_name}")
            return None

        df_ts, body_size_median = build_timeseries_variables(df_raw)
        if df_ts.empty:
            print(f"[경고] time-series 변수 생성 결과가 비어 있습니다: {video_name}")
            return None

        df_summary = build_summary_features(df_ts, body_size_median)
        if df_summary.empty:
            print(f"[경고] summary feature 생성 결과가 비어 있습니다: {video_name}")
            return None

        feature_cols = [c for c in df_summary.columns if c not in ("id", "label", "body_size_median")]
        print(f"[변수 수] {len(feature_cols)}개 (기대값: 86)")

        save_path = get_output_csv_path(video_path)
        save_dataframe(df_summary, save_path)
        print(f"[저장 완료] {save_path}")
        print(f"[처리 완료] {video_name}")
        return df_summary

    except Exception as e:
        print(f"[오류] {video_name} 처리 중 예외 발생")
        print(str(e))
        traceback.print_exc()
        return None

# =========================================================
# 17. 전체 영상 처리
# =========================================================

def process_all_videos() -> None:
    ensure_directories()

    if not os.path.exists(POSE_MODEL_PATH):
        raise FileNotFoundError(
            f"Pose Landmarker 모델 파일을 찾을 수 없습니다: {POSE_MODEL_PATH}\n"
            "예: ./models/pose_landmarker_full.task"
        )

    video_files = get_video_file_list(INPUT_VIDEO_DIR)
    if len(video_files) == 0:
        print(f"[안내] 입력 폴더에 영상이 없습니다: {INPUT_VIDEO_DIR}")
        return

    print(f"[안내] 총 {len(video_files)}개 영상 발견")

    success_count = 0
    fail_count    = 0

    for i, video_path in enumerate(video_files, start=1):
        print(f"\n[{i}/{len(video_files)}] 처리 중...")
        result = process_single_video(video_path)
        if result is None or result.empty:
            fail_count += 1
        else:
            success_count += 1

    print("\n" + "=" * 80)
    print("분석 완료")
    print(f"성공: {success_count}개 | 실패: {fail_count}개")
    print(f"저장 폴더: {OUTPUT_ROOT_DIR}")
    print("=" * 80)

# =========================================================
# 18. 메인 실행
# =========================================================

if __name__ == "__main__":
    process_all_videos()