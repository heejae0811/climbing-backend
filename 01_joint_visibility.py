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
OUTPUT_DIR = "./visibility/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_INTERVAL = 1
VISIBILITY_MIN = 0.5
POSE_MODEL_PATH = "./models/pose_landmarker_full.task"

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# ==========================================================
# 1. 단일 영상 관절 인식률 분석
# ==========================================================
def analyze_joint_visibility(video_path):
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"❌ Pose 모델 파일 없음 → {POSE_MODEL_PATH}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 열기 실패 → {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if fps <= 0 else fps

    total_processed_frames = 0
    pose_detected_frames   = 0

    visibility_sums    = np.zeros(33, dtype=float)
    visibility_sq_sums = np.zeros(33, dtype=float)   # 표준편차 계산용 제곱합
    recognized_counts  = np.zeros(33, dtype=int)

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

    frame_idx = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_INTERVAL != 0:
                frame_idx += 1
                continue

            total_processed_frames += 1

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp_image.Image(
                image_format=mp_image.ImageFormat.SRGB,
                data=rgb
            )
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_frame, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                pose_detected_frames += 1
                lm = result.pose_landmarks[0]

                for i in range(33):
                    vis = getattr(lm[i], "visibility", np.nan)

                    if np.isfinite(vis):
                        visibility_sums[i]    += vis
                        visibility_sq_sums[i] += vis ** 2
                        if vis >= VISIBILITY_MIN:
                            recognized_counts[i] += 1

            frame_idx += 1

    cap.release()

    if total_processed_frames == 0:
        print(f"❌ 처리된 프레임 없음 → {video_path}")
        return None

    rows = []
    n = total_processed_frames

    for i, name in enumerate(LANDMARK_NAMES):
        mean_vis = visibility_sums[i] / n

        # 표본 표준편차 (n-1 사용, n=1이면 0)
        if n > 1:
            variance = (visibility_sq_sums[i] - n * mean_vis ** 2) / (n - 1)
            std_vis  = np.sqrt(max(variance, 0.0))
        else:
            std_vis  = 0.0

        recognition_rate = (recognized_counts[i] / n) * 100.0

        rows.append({
            "video_name":              os.path.basename(video_path),
            "landmark_id":             i,
            "landmark_name":           name,
            "total_processed_frames":  total_processed_frames,
            "pose_detected_frames":    pose_detected_frames,
            "recognized_frames":       int(recognized_counts[i]),
            "average_visibility":      round(float(mean_vis), 2),
            "std_visibility":          round(float(std_vis), 2),
            "recognition_rate_percent":round(float(recognition_rate), 2)
        })

    df = pd.DataFrame(rows)
    return df


# ==========================================================
# 2. 전체 영상 분석
# ==========================================================
def main():
    files = (
        glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) +
        glob.glob(os.path.join(VIDEO_DIR, "*.mov")) +
        glob.glob(os.path.join(VIDEO_DIR, "*.avi")) +
        glob.glob(os.path.join(VIDEO_DIR, "*.mkv"))
    )

    if not files:
        print("❌ 분석할 비디오가 없습니다.")
        return

    print(f"📁 총 {len(files)}개 비디오 분석 시작\n")

    all_results   = []
    success_count = 0
    fail_count    = 0

    for idx, video_path in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{idx}/{len(files)}] {os.path.basename(video_path)}")

        df = analyze_joint_visibility(video_path)

        if df is None:
            print(f"  ❌ 분석 실패 → {video_path}\n")
            fail_count += 1
            continue

        all_results.append(df)
        success_count += 1

    # ======================================================
    # 3. 전체 평균 summary 저장
    # ======================================================
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        final_save = os.path.join(OUTPUT_DIR, "all_videos_joint_visibility.csv")
        final_df.to_csv(final_save, index=False, encoding="utf-8-sig")

        summary_df = (
            final_df.groupby(["landmark_id", "landmark_name"], as_index=False)
            .agg(
                average_visibility_mean    = ("average_visibility",       "mean"),
                average_visibility_std     = ("average_visibility",       "std"),
                recognition_rate_mean      = ("recognition_rate_percent", "mean"),
                recognition_rate_std       = ("recognition_rate_percent", "std"),
            )
            .sort_values(by="recognition_rate_mean", ascending=False)
        )

        # 소수점 2자리 통일
        for col in ["average_visibility_mean", "average_visibility_std",
                    "recognition_rate_mean",   "recognition_rate_std"]:
            summary_df[col] = summary_df[col].round(2)

        summary_save = os.path.join(OUTPUT_DIR, "joint_visibility_summary.xlsx")
        with pd.ExcelWriter(summary_save, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")

            # 열 너비 자동 조정
            ws = writer.sheets["Summary"]
            for col in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0
                              for cell in col)
                ws.column_dimensions[col[0].column_letter].width = max_len + 4

        print("=" * 60)
        print("🎉 관절 인식률 분석 완료!")
        print(f"   성공: {success_count}개 | 실패: {fail_count}개")
        print(f"   전체 결과: {final_save}")
        print(f"   평균 요약: {summary_save}")
        print("=" * 60)

        print("\n📊 인식률 상위 10개 관절")
        print(summary_df.head(10).to_string(index=False))
    else:
        print("❌ 저장할 결과가 없습니다.")


if __name__ == "__main__":
    main()