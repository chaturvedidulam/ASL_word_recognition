from pathlib import Path
import json

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from concurrent.futures import ProcessPoolExecutor

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
LANDMARK_DIR = DATA_DIR / "landmarks"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# CONFIG
# ==========================================================

SEQUENCE_LENGTH = 64
NUM_SAMPLES = 10000  # Change to 10000 after testing
RANDOM_STATE = 42
NUM_HANDS = 2
_worker_detector = None  # Global variable for multiprocessing


# ==========================================================
# MEDIAPIPE
# ==========================================================

def create_detector():
    """Create MediaPipe hand detector."""
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=NUM_HANDS,
    )
    return vision.HandLandmarker.create_from_options(options)

def init_worker():
    global _worker_detector

    _worker_detector = create_detector()

# ==========================================================
# VIDEO LOADING
# ==========================================================

def load_video(video_path):
    """Load all frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    return frames


# ==========================================================
# FRAME SAMPLING
# ==========================================================

def sample_video_frames(frames, sequence_length=SEQUENCE_LENGTH):
    """Uniformly sample frames."""
    if len(frames) == 0:
        raise ValueError("Video contains no frames")

    indices = np.linspace(0, len(frames) - 1, sequence_length).astype(int)
    return [frames[idx] for idx in indices]


# ==========================================================
# LANDMARK EXTRACTION
# ==========================================================

def extract_landmarks(frame, detector):
    """
    Extract two-hand landmarks.

    Output:
        (126,)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if result.hand_landmarks:
        for hand_idx, hand in enumerate(result.hand_landmarks[:2]):
            coords = []
            for landmark in hand:
                coords.extend([landmark.x, landmark.y, landmark.z])

            coords = np.asarray(coords, dtype=np.float32)

            if hand_idx == 0:
                left_hand = coords
            elif hand_idx == 1:
                right_hand = coords

    return np.concatenate([left_hand, right_hand])


# ==========================================================
# VIDEO -> SEQUENCE
# ==========================================================

def process_video(row_dict):
    global _worker_detector

    detector = _worker_detector

    try:
        video_id = str(
            row_dict["video_id"]
        ).zfill(5)

        video_path = (
            DATA_DIR
            / "videos"
            / f"{video_id}.mp4"
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"{video_path} not found"
            )

        sequence = video_to_sequence(
            video_path,
            detector,
        )

        return (
            sequence,
            row_dict["label"],
            None,
        )

    except Exception as exc:

        return (
            None,
            None,
            {
                "video_id": row_dict["video_id"],
                "error": str(exc),
            },
        )

def video_to_sequence(video_path, detector):
    """
    Convert a video into:

    (64, 126)
    """
    frames = load_video(video_path)
    sampled_frames = sample_video_frames(frames)

    sequence = []
    for frame in sampled_frames:
        landmarks = extract_landmarks(frame, detector)
        sequence.append(landmarks)

    return np.asarray(sequence, dtype=np.float32)


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("Loading manifest...")
    manifest = pd.read_csv(MANIFEST_PATH)

    subset = manifest.sample(
        n=min(NUM_SAMPLES, len(manifest)),
        random_state=RANDOM_STATE,
    )
    subset.to_csv(LANDMARK_DIR / "sampled_manifest.csv", index=False)

    print(f"Selected {len(subset)} videos")
    print(f"Unique classes: {subset['label'].nunique()}")

    # --------------------------------------------------
    # Label mappings
    # --------------------------------------------------
    label_to_gloss = (
        manifest
        .drop_duplicates("label")
        .set_index("label")["gloss"]
        .to_dict()
    )
    gloss_to_label = {v: k for k, v in label_to_gloss.items()}

    with open(LANDMARK_DIR / "label_to_gloss.json", "w") as file:
        json.dump(
            {str(k): v for k, v in label_to_gloss.items()},
            file,
            indent=4,
        )

    with open(LANDMARK_DIR / "gloss_to_label.json", "w") as file:
        json.dump(gloss_to_label, file, indent=4)

    detector = create_detector()
    X = []
    y = []
    failed_videos = []

    print("\nStarting landmark extraction...\n")

    rows = subset.to_dict(
    orient="records"
    )

    with ProcessPoolExecutor(
        max_workers=6,
        initializer=init_worker,
    ) as executor:

        results = executor.map(
            process_video,
            rows,
        )

        for (
            sequence,
            label,
            error,
        ) in tqdm(
            results,
            total=len(rows),
        ):

            if error is not None:

                failed_videos.append(
                    error
                )

                continue
            
            X.append(sequence)
            y.append(label)
            if len(X) % 500 == 0:
                print(
                    f"\nProcessed {len(X)} videos"
                )

    print()
    print(f"Successful videos: {len(X)}")
    print(f"Failed videos: {len(failed_videos)}")

    if len(X) == 0:
        raise RuntimeError("No videos were processed.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    zero_frame_ratio = np.mean(np.all(X == 0, axis=-1))

    print(f"Zero frame ratio: {zero_frame_ratio:.4f}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    np.save(LANDMARK_DIR / f"X_{NUM_SAMPLES}.npy", X)
    np.save(LANDMARK_DIR / f"y_{NUM_SAMPLES}.npy", y)

    with open(LANDMARK_DIR / "failed_videos.json", "w") as file:
        json.dump(failed_videos, file, indent=4)

    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": 126,
        "num_hands": NUM_HANDS,
        "requested_samples": NUM_SAMPLES,
        "successful_samples": len(X),
        "failed_samples": len(failed_videos),
        "zero_frame_ratio": float(zero_frame_ratio),
    }

    with open(LANDMARK_DIR / "extraction_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nDone.")


if __name__ == "__main__":
    main()