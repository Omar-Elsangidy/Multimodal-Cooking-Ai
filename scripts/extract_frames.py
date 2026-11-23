"""Extract evenly spaced RGB frames from cooking videos for YOLO labeling.

Samples frames at a target FPS (default: 1.0 FPS) from all videos in an input
directory so adjacent frames aren't near-duplicates. Outputs `.jpg` images ready
to upload to Roboflow / CVAT / Label Studio.

Example:
    python scripts/extract_frames.py --videos-dir raw_videos --out-dir datasets/raw_frames --fps 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_from_video(video_path: Path, out_dir: Path, target_fps: float, max_frames: int | None) -> int:
    """Sample frames from a single video at `target_fps` and save as JPEG."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: could not open {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps)))
    saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            out_file = out_dir / f"{video_path.stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from cooking videos for YOLO annotation.")
    parser.add_argument("--videos-dir", type=str, required=True, help="Folder containing .mp4/.mov/.avi videos.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="datasets/raw_frames",
        help="Destination folder for extracted .jpg frames.",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample (default: 1.0).")
    parser.add_argument(
        "--max-per-video",
        type=int,
        default=None,
        help="Optional cap on extracted frames per video.",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    videos_dir = (workspace / args.videos_dir).resolve() if not Path(args.videos_dir).is_absolute() else Path(args.videos_dir)
    out_dir = (workspace / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    videos = sorted(p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No videos found in {videos_dir} (supported: {sorted(VIDEO_EXTENSIONS)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in videos:
        count = extract_from_video(video, out_dir, args.fps, args.max_per_video)
        total += count
        print(f"{video.name}: extracted {count} frames")

    print(f"Done. Saved {total} frames to {out_dir}")


if __name__ == "__main__":
    main()

