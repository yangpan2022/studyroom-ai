"""
scripts/extract_frames.py — Extract frames from a video file.

Usage:
    python scripts/extract_frames.py --video data/raw_videos/room.mp4
    python scripts/extract_frames.py --video data/raw_videos/room.mp4 --interval 15
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.utils.paths import ensure_dirs, get_frames_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_frames(video_path: Path, interval: int = config.FRAME_EXTRACTION_INTERVAL) -> int:
    """Extract frames from *video_path* at every *interval*-th frame.

    Saves frames as ``frame_XXXXXX.jpg`` in ``data/frames/<video-stem>/``.

    Args:
        video_path: Absolute or relative path to the input video file.
        interval:   Save one frame every *interval* frames.

    Returns:
        Total number of frames saved.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError:       If OpenCV cannot open the video.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_dir: Path = get_frames_dir(video_path.stem)
    logger.info("Saving frames to %s (every %d frames)", output_dir, interval)

    frame_index: int = 0
    saved_count: int = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval == 0:
            filename = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    logger.info("Done — %d frames saved from %d total.", saved_count, frame_index)
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a CCTV-style video.")
    parser.add_argument("--video", required=True, type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--interval", type=int, default=config.FRAME_EXTRACTION_INTERVAL,
        help=f"Save every Nth frame (default: {config.FRAME_EXTRACTION_INTERVAL}).",
    )
    args = parser.parse_args()
    ensure_dirs()
    extract_frames(args.video, args.interval)


if __name__ == "__main__":
    main()
