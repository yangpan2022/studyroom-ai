"""
utils/paths.py — Directory bootstrapping utilities.

Call `ensure_dirs()` at application startup to guarantee every
required directory exists before the rest of the code runs.
"""

from pathlib import Path

from app import config


def ensure_dirs() -> None:
    """Create all project directories that are expected to exist.

    Safe to call multiple times (uses exist_ok=True).
    """
    directories: list[Path] = [
        config.RAW_VIDEO_DIR,
        config.FRAMES_DIR,
        config.DATASETS_DIR,
        config.PRETRAINED_DIR,
        config.TRAINED_DIR,
        config.OUTPUTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[paths] Ensured directory: {directory}")


def get_output_path(filename: str) -> Path:
    """Return a full path inside the outputs directory.

    Args:
        filename: Desired output filename (e.g. ``"result.mp4"``).

    Returns:
        Absolute :class:`pathlib.Path` pointing to the output file.
    """
    return config.OUTPUTS_DIR / filename


def get_frames_dir(video_stem: str) -> Path:
    """Return the frame-extraction target directory for a given video.

    A sub-directory named after the video (without extension) is created
    inside ``data/frames/``, keeping frames organised per video.

    Args:
        video_stem: Filename stem of the source video (e.g. ``"room_cam_01"``).

    Returns:
        Absolute :class:`pathlib.Path` to the per-video frames folder.
    """
    target: Path = config.FRAMES_DIR / video_stem
    target.mkdir(parents=True, exist_ok=True)
    return target
