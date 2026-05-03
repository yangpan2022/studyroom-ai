"""
training/dataset_check.py — YOLO dataset structure validator.

Validates that a dataset directory follows the expected YOLO format before
kicking off a training run, giving early and clear error messages.

Expected layout
---------------
<dataset_root>/
    dataset.yaml
    images/
        train/   *.jpg / *.png
        val/     *.jpg / *.png
    labels/
        train/   *.txt
        val/     *.txt
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Recognised image extensions
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def check_dataset(dataset_root: Path) -> bool:
    """Validate a YOLO-format dataset directory.

    Checks:
    1. ``dataset.yaml`` exists and is parseable.
    2. ``images/train`` and ``images/val`` sub-directories exist and are non-empty.
    3. Every image file has a matching label ``.txt`` in ``labels/<split>/``.
    4. Class names in YAML match :attr:`config.CLASS_NAMES`.

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        ``True`` if **all** checks pass, ``False`` otherwise.
        All issues are *logged* rather than raised so callers can decide
        whether to abort or proceed with warnings.
    """
    valid: bool = True

    # ── 1. YAML file ──────────────────────────────────────────────────────────
    yaml_path: Path = dataset_root / "dataset.yaml"
    if not yaml_path.exists():
        logger.error("Missing dataset.yaml in %s", dataset_root)
        return False

    with yaml_path.open("r", encoding="utf-8") as f:
        try:
            meta: dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error("dataset.yaml parse error: %s", exc)
            return False

    logger.info("dataset.yaml loaded — nc=%s  names=%s", meta.get("nc"), meta.get("names"))

    # ── 2. Split directories ───────────────────────────────────────────────────
    for split in ("train", "val"):
        img_dir: Path = dataset_root / "images" / split
        lbl_dir: Path = dataset_root / "labels" / split

        if not img_dir.is_dir():
            logger.error("Missing images/%s directory", split)
            valid = False
            continue

        if not lbl_dir.is_dir():
            logger.error("Missing labels/%s directory", split)
            valid = False
            continue

        # ── 3. Image ↔ label pairing ─────────────────────────────────────────
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            logger.warning("No images found in images/%s", split)
            valid = False

        missing_labels: list[str] = []
        for img_path in images:
            label_path: Path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                missing_labels.append(img_path.name)

        if missing_labels:
            logger.error(
                "[%s] %d image(s) have no matching label file: %s …",
                split, len(missing_labels), missing_labels[:5],
            )
            valid = False
        else:
            logger.info("[%s] Checked %d images — all label files present.", split, len(images))

    # ── 4. Class name cross-check ─────────────────────────────────────────────
    from app import config  # late import to keep this module importable standalone

    yaml_names: list[str] = meta.get("names", [])
    if set(yaml_names) != set(config.CLASS_NAMES):
        logger.warning(
            "Class name mismatch — YAML: %s | config: %s",
            yaml_names, config.CLASS_NAMES,
        )
        # Warning only; do not fail — user may have extra classes intentionally

    if valid:
        logger.info("Dataset validation PASSED: %s", dataset_root)
    else:
        logger.error("Dataset validation FAILED: fix the issues above before training.")

    return valid


def print_dataset_summary(dataset_root: Path) -> None:
    """Print a quick human-readable summary of dataset counts.

    Args:
        dataset_root: Root directory of the dataset.
    """
    for split in ("train", "val", "test"):
        img_dir: Path = dataset_root / "images" / split
        if not img_dir.is_dir():
            continue
        count = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"  [{split:5s}]  {count:5d} images  →  {img_dir}")
