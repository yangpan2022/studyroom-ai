"""
training/trainer.py — YOLO model training wrapper.

Wraps the Ultralytics training API so the rest of the codebase only
imports from this module and never calls Ultralytics directly.
"""

import logging
from pathlib import Path

from ultralytics import YOLO

from app import config

logger = logging.getLogger(__name__)


def train_model(
    dataset_yaml: Path,
    base_weights: Path | None = None,
    epochs: int = config.TRAIN_EPOCHS,
    image_size: int = config.TRAIN_IMAGE_SIZE,
    batch: int = config.TRAIN_BATCH_SIZE,
    device: str = config.TRAIN_DEVICE,
    project_dir: Path | None = None,
    run_name: str = "studyroom_yolo",
) -> Path:
    """Fine-tune (or train from scratch) a YOLO model.

    The function uses ``base_weights`` as the starting checkpoint (transfer
    learning).  When ``base_weights`` is *None* it falls back to the value
    defined in :mod:`app.config`.

    Args:
        dataset_yaml: Path to the YOLO-format ``dataset.yaml`` file.
        base_weights: Starting ``.pt`` file.  ``None`` → ``config.DEFAULT_MODEL_WEIGHTS``.
        epochs:       Number of training epochs.
        image_size:   Input resolution (square, in pixels).
        batch:        Batch size (use ``-1`` for auto-batch).
        device:       Compute device — ``"cpu"``, ``"0"``, ``"0,1"``, etc.
        project_dir:  Directory where Ultralytics saves run artefacts.
                      Defaults to ``models/trained/``.
        run_name:     Sub-directory name for this training run.

    Returns:
        Path to the ``best.pt`` weights produced by this run.

    Raises:
        FileNotFoundError: If *dataset_yaml* does not exist.
    """
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    weights: Path = base_weights or config.DEFAULT_MODEL_WEIGHTS
    save_dir: Path = project_dir or config.TRAINED_DIR

    logger.info(
        "Starting training — weights=%s  epochs=%d  imgsz=%d  device=%s",
        weights, epochs, image_size, device,
    )

    model = YOLO(str(weights))
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        project=str(save_dir),
        name=run_name,
        exist_ok=True,     # Allow re-running without removing old results
        verbose=True,
    )

    best_weights: Path = save_dir / run_name / "weights" / "best.pt"
    logger.info("Training complete. Best weights → %s", best_weights)
    return best_weights
