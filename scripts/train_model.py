"""
scripts/train_model.py — Start YOLO model training.

Usage:
    python scripts/train_model.py --data data/datasets/studyroom/dataset.yaml
    python scripts/train_model.py --data data/datasets/studyroom/dataset.yaml \\
        --weights models/pretrained/yolov8n.pt \\
        --epochs 100 --batch 8 --device 0
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.training.dataset_check import check_dataset, print_dataset_summary
from app.training.trainer import train_model
from app.utils.paths import ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO model for study-room detection.")
    parser.add_argument(
        "--data", required=True, type=Path,
        help="Path to dataset.yaml (YOLO format).",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Starting weights .pt file (default: config.DEFAULT_MODEL_WEIGHTS).",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.TRAIN_EPOCHS,
        help=f"Training epochs (default: {config.TRAIN_EPOCHS}).",
    )
    parser.add_argument(
        "--batch", type=int, default=config.TRAIN_BATCH_SIZE,
        help=f"Batch size (default: {config.TRAIN_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device", type=str, default=config.TRAIN_DEVICE,
        help=f"Compute device, e.g. 'cpu' or '0' (default: {config.TRAIN_DEVICE}).",
    )
    parser.add_argument(
        "--name", type=str, default="studyroom_yolo",
        help="Run name (sub-directory under models/trained/).",
    )
    parser.add_argument(
        "--skip-check", action="store_true",
        help="Skip dataset validation step.",
    )
    args = parser.parse_args()

    ensure_dirs()

    dataset_root: Path = args.data.parent

    if not args.skip_check:
        print("\n── Dataset Summary ──────────────────────────────────")
        print_dataset_summary(dataset_root)
        print("─────────────────────────────────────────────────────\n")
        ok = check_dataset(dataset_root)
        if not ok:
            logger.error("Dataset validation failed. Fix issues or use --skip-check to bypass.")
            sys.exit(1)

    best_weights = train_model(
        dataset_yaml=args.data,
        base_weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        run_name=args.name,
    )

    print(f"\n✅ Training complete. Best weights saved to:\n   {best_weights}\n")


if __name__ == "__main__":
    main()
