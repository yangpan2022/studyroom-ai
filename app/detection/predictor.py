"""
detection/predictor.py — Core YOLO video inference engine.

VideoPredictor loads a YOLO model and runs frame-by-frame detection on a
video file, applying post-processing logic and persisting annotated output.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from app import config
from app.detection.postprocess import (
    determine_seat_occupancy,
    detect_phone_usage,
    extract_boxes_by_class,
)
from app.utils.paths import ensure_dirs, get_output_path

logger = logging.getLogger(__name__)


class VideoPredictor:
    """Run YOLO inference on a video file and emit structured results.

    Args:
        model_path: Path to YOLO ``.pt`` weights file.
            Defaults to :attr:`config.DEFAULT_MODEL_WEIGHTS`.
        confidence: Minimum detection confidence score.
        iou: Non-maximum suppression IoU threshold.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        confidence: float = config.DETECTION_CONFIDENCE,
        iou: float = config.DETECTION_IOU,
    ) -> None:
        ensure_dirs()
        self.model_path: Path = model_path or config.DEFAULT_MODEL_WEIGHTS
        self.confidence: float = confidence
        self.iou: float = iou

        logger.info("Loading YOLO model from %s", self.model_path)
        self.model: YOLO = YOLO(str(self.model_path))

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def predict_video(
        self,
        video_path: Path,
        save_annotated: bool = True,
    ) -> list[dict[str, Any]]:
        """Run detection on every frame of *video_path*.

        Args:
            video_path: Absolute path to the input video file.
            save_annotated: If ``True``, write an annotated copy of the video
                and a JSON summary to ``outputs/``.

        Returns:
            A list of per-frame result dictionaries ready to send to the
            Spring Boot backend.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Prepare output video writer
        writer: cv2.VideoWriter | None = None
        output_video_path: Path | None = None
        if save_annotated:
            output_video_path = get_output_path(f"annotated_{video_path.name}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        all_frame_results: list[dict[str, Any]] = []
        frame_index: int = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_result = self._process_frame(frame, frame_index)
                all_frame_results.append(frame_result)

                if writer is not None:
                    annotated = self._draw_annotations(frame, frame_result)
                    writer.write(annotated)

                frame_index += 1

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if save_annotated:
            self._save_json_results(all_frame_results, video_path.stem)
            logger.info("Annotated video saved to %s", output_video_path)

        logger.info("Processed %d frames from %s", frame_index, video_path.name)
        return all_frame_results

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _process_frame(
        self, frame: "cv2.typing.MatLike", frame_index: int
    ) -> dict[str, Any]:
        """Run model inference on a single frame and build the result dict.

        Args:
            frame: BGR image array from OpenCV.
            frame_index: Zero-based frame number in the source video.

        Returns:
            Dictionary containing raw detections and post-processed status.
        """
        results: list[Results] = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )
        result: Results = results[0]

        # Parse bounding boxes into a plain Python structure
        detections: list[dict[str, Any]] = self._parse_detections(result)

        # Post-processing: occupancy & phone-use
        persons = extract_boxes_by_class(detections, "person")
        seats = extract_boxes_by_class(detections, "seat")
        phones = extract_boxes_by_class(detections, "phone")

        seat_statuses = determine_seat_occupancy(persons, seats)
        phone_alerts = detect_phone_usage(persons, phones)

        return {
            "frame_index": frame_index,
            "detections": detections,
            "seat_status": seat_statuses,
            "phone_alerts": phone_alerts,
        }

    @staticmethod
    def _parse_detections(result: Results) -> list[dict[str, Any]]:
        """Convert Ultralytics :class:`Results` to a serialisable list.

        Args:
            result: Single-image inference result from YOLO.

        Returns:
            List of detection dicts with keys ``class_name``, ``confidence``,
            and ``bbox`` (``[x1, y1, x2, y2]``).
        """
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names.get(class_id, str(class_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                }
            )
        return detections

    @staticmethod
    def _draw_annotations(
        frame: "cv2.typing.MatLike", frame_result: dict[str, Any]
    ) -> "cv2.typing.MatLike":
        """Draw bounding boxes and status labels on a copy of *frame*.

        Args:
            frame: Original BGR image.
            frame_result: Result dictionary produced by :meth:`_process_frame`.

        Returns:
            Annotated BGR image (deep copy of the input).
        """
        annotated = frame.copy()
        color_map = {
            "person": (0, 255, 0),
            "seat": (255, 165, 0),
            "phone": (0, 0, 255),
        }

        for det in frame_result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            color = color_map.get(det["class_name"], (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        # Overlay seat statuses
        for idx, status in enumerate(frame_result["seat_status"]):
            text = f"Seat {idx}: {status['status']}"
            cv2.putText(
                annotated, text, (10, 30 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        # Overlay phone alerts
        for alert in frame_result["phone_alerts"]:
            cx, cy = alert["person_center"]
            cv2.putText(
                annotated, "PHONE USE", (cx, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        return annotated

    @staticmethod
    def _save_json_results(
        all_frame_results: list[dict[str, Any]], video_stem: str
    ) -> None:
        """Persist all frame results as a JSON file in ``outputs/``.

        Args:
            all_frame_results: List of per-frame result dicts.
            video_stem: Filename stem of the source video (used for naming).
        """
        output_path = get_output_path(f"{video_stem}_results.json")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_frame_results, f, indent=2, ensure_ascii=False)
        logger.info("JSON results saved to %s", output_path)
