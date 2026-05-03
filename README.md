# studyroom-ai — AI Module

> YOLO-based detection module for the AI-Powered Study Room Management System.
> Detects **person**, **seat**, and **phone** from CCTV-style video footage and
> produces seat-occupancy + phone-usage results ready for the Spring Boot backend.

---

## Project Structure

```
studyroom-ai/
├── app/
│   ├── config.py              # Centralized paths & hyper-parameters
│   ├── utils/
│   │   └── paths.py           # Directory bootstrapping helpers
│   ├── detection/
│   │   ├── predictor.py       # VideoPredictor – core inference engine
│   │   └── postprocess.py     # Seat occupancy & phone-use logic
│   ├── training/
│   │   ├── trainer.py         # YOLO training wrapper
│   │   └── dataset_check.py   # Dataset structure validator
│   └── api/                   # (Placeholder) REST endpoints for Spring Boot
├── data/
│   ├── raw_videos/            # Drop CCTV .mp4 files here
│   ├── frames/                # Extracted frames (auto-created)
│   └── datasets/              # YOLO-format dataset(s)
├── models/
│   ├── pretrained/            # Base weights (e.g. yolov8n.pt)
│   └── trained/               # Fine-tuned weights (auto-saved here)
├── outputs/                   # Annotated videos + JSON result files
├── scripts/
│   ├── extract_frames.py      # Extract frames from a video
│   ├── run_predict_video.py   # Run detection on a video
│   └── train_model.py         # Kick off model training
├── tests/                     # Unit tests (pytest)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download / place model weights

Either download `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/ultralytics)
and place it in `models/pretrained/`, **or** let the scripts auto-download it on first run.

### 3. Run detection on a video

```bash
python scripts/run_predict_video.py --video data/raw_videos/room.mp4
```

Results are written to `outputs/`:
- `annotated_room.mp4` — video with bounding boxes drawn
- `room_results.json` — per-frame detection + status data

### 4. Extract frames (for dataset building)

```bash
python scripts/extract_frames.py --video data/raw_videos/room.mp4 --interval 30
```

Frames are saved to `data/frames/room/`.

### 5. Train a custom model

Prepare a YOLO-format dataset under `data/datasets/studyroom/` then run:

```bash
python scripts/train_model.py --data data/datasets/studyroom/dataset.yaml --epochs 50
```

Best weights are saved to `models/trained/studyroom_yolo/weights/best.pt`.

---

## Configuration

All tuneable constants live in **`app/config.py`** — edit there, not in scripts:

| Constant | Default | Purpose |
|---|---|---|
| `CLASS_NAMES` | `["person","seat","phone"]` | Detection classes |
| `DETECTION_CONFIDENCE` | `0.4` | Min confidence threshold |
| `SEAT_OCCUPANCY_IOU_THRESHOLD` | `0.10` | IoU to count seat as occupied |
| `PHONE_USE_DISTANCE_THRESHOLD` | `150 px` | Centroid distance for phone alert |
| `FRAME_EXTRACTION_INTERVAL` | `30` | Extract every Nth frame |
| `TRAIN_EPOCHS` | `50` | Default training epochs |
| `TRAIN_DEVICE` | `"cpu"` | Set to `"0"` for GPU |

---

## JSON Output Format (per frame)

```json
{
  "frame_index": 42,
  "detections": [
    {"class_name": "person", "confidence": 0.87, "bbox": [120, 80, 310, 420]},
    {"class_name": "seat",   "confidence": 0.76, "bbox": [100, 300, 350, 480]}
  ],
  "seat_status": [
    {"seat_bbox": [100, 300, 350, 480], "status": "occupied", "iou": 0.32}
  ],
  "phone_alerts": []
}
```

---

## Integration with Spring Boot

The `outputs/<video_stem>_results.json` file (or each frame's dict) can be
forwarded to the backend via an HTTP POST.  The `app/api/` package is reserved
for a FastAPI wrapper that will expose a `/detect` endpoint.
# studyroom-ai
