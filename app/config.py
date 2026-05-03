"""
config.py — Centralized configuration for the AI Study Room module.

All paths are derived from the project root so the project stays portable.
Edit CLASS_NAMES and model/output settings here; avoid touching other files.
"""

from pathlib import Path

# ─── Project Root ──────────────────────────────────────────────────────────────
# Two levels up from this file: studyroom-ai/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ─── Data Directories ──────────────────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_VIDEO_DIR: Path = DATA_DIR / "raw_videos"
FRAMES_DIR: Path = DATA_DIR / "frames"
DATASETS_DIR: Path = DATA_DIR / "datasets"

# ─── Model Directories ─────────────────────────────────────────────────────────
MODELS_DIR: Path = PROJECT_ROOT / "models"
PRETRAINED_DIR: Path = MODELS_DIR / "pretrained"
TRAINED_DIR: Path = MODELS_DIR / "trained"

# ─── Output Directory ──────────────────────────────────────────────────────────
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# ─── Default Model Weights ─────────────────────────────────────────────────────
# Place a downloaded .pt file here, or let Ultralytics download it automatically.
DEFAULT_MODEL_WEIGHTS: Path = PRETRAINED_DIR / "yolov8n.pt"

# ─── Detection Classes ─────────────────────────────────────────────────────────
# Must match the class names used during training (or in the pretrained model).
CLASS_NAMES: list[str] = ["person", "seat", "phone"]

# ─── Detection Hyper-parameters ────────────────────────────────────────────────
DETECTION_CONFIDENCE: float = 0.4   # Minimum confidence to keep a detection
DETECTION_IOU: float = 0.45         # NMS IoU threshold

# ─── Proximity / Association Thresholds ────────────────────────────────────────
# IoU threshold: person bbox overlaps seat bbox → seat is occupied
SEAT_OCCUPANCY_IOU_THRESHOLD: float = 0.10

# Pixel distance: centroid of phone is within N px of person centroid → phone use
PHONE_USE_DISTANCE_THRESHOLD: float = 150.0

# ─── Phone Detection ───────────────────────────────────────────────────────────
# 允许的 phone → 最近 person 最大像素距离（放宽阈值确保链路跑通）
PHONE_PERSON_MAX_DISTANCE: float = 1000.0

# phone 检测框最低置信度（低于此值直接忽略）
PHONE_MIN_CONFIDENCE: float = 0.25

# ─── Video / Frame Settings ────────────────────────────────────────────────────
FRAME_EXTRACTION_INTERVAL: int = 30  # Extract every Nth frame (1 = every frame)

# ─── Training Settings ─────────────────────────────────────────────────────────
TRAIN_EPOCHS: int = 50
TRAIN_IMAGE_SIZE: int = 640
TRAIN_BATCH_SIZE: int = 16
TRAIN_DEVICE: str = "cpu"           # "0" for first GPU, "cpu" for CPU

# ─── Spring Boot Backend ───────────────────────────────────────────────────────
SPRING_BOOT_BASE_URL: str = "http://localhost:8088"
DEFAULT_ROOM_ID: int = 1          # 默认自习室 ID，可按需修改

# ─── 统一视频路径（与前端保持一致）──────────────────────────────────────────────
VIDEO_SOURCE = {
    1: "/Users/yang/Desktop/HBcode/StudyRoomFrontend/public/media/videos/2_people_at_a_table_for_4.mp4",
    2: "/Users/yang/Desktop/HBcode/StudyRoomFrontend/public/media/videos/4_people_at_studyroom.mp4",
    3: "/Users/yang/Desktop/HBcode/StudyRoomFrontend/public/media/videos/women-playing-phone.mp4"
}

# ─── Realtime Monitor ──────────────────────────────────────────────────────────
DETECT_EVERY_N_FRAMES: int = 5    # 每隔 N 帧执行一次 YOLO 检测
REUPLOAD_OCCUPIED_INTERVAL_SECONDS: int = 60 # 持续 occupied=True 时的重传周期

# 前端可访问的 latest.json 输出路径（Vite dev server 的 public 静态目录）
# Python 直接写入该路径，前端通过 fetch('/media/detections/latest.json') 读取
FRONTEND_LATEST_JSON: Path = Path(
    "/Users/yang/Desktop/HBcode/StudyRoomFrontend/public/media/detections/latest.json"
)
