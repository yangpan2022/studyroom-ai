"""
scripts/run_predict_video.py — 按 roomId 运行的视频检测闭环

流程:
    1. 从 Spring Boot 获取 room.videoUrl / room.videoType
    2. 将 videoUrl 映射为本地视频文件路径
    3. 打开视频，读取分辨率
    4. 从 Spring Boot 获取该 room 的 seat_regions（像素坐标）
    5. YOLO 帧采样检测（person + cell phone）
    6. postprocess：判断 seat 占用状态
    7. 写出 outputs/detection_result.json（含 roomId / videoUrl / 分辨率）
    8. 汇总上报 Spring Boot（POST /recognition/upload）

运行方式:
    # 使用 config.DEFAULT_ROOM_ID
    python scripts/run_predict_video.py

    # 指定 roomId
    python scripts/run_predict_video.py --room-id 2
"""

import argparse
import json
import sys
import logging
from pathlib import Path

# ── 确保从项目根目录运行也能 import app 包 ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from app import config
from app.api.client import upload_recognition_batch
from app.api.room_client import get_room_video_source, resolve_local_video_path
from app.api.seat_client import get_seat_regions
from app.detection.postprocess import check_seat_status_from_regions
from app.utils.paths import ensure_dirs

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 运行参数（可通过命令行覆盖）
# ══════════════════════════════════════════════════════════════════════════════

# 每 SAMPLE_EVERY 帧采样一次（降低处理压力）
SAMPLE_EVERY: int = 5

# 最多处理的帧数（避免超长视频耗时过长）
MAX_FRAMES: int = 200

# YOLO 目标类别白名单（coco 模型名称）
TARGET_CLASSES: set[str] = {"person", "cell phone"}

# 模型权重
MODEL_WEIGHTS = config.DEFAULT_MODEL_WEIGHTS


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="studyroom-ai 视频检测")
    parser.add_argument(
        "--room-id", type=int, default=config.DEFAULT_ROOM_ID,
        help=f"指定自习室 ID（默认: {config.DEFAULT_ROOM_ID}）",
    )
    return parser.parse_args()


def parse_detections(result) -> list[dict]:
    """将 Ultralytics Results 转换为可序列化的检测列表（仅保留目标类别）."""
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names.get(class_id, str(class_id))
        if class_name not in TARGET_CLASSES:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_name": class_name,
            "type":       class_name,
            "x1": round(x1), "y1": round(y1),
            "x2": round(x2), "y2": round(y2),
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "confidence": round(confidence, 4),
        })
    return detections


def detection_to_output(det: dict) -> dict:
    """将内部 detection dict 转换为输出 JSON 格式."""
    return {
        "type": det["type"],
        "x1": det["x1"], "y1": det["y1"],
        "x2": det["x2"], "y2": det["y2"],
        "confidence": det["confidence"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    room_id: int = args.room_id

    # ── 1. 初始化目录 ────────────────────────────────────────────────────────
    ensure_dirs()

    # ── 2. 从 Spring Boot 获取 room 的视频源信息 ──────────────────────────────
    room_info = get_room_video_source(room_id)
    video_url: str = room_info["videoUrl"]

    # ── 3. 将 videoUrl 映射为本地文件路径 ────────────────────────────────────
    video_path = resolve_local_video_path(video_url)

    # ── 4. 加载 YOLO 模型 ─────────────────────────────────────────────────────
    logger.info("加载 YOLO 模型: %s", MODEL_WEIGHTS)
    model = YOLO(str(MODEL_WEIGHTS))
    logger.info("模型加载完成")

    # ── 5. 打开视频，读取分辨率 ───────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("视频信息: %d 帧, %.1f FPS, %dx%d → %s",
                total_video_frames, fps, frame_width, frame_height, video_path.name)

    # ── 6. 从 Spring Boot 获取 seat_regions ──────────────────────────────────
    seat_regions = get_seat_regions(
        room_id=room_id,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if not seat_regions:
        logger.error("未能获取任何 seat region，终止执行。请确认：")
        logger.error("  1. Spring Boot 已启动 (%s)", config.SPRING_BOOT_BASE_URL)
        logger.error("  2. roomId=%d 下有 cameraEnabled=1 且已完成 region 标定的座位", room_id)
        cap.release()
        return

    # ── 7. 逐帧检测 ──────────────────────────────────────────────────────────
    frame_results: list[dict] = []
    frame_index: int = 0
    processed_count: int = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % SAMPLE_EVERY == 0:
            results = model.predict(
                source=frame,
                conf=config.DETECTION_CONFIDENCE,
                iou=config.DETECTION_IOU,
                verbose=False,
            )
            detections = parse_detections(results[0])
            seat_status = check_seat_status_from_regions(detections, seat_regions)

            det_count = len(detections)
            print(f"Frame {frame_index}: {det_count} objects detected "
                  f"({', '.join(d['type'] for d in detections) or 'none'})")

            if det_count > 0:
                frame_results.append({
                    "frame_index":  frame_index,
                    "detections":   [detection_to_output(d) for d in detections],
                    "seat_status":  seat_status,
                })

            processed_count += 1
            if processed_count >= MAX_FRAMES:
                logger.info("已达到最大处理帧数 %d，提前退出", MAX_FRAMES)
                break

        frame_index += 1

    cap.release()
    logger.info("处理完成: 共读取 %d 帧，实际推理 %d 帧，有结果帧 %d 帧",
                frame_index, processed_count, len(frame_results))

    # ── 8. 写出 JSON（含 roomId / videoUrl / 分辨率上下文）──────────────────
    output_data = {
        "roomId":      room_id,
        "videoUrl":    video_url,
        "frameWidth":  frame_width,
        "frameHeight": frame_height,
        "video":       video_path.name,
        "frame_results": frame_results,
    }

    output_path = config.OUTPUTS_DIR / "detection_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("✅ 检测结果已写入: %s  (Detection output belongs to roomId=%d)",
                output_path, room_id)
    print(f"\n输出文件: {output_path}")
    print(f"roomId: {room_id}  videoUrl: {video_url}")
    print(f"总帧数（有结果）: {len(frame_results)}")

    # ── 9. 汇总上报 Spring Boot ───────────────────────────────────────────────
    _report_to_spring_boot(frame_results)


def _report_to_spring_boot(frame_results: list[dict]) -> None:
    """对视频检测结果做 seat 级别汇总，去重后上报 Spring Boot。

    去重策略：
        - 遍历所有帧的 seat_status
        - 用 dict 以 seatId 为键，occupied=True 优先（帧级 OR 合并）
        - 累积 person confidence 均值作为上报置信度
        - 最终每个 seatId 只调用一次 upload_recognition()
    """
    from datetime import datetime

    if not frame_results:
        logger.info("无检测帧，跳过上报")
        return

    detect_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # seat_id → {"occupied": bool, "conf_sum": float, "conf_cnt": int}
    seat_agg: dict[int, dict] = {}

    for fr in frame_results:
        person_confs = [
            d["confidence"] for d in fr.get("detections", [])
            if d.get("type") == "person"
        ]
        frame_conf = sum(person_confs) / len(person_confs) if person_confs else 0.0

        for seat in fr.get("seat_status", []):
            sid: int = seat["seatId"]
            occupied: bool = seat.get("occupied", False)

            if sid not in seat_agg:
                seat_agg[sid] = {"occupied": False, "conf_sum": 0.0, "conf_cnt": 0}

            if occupied:
                seat_agg[sid]["occupied"] = True
                seat_agg[sid]["conf_sum"] += frame_conf
                seat_agg[sid]["conf_cnt"] += 1

    if not seat_agg:
        logger.info("seat_agg 为空，跳过上报")
        return

    records = []
    for sid, agg in sorted(seat_agg.items()):
        cnt = agg["conf_cnt"]
        avg_conf = agg["conf_sum"] / cnt if cnt > 0 else 0.90
        records.append({
            "seat_id":    sid,
            "occupied":   agg["occupied"],
            "confidence": avg_conf,
            "detect_time": detect_time,
        })

    logger.info("\n开始上报 Spring Boot，共 %d 条 seat 记录...", len(records))
    results = upload_recognition_batch(records)
    success = sum(1 for ok in results.values() if ok)
    logger.info("上报完成: %d/%d 成功", success, len(results))


if __name__ == "__main__":
    main()
