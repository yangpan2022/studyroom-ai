"""
scripts/run_realtime_monitor.py — 准实时常驻检测脚本

模式：单房间、视频循环播放、每 N 帧检测一次、持续覆盖写 latest.json。

运行方式:
    python scripts/run_realtime_monitor.py            # 使用 config.DEFAULT_ROOM_ID
    python scripts/run_realtime_monitor.py --room-id 2

退出：Ctrl+C
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 项目根目录注入 sys.path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from app import config
from app.api.client import upload_recognition, upload_behavior
from app.api.room_client import get_room_video_source, resolve_local_video_path
from app.api.seat_client import get_seat_regions
from app.detection.postprocess import check_seat_status_from_regions
from app.utils.paths import ensure_dirs

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── 输出路径（多房间独立输出）────────────────────────────────────
# 不再使用全局写死的 latest.json，在写出时根据 roomId 动态生成

# ── YOLO 目标类别 ─────────────────────────────────────────────────────────────
TARGET_CLASSES: set[str] = {"person", "cell phone"}


# ══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="studyroom-ai 准实时检测监控")
    p.add_argument(
        "--room-id", type=int, default=config.DEFAULT_ROOM_ID,
        help=f"自习室 ID（默认: {config.DEFAULT_ROOM_ID}）",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 辅助：YOLO 检测结果解析
# ══════════════════════════════════════════════════════════════════════════════

def parse_detections(result, frame_width: int, frame_height: int) -> list[dict]:
    """解析 YOLO 结果，返回同时包含像素坐标和相对坐标（0~1）的 detection 列表。

    内部 postprocess 使用像素坐标（bbox 字段），
    latest.json 输出使用相对坐标（x1/y1/x2/y2 字段，减少传输依赖分辨率）。
    """
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id   = int(box.cls[0])
        class_name = result.names.get(class_id, str(class_id))
        if class_name not in TARGET_CLASSES:
            continue
        conf = float(box.conf[0])
        px1, py1, px2, py2 = [round(v) for v in box.xyxy[0].tolist()]
        detections.append({
            # 像素坐标（供 postprocess 使用）
            "class_name": class_name,
            "type":       class_name,
            "bbox":       [px1, py1, px2, py2],
            # 相对坐标（写入 latest.json）
            "x1": round(px1 / frame_width,  4),
            "y1": round(py1 / frame_height, 4),
            "x2": round(px2 / frame_width,  4),
            "y2": round(py2 / frame_height, 4),
            "confidence": round(conf, 4),
        })
    return detections


# ══════════════════════════════════════════════════════════════════════════════
# 辅助：原子写 latest.json
# ══════════════════════════════════════════════════════════════════════════════

def write_latest(payload: dict) -> None:
    """原子覆盖写 latest_{roomId}.json，同时写出到两个目标路径：

    1. outputs/latest_{roomId}.json         — 本地存档，供调试使用
    2. public/media/detections/latest_{roomId}.json — 前端 fetch 读取

    写入前强制断言 roomId 存在，避免前端收到无 roomId 的结果。
    使用「先写 .tmp 再 rename」的原子操作，防止前端读到写到一半的文件。
    """
    # 安全保护：roomId 缺失时拒绝写出，打印明确错误
    room_id = payload.get("roomId")
    if not room_id:
        logger.error("[write_latest] 拒绝写出：payload 缺少 roomId！payload keys=%s",
                     list(payload.keys()))
        return

    content = json.dumps(payload, ensure_ascii=False, indent=2)

    # 动态生成文件路径
    local_path = config.OUTPUTS_DIR / f"latest_{room_id}.json"
    frontend_path = config.FRONTEND_LATEST_JSON.with_name(f"latest_{room_id}.json")

    # ── 写目标 1：outputs/latest_{roomId}.json ────────────────────────────────────────
    tmp1 = local_path.with_suffix(".tmp")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp1.write_text(content, encoding="utf-8")
    tmp1.replace(local_path)

    # ── 写目标 2：前端 public/media/detections/latest_{roomId}.json ───────────────────
    try:
        frontend_path.parent.mkdir(parents=True, exist_ok=True)
        tmp2 = frontend_path.with_suffix(".tmp")
        tmp2.write_text(content, encoding="utf-8")
        tmp2.replace(frontend_path)
    except OSError as e:
        logger.warning("[write_latest] 无法写入前端路径 %s: %s", frontend_path, e)


# ══════════════════════════════════════════════════════════════════════════════
# 辅助：recognition 去重上报
# ══════════════════════════════════════════════════════════════════════════════

def upload_changed_seats(
    seat_status: list[dict],
    last_state: dict[int, bool],
    last_upload_time: dict[int, float],
) -> tuple[dict[int, bool], dict[int, float]]:
    """首次出现 occupied 或状态变化时立即上报，持续 occupied=True 则定期重传。

    Args:
        seat_status: 本次 postprocess 返回的 seat 状态列表。
        last_state:  上次上报时的 {seatId: occupied} 字典（内存中维护）。
        last_upload_time: 最后一次成功上报的时间戳 {seatId: timestamp}。

    Returns:
        (更新后的 last_state, 更新后的 last_upload_time)
    """
    detect_time_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    now_ts = time.time()
    
    new_state = dict(last_state)
    new_upload_time = dict(last_upload_time)

    for seat in seat_status:
        sid: int       = seat["seatId"]
        occupied: bool = seat.get("occupied", False)
        prev: bool | None = last_state.get(sid, None)   # None = 首次见到该 seat

        should_upload = False
        logger_msg = ""

        # 状态变化（含首次）
        if prev != occupied:
            should_upload = True
            logger_msg = f"[Upload] recognition changed: seatId={sid} {prev} → {occupied}"
        # 状态未变，且为 occupied=true，检查是否达到定期重传间隔
        elif occupied is True:
            last_time = last_upload_time.get(sid, 0.0)
            if now_ts - last_time >= config.REUPLOAD_OCCUPIED_INTERVAL_SECONDS:
                should_upload = True
                logger_msg = f"[Upload] recognition periodic reupload: seatId={sid} occupied=True"

        if should_upload:
            logger.info(logger_msg)
            ok = upload_recognition(
                seat_id=sid,
                occupied=occupied,
                confidence=0.85,        # 实时模式暂用固定置信度
                detect_time=detect_time_str,
            )
            if ok:
                new_state[sid] = occupied
                new_upload_time[sid] = now_ts

    return new_state, new_upload_time


# ══════════════════════════════════════════════════════════════════════════════
# 辅助：phone 去重上报
# ══════════════════════════════════════════════════════════════════════════════

def upload_changed_phone_states(
    seat_status: list[dict],
    detections:  list[dict],
    last_phone_state: dict[int, bool],
) -> dict[int, bool]:
    """仅对 phone_detected 状态发生变化的 seat 上报 /behavior/upload。

    策略：
        False → True  ：上报一次
        True  → True  ：不重复上报
        True  → False ：当前不上报“手机结束使用”（可扩展）

    confidence 取当前帧中所有 phone/cell phone 检测框的最高置信度。

    Args:
        seat_status:      本次 postprocess 返回的 seat 状态列表。
        detections:       本次帧所有 detection（用于取 phone confidence）。
        last_phone_state: 上次帧的 {seatId: phone_detected} 字典。

    Returns:
        更新后的 last_phone_state。
    """
    detect_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    new_state   = dict(last_phone_state)

    # 取本帧最高 phone 置信度（用于上报）
    phone_confs = [
        d.get("confidence", 0.5)
        for d in detections
        if d.get("type", d.get("class_name", "")) in ("cell phone", "phone")
    ]
    best_phone_conf = max(phone_confs) if phone_confs else 0.5

    for seat in seat_status:
        sid: int             = seat["seatId"]
        phone_now: bool      = seat.get("phone_detected", False)
        phone_prev: bool | None = last_phone_state.get(sid, None)   # None = 首次

        if phone_prev != phone_now and phone_now:   # False/None → True：上报
            logger.info(
                "[Behavior] Upload behavior changed: seatId=%d  phone_detected=%s → %s",
                sid, phone_prev, phone_now,
            )
            ok = upload_behavior(
                seat_id=sid,
                behavior_type="phone_detected",
                confidence=best_phone_conf,
                detect_time=detect_time,
            )
            if ok:
                new_state[sid] = phone_now
        elif phone_prev != phone_now and not phone_now:
            # True → False：更新内存状态但不上报
            new_state[sid] = phone_now

    return new_state



# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = parse_args()
    room_id = args.room_id

    # ── 1. 初始化目录 ────────────────────────────────────────────────────────
    ensure_dirs()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 2. 从 config 中获取前后端统一的视频源 ──────────────────────────────
    video_path_str = config.VIDEO_SOURCE.get(room_id)
    if not video_path_str:
        raise ValueError(f"未配置 roomId={room_id} 的视频路径")

    logger.info("Room ID: %d", room_id)
    logger.info("Video Path: %s", video_path_str)

    if not os.path.exists(video_path_str):
        raise FileNotFoundError(f"视频文件不存在: {video_path_str}")

    video_path = Path(video_path_str)
    video_url = f"/media/videos/{video_path.name}" # 保持传递给前端的形式一致

    # ── 3. 加载 YOLO 模型 ─────────────────────────────────────────────────────
    logger.info("加载 YOLO 模型: %s", config.DEFAULT_MODEL_WEIGHTS)
    model = YOLO(str(config.DEFAULT_MODEL_WEIGHTS))
    logger.info("模型加载完成")

    # ── 4. 打开视频，获取分辨率 ───────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    logger.info("视频: %dx%d  %.1f FPS  共 %d 帧  → %s",
                frame_width, frame_height, fps, total_frames, video_path.name)

    # ── 5. 从 Spring Boot 获取 seat_regions ──────────────────────────────────
    seat_regions = get_seat_regions(
        room_id=room_id,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if not seat_regions:
        logger.error("未能获取任何 seat region，退出。请完成座位标定后重试。")
        cap.release()
        return

    logger.info("准实时监控启动 — roomId=%d  Ctrl+C 退出", room_id)
    local_out_path = config.OUTPUTS_DIR / f"latest_{room_id}.json"
    frontend_out_path = config.FRONTEND_LATEST_JSON.with_name(f"latest_{room_id}.json")
    
    logger.info("videoUrl      : %s", video_url)
    logger.info("outputs       : %s", local_out_path)
    logger.info("frontend path : %s", frontend_out_path)
    logger.info("frontend readable: %s", frontend_out_path.parent.exists())

    # ── 6. 准实时检测主循环 ───────────────────────────────────────────────────
    detect_every   = config.DETECT_EVERY_N_FRAMES
    frame_index    = 0        # 当前帧序号（循环后不重置，单调递增供日志区分）
    read_index     = 0        # 视频内帧序号（到末尾后重置为 0）
    last_state: dict[int, bool]      = {}   # {seatId: occupied}
    last_upload_time: dict[int, float] = {} # {seatId: timestamp}
    last_phone_state: dict[int, bool] = {}   # {seatId: phone_detected}

    try:
        while True:
            ret, frame = cap.read()

            # 到达视频末尾 → 跳回第一帧（循环播放）
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_index = 0
                ret, frame = cap.read()
                if not ret:
                    logger.error("视频读取失败，退出")
                    break

            # 每 N 帧执行一次 YOLO 检测
            if read_index % detect_every == 0:
                # YOLO 推理
                results = model.predict(
                    source=frame,
                    conf=config.DETECTION_CONFIDENCE,
                    iou=config.DETECTION_IOU,
                    verbose=False,
                )
                detections = parse_detections(results[0], frame_width, frame_height)

                # postprocess：判断 seat 状态（使用像素坐标）
                seat_status = check_seat_status_from_regions(detections, seat_regions)

                # 日志
                person_cnt = sum(1 for d in detections if d.get("type", d.get("class_name")) == "person")
                phone_cnt  = sum(1 for d in detections if d.get("type", d.get("class_name", "")) in ("cell phone", "phone"))
                logger.info("Frame %d: %d person(s), %d phone(s) detected", frame_index, person_cnt, phone_cnt)

                # phone 关联日志
                if phone_cnt > 0:
                    for ss in seat_status:
                        if ss.get("phone_detected"):
                            logger.info("[Phone] 已关联到 seatId=%d", ss["seatId"])
                    if not any(ss.get("phone_detected") for ss in seat_status):
                        logger.info("[Phone] Phone detected but no nearby person matched")

                # 写 latest_{roomId}.json（相对坐标，去掉内部 bbox / class_name 字段）
                output_detections = [
                    {
                        "type":       d["type"],
                        "x1":         d["x1"],
                        "y1":         d["y1"],
                        "x2":         d["x2"],
                        "y2":         d["y2"],
                        "confidence": d["confidence"],
                    }
                    for d in detections
                ]
                payload = {
                    "roomId":      room_id,
                    "videoUrl":    video_url,
                    "frameIndex":  frame_index,
                    "frameWidth":  frame_width,
                    "frameHeight": frame_height,
                    "updatedAt":   datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "detections":  output_detections,
                    "seatStatus":  seat_status,
                }
                logger.info("Writing latest_%d.json: videoUrl=%s  frame=%d",
                            room_id, video_url, frame_index)
                write_latest(payload)

                # 去重/定期上报：recognition (occupied)
                last_state, last_upload_time = upload_changed_seats(
                    seat_status, last_state, last_upload_time
                )
                # 去重上报：behavior (phone_detected)
                last_phone_state = upload_changed_phone_states(
                    seat_status, detections, last_phone_state
                )

            frame_index += 1
            read_index  += 1

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正常退出")
    finally:
        cap.release()
        logger.info("视频资源已释放，监控结束")


if __name__ == "__main__":
    main()
