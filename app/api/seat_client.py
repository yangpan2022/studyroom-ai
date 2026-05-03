"""
app/api/seat_client.py — 从 Spring Boot 获取座位区域数据

流程：
    1. GET /rooms/{roomId}/seats          → 拿到 cameraEnabled=1 的座位列表
    2. GET /seats/{seatId}/region (并发)  → 拿每个座位的相对坐标 (0~1)
    3. 过滤掉 region=null 的座位（未标定）
    4. 将相对坐标转换为像素坐标并返回

返回格式与 postprocess.check_seat_status_from_regions() 所需格式兼容：
    [{"seatId": 1, "x1": 350, "y1": 450, "x2": 700, "y2": 1000}, ...]
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from app import config

logger = logging.getLogger(__name__)

_TIMEOUT = 5  # 单次请求超时（秒）


# ══════════════════════════════════════════════════════════════════════════════
# 私有：基础 HTTP 工具
# ══════════════════════════════════════════════════════════════════════════════

def _get(path: str) -> Optional[dict]:
    """向 Spring Boot 发 GET 请求，返回 data 字段内容；失败返回 None。"""
    url = f"{config.SPRING_BOOT_BASE_URL}{path}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") == 200:
            return body.get("data")
        logger.warning("[SeatClient] 业务错误 %s → code=%s msg=%s",
                       path, body.get("code"), body.get("message"))
    except requests.exceptions.ConnectionError:
        logger.error("[SeatClient] 无法连接 Spring Boot: %s", url)
    except requests.exceptions.Timeout:
        logger.error("[SeatClient] 请求超时 (%ds): %s", _TIMEOUT, url)
    except requests.exceptions.HTTPError as e:
        logger.error("[SeatClient] HTTP %s: %s", e.response.status_code, url)
    except Exception as e:  # noqa: BLE001
        logger.error("[SeatClient] 未知错误 %s: %s", url, e)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 私有：拉取单个 seat 的 region（用于并发拉取）
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_region(seat_id: int) -> Optional[dict]:
    """返回该 seat 的 region dict（原始相对坐标），或 None（未标定/失败）。"""
    return _get(f"/seats/{seat_id}/region")


# ══════════════════════════════════════════════════════════════════════════════
# 公开 API
# ══════════════════════════════════════════════════════════════════════════════

def get_seat_regions(
    room_id: int,
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    """从 Spring Boot 获取指定自习室的所有座位区域，转换为像素坐标列表。

    流程：
        1. GET /rooms/{room_id}/seats → 筛选 cameraEnabled=1 的座位
        2. 并发 GET /seats/{id}/region → 跳过 region=null 的座位
        3. 将相对坐标 (0~1) × 分辨率 → 像素坐标

    Args:
        room_id:      自习室 ID（对应 config.DEFAULT_ROOM_ID）。
        frame_width:  视频帧宽度（像素），用于相对坐标转换。
        frame_height: 视频帧高度（像素），用于相对坐标转换。

    Returns:
        与 postprocess.check_seat_status_from_regions() 所需格式兼容的列表：
        [{"seatId": 1, "x1": 350, "y1": 120, "x2": 700, "y2": 480}, ...]
        若无法获取数据则返回空列表。
    """
    # ── Step 1: 拿座位列表 ────────────────────────────────────────────────────
    seats_data = _get(f"/rooms/{room_id}/seats")
    if not seats_data:
        logger.error("[SeatClient] 无法获取 roomId=%d 的座位列表", room_id)
        return []

    # 只处理摄像头已启用的座位
    camera_seats = [s for s in seats_data if s.get("cameraEnabled") == 1]
    logger.info("[SeatClient] roomId=%d 共 %d 个摄像头座位，开始拉取 region...",
                room_id, len(camera_seats))

    # ── Step 2: 并发拉取 region ───────────────────────────────────────────────────
    seat_regions: list[dict] = []

    if not camera_seats:
        logger.warning("[SeatClient] roomId=%d 没有启用摄像头的座位（cameraEnabled=1）", room_id)
        return []

    with ThreadPoolExecutor(max_workers=min(len(camera_seats), 8)) as pool:
        future_map = {
            pool.submit(_fetch_region, s["seatId"]): s["seatId"]
            for s in camera_seats
        }
        for future in as_completed(future_map):
            sid = future_map[future]
            region = future.result()

            if region is None:
                logger.warning("[SeatClient] seatId=%d region=null，跳过（未标定）", sid)
                continue

            # ── Step 3: 相对坐标 → 像素坐标 ──────────────────────────────────
            x1_px = round(float(region["x1"]) * frame_width)
            y1_px = round(float(region["y1"]) * frame_height)
            x2_px = round(float(region["x2"]) * frame_width)
            y2_px = round(float(region["y2"]) * frame_height)

            seat_regions.append({
                "seatId": sid,
                "x1": x1_px,
                "y1": y1_px,
                "x2": x2_px,
                "y2": y2_px,
            })
            logger.debug("[SeatClient] seatId=%d → px(%d,%d,%d,%d)",
                         sid, x1_px, y1_px, x2_px, y2_px)

    # 按 seatId 排序，保证顺序稳定
    seat_regions.sort(key=lambda r: r["seatId"])

    logger.info("[SeatClient] ✅ Loaded seat regions from backend: %d seats",
                len(seat_regions))
    return seat_regions
