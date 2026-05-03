"""
app/api/client.py — Spring Boot API 客户端

封装对 Spring Boot 后端的 HTTP 调用，目前实现：
    - upload_recognition(): POST /recognition/upload

所有请求使用 requests 库，超时 5 秒，失败时记录日志并继续（不中断主流程）。
"""

import logging
from datetime import datetime
from typing import Optional

import requests

from app import config

logger = logging.getLogger(__name__)

# 请求超时（秒）
_TIMEOUT = 5


def upload_recognition(
    seat_id: int,
    occupied: bool,
    confidence: float,
    detect_time: Optional[str] = None,
) -> bool:
    """向 Spring Boot 上报一条座位识别结果。

    Args:
        seat_id:     座位 ID，对应 recognition_record.seat_id。
        occupied:    是否有人占用。
        confidence:  平均检测置信度（0~1）。
        detect_time: ISO 8601 格式的时间字符串，例如 "2026-04-20T10:30:00"。
                     若为 None，则自动取当前本地时间。

    Returns:
        True 表示上报成功（HTTP 2xx），False 表示失败。
    """
    if detect_time is None:
        detect_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    payload = {
        "seatId": seat_id,
        "occupied": occupied,
        "confidence": round(confidence, 4),
        "detectTime": detect_time,
    }

    logger.info(
        "[Upload] seatId=%d  occupied=%s  confidence=%.4f  time=%s",
        seat_id, occupied, confidence, detect_time,
    )

    try:
        url = f"{config.SPRING_BOOT_BASE_URL}/recognition/upload"
        resp = requests.post(url, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()          # 4xx / 5xx → 抛出异常
        logger.info("[Upload] ✅ 成功: seatId=%d", seat_id)
        return True

    except requests.exceptions.ConnectionError:
        logger.error(
            "[Upload] ❌ 失败: seatId=%d, reason=无法连接 Spring Boot (%s)",
            seat_id, config.SPRING_BOOT_BASE_URL,
        )
    except requests.exceptions.Timeout:
        logger.error(
            "[Upload] ❌ 失败: seatId=%d, reason=请求超时 (%ds)", seat_id, _TIMEOUT
        )
    except requests.exceptions.HTTPError as e:
        logger.error(
            "[Upload] ❌ 失败: seatId=%d, reason=HTTP %s — %s",
            seat_id, e.response.status_code, e.response.text[:200],
        )
    except Exception as e:  # noqa: BLE001
        logger.error("[Upload] ❌ 失败: seatId=%d, reason=%s", seat_id, e)

    return False


def upload_recognition_batch(records: list[dict]) -> dict[int, bool]:
    """批量上报识别结果，每条记录调用一次 upload_recognition()。

    Args:
        records: 列表，每项包含 seat_id, occupied, confidence, detect_time。

    Returns:
        {seat_id: success_bool} 的结果字典。
    """
    results: dict[int, bool] = {}
    for rec in records:
        ok = upload_recognition(
            seat_id=rec["seat_id"],
            occupied=rec["occupied"],
            confidence=rec.get("confidence", 0.90),
            detect_time=rec.get("detect_time"),
        )
        results[rec["seat_id"]] = ok
    return results


def upload_behavior(
    seat_id: int,
    behavior_type: str,
    confidence: float,
    detect_time: Optional[str] = None,
) -> bool:
    """向 Spring Boot 上报一条行为检测记录。

    Args:
        seat_id:       座位 ID。
        behavior_type: 行为类型，目前固定为 ``"phone_detected"``。
        confidence:    检测置信度（取关联 phone 检测框的 conf）。
        detect_time:   ISO 8601 时间字符串；None 则自动取当前本地时间。

    Returns:
        True 表示上报成功（HTTP 2xx），False 表示失败。
    """
    if detect_time is None:
        detect_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    payload = {
        "seatId":       seat_id,
        "behaviorType": behavior_type,
        "confidence":   round(confidence, 4),
        "detectTime":   detect_time,
    }

    logger.info(
        "[Behavior] seatId=%d  behaviorType=%s  confidence=%.4f  time=%s",
        seat_id, behavior_type, confidence, detect_time,
    )

    try:
        url = f"{config.SPRING_BOOT_BASE_URL}/behavior/upload"
        resp = requests.post(url, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        logger.info("[Behavior] ✅ 成功: seatId=%d  behaviorType=%s", seat_id, behavior_type)
        return True

    except requests.exceptions.ConnectionError:
        logger.error(
            "[Behavior] ❌ 失败: seatId=%d, reason=无法连接 Spring Boot (%s)",
            seat_id, config.SPRING_BOOT_BASE_URL,
        )
    except requests.exceptions.Timeout:
        logger.error(
            "[Behavior] ❌ 失败: seatId=%d, reason=请求超时 (%ds)", seat_id, _TIMEOUT
        )
    except requests.exceptions.HTTPError as e:
        logger.error(
            "[Behavior] ❌ 失败: seatId=%d, reason=HTTP %s — %s",
            seat_id, e.response.status_code, e.response.text[:200],
        )
    except Exception as e:  # noqa: BLE001
        logger.error("[Behavior] ❌ 失败: seatId=%d, reason=%s", seat_id, e)

    return False
