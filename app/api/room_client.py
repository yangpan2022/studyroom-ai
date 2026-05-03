"""
app/api/room_client.py — 从 Spring Boot 获取房间信息及视频源

公开接口：
    get_room_video_source(room_id) → {"videoUrl": str, "videoType": str}
    resolve_local_video_path(video_url)  → Path（本地视频绝对路径）
"""

import logging
from pathlib import Path

import requests

from app import config

logger = logging.getLogger(__name__)

_TIMEOUT = 5  # 单次请求超时（秒）

# 当前只支持的视频类型
_SUPPORTED_VIDEO_TYPES = {"video"}


# ══════════════════════════════════════════════════════════════════════════════
# 私有工具
# ══════════════════════════════════════════════════════════════════════════════

def _get(path: str) -> dict | None:
    """向 Spring Boot 发 GET 请求，返回 data 字段；失败返回 None。"""
    url = f"{config.SPRING_BOOT_BASE_URL}{path}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") == 200:
            return body.get("data")
        logger.warning("[RoomClient] 业务错误 %s → code=%s msg=%s",
                       path, body.get("code"), body.get("message"))
    except requests.exceptions.ConnectionError:
        logger.error("[RoomClient] 无法连接 Spring Boot: %s", url)
    except requests.exceptions.Timeout:
        logger.error("[RoomClient] 请求超时 (%ds): %s", _TIMEOUT, url)
    except requests.exceptions.HTTPError as e:
        logger.error("[RoomClient] HTTP %s: %s", e.response.status_code, url)
    except Exception as e:  # noqa: BLE001
        logger.error("[RoomClient] 未知错误 %s: %s", url, e)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 公开 API
# ══════════════════════════════════════════════════════════════════════════════

def get_room_video_source(room_id: int) -> dict:
    """从 Spring Boot 获取指定房间的视频源信息。

    Args:
        room_id: 自习室 ID。

    Returns:
        {"videoUrl": str, "videoType": str, "roomName": str}

    Raises:
        RuntimeError: Spring Boot 不可达、房间不存在、videoUrl 为空、
                      或 videoType 为不支持的类型。
    """
    data = _get(f"/rooms/{room_id}")
    if data is None:
        raise RuntimeError(
            f"[RoomClient] 无法获取 roomId={room_id} 的信息，"
            f"请确认 Spring Boot 已启动 ({config.SPRING_BOOT_BASE_URL})"
        )

    video_url: str = data.get("videoUrl") or ""
    video_type: str = data.get("videoType") or ""
    room_name: str = data.get("roomName") or f"Room-{room_id}"

    if not video_url:
        raise RuntimeError(
            f"[RoomClient] roomId={room_id} 的 videoUrl 为空，"
            "请在管理后台为该自习室配置视频源"
        )

    if video_type not in _SUPPORTED_VIDEO_TYPES:
        raise RuntimeError(
            f"[RoomClient] roomId={room_id} 的 videoType='{video_type}' "
            f"暂不支持（当前仅支持: {_SUPPORTED_VIDEO_TYPES}）"
        )

    logger.info("[RoomClient] roomId=%d  roomName=%s", room_id, room_name)
    logger.info("[RoomClient] videoType=%s  videoUrl=%s", video_type, video_url)

    return {
        "roomId":    room_id,
        "roomName":  room_name,
        "videoUrl":  video_url,
        "videoType": video_type,
    }


def resolve_local_video_path(video_url: str) -> Path:
    """将数据库中存储的 videoUrl 映射为 Python 可访问的本地文件路径。

    映射规则（仅依赖文件名，忽略路径前缀）：
        数据库值:   "src/assets/vidoes/4_people_at_studyroom.mp4"
                 或 "/media/videos/4_people_at_studyroom.mp4"
        本地路径:   <project>/data/raw_videos/4_people_at_studyroom.mp4

    Args:
        video_url: Spring Boot 返回的 videoUrl 字段值。

    Returns:
        已验证存在的本地视频绝对路径。

    Raises:
        FileNotFoundError: 本地 data/raw_videos/ 中找不到对应文件。
    """
    filename = Path(video_url).name          # 只取文件名，丢弃前缀路径
    local_path = config.RAW_VIDEO_DIR / filename

    logger.info("[RoomClient] videoUrl → localVideoPath=%s", local_path)

    if not local_path.exists():
        raise FileNotFoundError(
            f"[RoomClient] 视频文件不存在: {local_path}\n"
            f"  videoUrl={video_url!r} → 文件名={filename!r}\n"
            f"  请将视频文件放入: {config.RAW_VIDEO_DIR}/"
        )

    return local_path
