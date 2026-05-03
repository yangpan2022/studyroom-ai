"""
scripts/summarize_detection_result.py — 检测结果摘要分析

输入:  outputs/detection_result.json
输出:
    1. 控制台统计摘要
    2. outputs/detection_summary.json

运行方式:
    python scripts/summarize_detection_result.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH  = ROOT / "outputs" / "detection_result.json"
OUTPUT_PATH = ROOT / "outputs" / "detection_summary.json"

# 每个 seat 最多保留几个样本帧
MAX_SAMPLES_PER_SEAT: int = 3


# ══════════════════════════════════════════════════════════════════════════════
# 核心分析
# ══════════════════════════════════════════════════════════════════════════════

def analyze(data: dict) -> dict:
    """从 detection_result.json 数据结构中提取统计摘要."""

    frame_results: list[dict] = data.get("frame_results", [])
    total_frames: int = len(frame_results)

    # ── 各计数器 ──────────────────────────────────────────────────────────────
    frames_with_detections: int = 0
    occupied_frames: int        = 0
    phone_frames: int           = 0

    seat_hit_count: dict[str, int]        = defaultdict(int)   # seatId → 命中帧数
    seat_samples:   dict[str, list[int]]  = defaultdict(list)  # seatId → 样本 frame_index
    detection_type_count: dict[str, int]  = defaultdict(int)   # class → 总检测数
    all_seat_ids:   set[str]              = set()              # 所有出现过的 seatId

    for fr in frame_results:
        frame_idx: int         = fr.get("frame_index", -1)
        detections: list[dict] = fr.get("detections", [])
        seat_status: list[dict]= fr.get("seat_status", [])

        # ── 检测框统计 ────────────────────────────────────────────────────────
        if detections:
            frames_with_detections += 1
            for det in detections:
                cls = det.get("type", det.get("class_name", "unknown"))
                detection_type_count[cls] += 1

        # ── seat 状态统计（按帧去重：每帧每个 seatId 最多计 1 次）────────────
        frame_has_occupied = False
        frame_has_phone    = False

        # 用 set 收集本帧内命中的 seatId，防止同一 seatId 重复出现时重复累计
        occupied_seats_this_frame: set[str] = set()

        for seat in seat_status:
            seat_id        = str(seat.get("seatId", "?"))
            occupied       = seat.get("occupied", False)
            phone_detected = seat.get("phone_detected", False)

            all_seat_ids.add(seat_id)  # 收集全量 seatId（不依赖第一帧）

            if occupied:
                occupied_seats_this_frame.add(seat_id)
                frame_has_occupied = True

            if phone_detected:
                frame_has_phone = True

        # 统一在帧结束后对去重后的命中 seat 累计计数
        for seat_id in occupied_seats_this_frame:
            seat_hit_count[seat_id] += 1
            if len(seat_samples[seat_id]) < MAX_SAMPLES_PER_SEAT:
                seat_samples[seat_id].append(frame_idx)

        if frame_has_occupied:
            occupied_frames += 1
        if frame_has_phone:
            phone_frames += 1

    # ── 未命中 seat 列表（all_seat_ids 在循环中逐帧收集，已含全量） ──────────
    unhit_seats: list[str] = sorted(
        sid for sid in all_seat_ids if seat_hit_count.get(sid, 0) == 0
    )

    # ── 组装结果 ──────────────────────────────────────────────────────────────
    summary = {
        "video": data.get("video", "unknown"),
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "occupied_frames": occupied_frames,
        "phone_frames": phone_frames,
        "seat_summary": dict(sorted(seat_hit_count.items())),
        "unhit_seats": unhit_seats,
        "detection_type_count": dict(sorted(detection_type_count.items())),
        "seat_samples": {k: v for k, v in sorted(seat_samples.items())},
    }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 控制台打印
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(summary: dict) -> None:
    """将摘要以可读格式打印到控制台."""

    SEP  = "=" * 52
    SEP2 = "-" * 52

    print(f"\n{SEP}")
    print(f"  检测结果摘要分析  ·  {summary['video']}")
    print(SEP)

    total = summary["total_frames"]
    det   = summary["frames_with_detections"]
    occ   = summary["occupied_frames"]
    ph    = summary["phone_frames"]

    print(f"  总帧数             : {total}")
    print(f"  有检测结果帧数     : {det}  ({_pct(det, total)})")
    print(f"  有座位占用帧数     : {occ}  ({_pct(occ, total)})")
    print(f"  有手机检测帧数     : {ph}  ({_pct(ph, total)})")
    print(SEP2)

    # 检测类型
    print("  检测框类型分布:")
    for cls, cnt in summary["detection_type_count"].items():
        print(f"    · {cls:<14} {cnt} 个")
    print(SEP2)

    # Seat 命中统计
    print("  Seat 命中统计 (occupied 帧数):")
    seat_summary = summary["seat_summary"]
    seat_samples = summary["seat_samples"]
    if seat_summary:
        for sid, cnt in seat_summary.items():
            samples = seat_samples.get(sid, [])
            sample_str = ", ".join(str(f) for f in samples)
            print(f"    seatId {sid}: {cnt} 帧  样本帧=[{sample_str}]")
    else:
        print("    （无任何座位被命中）")
    print(SEP2)

    # 未命中座位
    unhit = summary["unhit_seats"]
    if unhit:
        print(f"  ⚠️  从未被命中的 seatId: {unhit}")
    else:
        print("  ✅ 所有 seat 均被命中过")

    print(SEP)
    print(f"  摘要已写入: {OUTPUT_PATH}")
    print(SEP + "\n")


def _pct(part: int, total: int) -> str:
    """格式化百分比，避免除零."""
    if total == 0:
        return "N/A"
    return f"{part / total * 100:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] 找不到输入文件: {INPUT_PATH}", file=sys.stderr)
        print("请先运行:  python scripts/run_predict_video.py", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 读取: {INPUT_PATH}")
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = analyze(data)
    print_summary(summary)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
