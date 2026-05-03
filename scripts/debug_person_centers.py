"""
scripts/debug_person_centers.py — Person 检测框中心点分布调试

读取已有的 outputs/detection_result.json，打印前 N 帧中每个 person 的
bbox 和中心点坐标，方便重新划分 seat_region。

运行方式:
    python scripts/debug_person_centers.py
"""

import json
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
INPUT_PATH  = ROOT / "outputs" / "detection_result.json"
PRINT_FRAMES: int = 20   # 只打印前 N 帧


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[ERROR] 找不到: {INPUT_PATH}")
        print("请先运行: python scripts/run_predict_video.py")
        return

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frame_results = data.get("frame_results", [])
    total = len(frame_results)

    print(f"\n{'='*54}")
    print(f"  Person 中心点分布  ·  前 {PRINT_FRAMES} 帧（共 {total} 帧）")
    print(f"{'='*54}")

    # 收集所有中心点，用于打印整体分布范围
    all_cx, all_cy = [], []

    for fr in frame_results[:PRINT_FRAMES]:
        frame_idx  = fr.get("frame_index", -1)
        detections = fr.get("detections", [])
        persons    = [d for d in detections if d.get("type") == "person"]

        if not persons:
            print(f"\nFrame {frame_idx:>4}:  (无 person 检测)")
            continue

        print(f"\nFrame {frame_idx:>4}:")
        for d in persons:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = d.get("confidence", 0)
            all_cx.append(cx)
            all_cy.append(cy)
            print(f"  person  bbox=({x1:>4}, {y1:>4}, {x2:>4}, {y2:>4})"
                  f"  center=({cx:>4}, {cy:>4})  conf={conf:.2f}")

    # ── 整体范围统计（帮助划定 seat 区域） ────────────────────────────────────
    if all_cx:
        print(f"\n{'─'*54}")
        print(f"  前 {PRINT_FRAMES} 帧 person 中心点范围汇总:")
        print(f"    centerX : min={min(all_cx):>4}  max={max(all_cx):>4}"
              f"  avg={sum(all_cx)//len(all_cx):>4}")
        print(f"    centerY : min={min(all_cy):>4}  max={max(all_cy):>4}"
              f"  avg={sum(all_cy)//len(all_cy):>4}")

    print(f"\n{'='*54}")
    print("  💡 根据以上中心点，建议按象限划分 4 个 seat_region:")
    print("     seat 1 (左下): x1~x2 覆盖左侧 centerX, y1~y2 覆盖下方 centerY")
    print("     seat 2 (左上): x1~x2 覆盖左侧 centerX, y1~y2 覆盖上方 centerY")
    print("     seat 3 (右上): x1~x2 覆盖右侧 centerX, y1~y2 覆盖上方 centerY")
    print("     seat 4 (右下): x1~x2 覆盖右侧 centerX, y1~y2 覆盖下方 centerY")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()
