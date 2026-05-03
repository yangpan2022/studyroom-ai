"""
detection/postprocess.py — Post-processing logic for detection results.

Converts raw bounding-box detections into higher-level seat occupancy states
and phone-usage alerts used by the rest of the application and the backend API.
"""

from typing import Any, Optional

from app import config


# ─── Type alias ────────────────────────────────────────────────────────────────
BBox = list[int]          # [x1, y1, x2, y2]
Detection = dict[str, Any]


# ─── Low-level geometry helpers ────────────────────────────────────────────────

def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Compute the Intersection-over-Union (IoU) of two bounding boxes.

    Args:
        box_a: ``[x1, y1, x2, y2]`` of the first box.
        box_b: ``[x1, y1, x2, y2]`` of the second box.

    Returns:
        IoU value in ``[0.0, 1.0]``.
    """
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def centroid(bbox: BBox) -> tuple[int, int]:
    """Return the (cx, cy) centroid of a bounding box.

    Args:
        bbox: ``[x1, y1, x2, y2]``.

    Returns:
        Integer ``(cx, cy)`` tuple.
    """
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def euclidean_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Euclidean pixel distance between two 2-D points.

    Args:
        p1: First point ``(x, y)``.
        p2: Second point ``(x, y)``.

    Returns:
        Distance in pixels.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# ─── Class filtering ───────────────────────────────────────────────────────────

def extract_boxes_by_class(
    detections: list[Detection], class_name: str
) -> list[Detection]:
    """Filter detections to only those matching *class_name*.

    Args:
        detections: All detections from a single frame.
        class_name: Target class label (e.g. ``"person"``).

    Returns:
        Subset of *detections* whose ``class_name`` field matches.
    """
    return [d for d in detections if d["class_name"] == class_name]


# ─── Seat Occupancy ────────────────────────────────────────────────────────────

def determine_seat_occupancy(
    persons: list[Detection],
    seats: list[Detection],
    iou_threshold: float = config.SEAT_OCCUPANCY_IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """Decide whether each detected seat is occupied or empty.

    A seat is considered *occupied* if any detected person's bounding box
    overlaps the seat's bounding box with IoU ≥ *iou_threshold*.

    Args:
        persons: Person detections from :func:`extract_boxes_by_class`.
        seats:   Seat detections from :func:`extract_boxes_by_class`.
        iou_threshold: Minimum IoU to count as overlap.

    Returns:
        List of dicts, one per seat:
        ``{"seat_bbox": [...], "status": "occupied"|"empty", "iou": float}``.
    """
    results: list[dict[str, Any]] = []

    for seat in seats:
        seat_bbox: BBox = seat["bbox"]
        best_iou: float = 0.0

        for person in persons:
            iou = compute_iou(person["bbox"], seat_bbox)
            if iou > best_iou:
                best_iou = iou

        status = "occupied" if best_iou >= iou_threshold else "empty"
        results.append(
            {
                "seat_bbox": seat_bbox,
                "status": status,
                "iou": round(best_iou, 4),
            }
        )

    return results


# ─── Phone Usage Detection ─────────────────────────────────────────────────────

def detect_phone_usage(
    persons: list[Detection],
    phones: list[Detection],
    distance_threshold: float = config.PHONE_USE_DISTANCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Identify persons who are likely using a phone.

    A phone is considered *in use* by a person when the Euclidean distance
    between their centroids is below *distance_threshold* pixels.

    Args:
        persons: Person detections from :func:`extract_boxes_by_class`.
        phones:  Phone detections from :func:`extract_boxes_by_class`.
        distance_threshold: Maximum pixel distance to consider association.

    Returns:
        List of alert dicts for each matched (person, phone) pair:
        ``{"person_bbox": [...], "phone_bbox": [...],
           "person_center": (cx, cy), "distance_px": float}``.
    """
    alerts: list[dict[str, Any]] = []

    for person in persons:
        person_center = centroid(person["bbox"])

        for phone in phones:
            phone_center = centroid(phone["bbox"])
            dist = euclidean_distance(person_center, phone_center)

            if dist <= distance_threshold:
                alerts.append(
                    {
                        "person_bbox": person["bbox"],
                        "phone_bbox": phone["bbox"],
                        "person_center": person_center,
                        "distance_px": round(dist, 2),
                    }
                )
                # One phone can only be associated to one person (the nearest)
                break

    return alerts


# ─── Region-based Seat Status (for fixed seat_regions JSON) ───────────────────

def _point_in_bbox(point: tuple[int, int], bbox: BBox) -> bool:
    """Return True if *point* (cx, cy) lies inside *bbox* [x1,y1,x2,y2]."""
    cx, cy = point
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _bbox_overlap_ratio(det_bbox: BBox, seat_bbox: BBox) -> float:
    """Return the fraction of *det_bbox* that overlaps *seat_bbox*.

    This is more useful than IoU when the detected person bbox is much
    larger than the seat region (which is common for fixed-camera setups).

    Returns:
        Overlap area / det_bbox area, in [0.0, 1.0].
    """
    ix1 = max(det_bbox[0], seat_bbox[0])
    iy1 = max(det_bbox[1], seat_bbox[1])
    ix2 = min(det_bbox[2], seat_bbox[2])
    iy2 = min(det_bbox[3], seat_bbox[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    det_area = max(1, (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]))
    return intersection / det_area


def _nearest_person(
    phone: "Detection",
    persons: "list[Detection]",
    max_dist: float,
) -> "Optional[Detection]":
    """找到与 phone bbox 中心点距离最近且在阈值内的 person。

    Args:
        phone:    phone detection 字典，含 ``bbox`` 像素坐标字段。
        persons:  同帧所有 person detections。
        max_dist: 最大关联像素距离；超过则认为不相关。

    Returns:
        最近的 person detection，或 None（没有满足条件的人）。

    设计说明：
        phone 是小目标，bbox 很小，直接与 seat_region 做区域碰撞容易因
        坐标偏移而漏检。person 的 bbox 更稳定、覆盖面更大，手机大概率就在
        使用者身体附近，因此以「最近持有者」作为归属桥梁更合理。
    """
    phone_c = centroid(phone["bbox"])
    best_person, best_dist = None, float("inf")
    for p in persons:
        d = euclidean_distance(phone_c, centroid(p["bbox"]))
        if d < best_dist:
            best_dist = d
            best_person = p
    if best_dist <= max_dist:
        return best_person
    return None



def check_seat_status_from_regions(
    detections: list[Detection],
    seat_regions: list[dict],
    person_overlap_threshold: float = 0.10,
) -> list[dict]:
    """Determine per-seat occupancy and phone use from predefined seat regions.

    Unlike :func:`determine_seat_occupancy` (which needs YOLO to detect seat
    bboxes), this function uses a pre-loaded list of seat regions, making it
    suitable for fixed-camera deployments where seat positions are known.

    Strategy
    --------
    * **Person → occupied**: person centroid inside seat bbox, **or** person
      bbox overlap-ratio ≥ ``person_overlap_threshold``.
    * **Phone → phone_detected** *(NEW)*: Instead of checking whether the phone
      directly overlaps the seat region (unreliable for small objects), we:
        1. Find the nearest person to the phone (Euclidean centroid distance ≤
           ``config.PHONE_PERSON_MAX_DISTANCE`` pixels).
        2. Inherit that person's seat assignment — if the nearest person occupies
           seat N, the phone is attributed to seat N.
      This is more robust because the person bbox is larger and more stable; the
      phone is almost always within arm's reach of its user.

    Args:
        detections:              All detections from a single frame, each with
                                 keys ``type`` (or ``class_name``), ``confidence``,
                                 ``bbox`` (pixel coords).
        seat_regions:            List of seat dicts from the backend with keys
                                 ``seatId``, ``x1``, ``y1``, ``x2``, ``y2``
                                 (pixel coordinates after rescaling).
        person_overlap_threshold: Min person-bbox overlap ratio to count a hit.

    Returns:
        List of dicts, one per seat::

            {
                "seatId": int,
                "occupied": bool,
                "phone_detected": bool,
            }
    """
    # ── 0. 分类：兼容 type / class_name 两种字段名，兼容 phone / cell phone ──
    def _get_type(d: Detection) -> str:
        return (d.get("type") or d.get("class_name") or "").lower()

    persons = [d for d in detections if _get_type(d) == "person"]
    phones  = [
        d for d in detections
        if _get_type(d) in ("cell phone", "phone")
        and d.get("confidence", 1.0) >= config.PHONE_MIN_CONFIDENCE
    ]

    # ── 1. 为每个 person 计算命中的 seatId（None 表示不在任何区域内）────────
    # 格式：{person 在列表中的 index → seatId or None}
    person_seat: dict[int, Optional[int]] = {}

    for pidx, person in enumerate(persons):
        matched_seat = None
        p_center = centroid(person["bbox"])
        for seat in seat_regions:
            seat_bbox: BBox = [seat["x1"], seat["y1"], seat["x2"], seat["y2"]]
            # 优先：中心点直接落在区域内
            if _point_in_bbox(p_center, seat_bbox):
                matched_seat = seat["seatId"]
                break
            # 次选：bbox 重叠率达到阈值
            if _bbox_overlap_ratio(person["bbox"], seat_bbox) >= person_overlap_threshold:
                matched_seat = seat["seatId"]
                break
        person_seat[pidx] = matched_seat

    # ── 2. 确定 phone_detected 的座位集合：phone → 最近 person → seatId ────
    phone_detected_seat_ids: set[int] = set()

    for phone in phones:
        nearest = _nearest_person(
            phone=phone,
            persons=persons,
            max_dist=config.PHONE_PERSON_MAX_DISTANCE,
        )
        if nearest is None:
            continue
        pidx = persons.index(nearest)
        sid  = person_seat.get(pidx)
        if sid is not None:
            phone_detected_seat_ids.add(sid)

    # ── 3. 汇总每个 seat 的最终状态 ─────────────────────────────────────────
    results: list[dict] = []
    for seat in seat_regions:
        sid = seat["seatId"]
        occupied = sid in {v for v in person_seat.values() if v is not None}
        results.append(
            {
                "seatId": sid,
                "occupied": occupied,
                "phone_detected": sid in phone_detected_seat_ids,
            }
        )

    return results
