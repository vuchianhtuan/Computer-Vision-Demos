"""So khớp PPE với công nhân bằng thuật toán Center Point Matching.

Ý tưởng (theo README - Phase 5):
- Tính tâm của mỗi bounding box PPE (mũ / áo).
- Nếu tâm nằm trong bounding box của một công nhân thì gán PPE đó cho họ.

Trạng thái tuân thủ (Phase 6):
    Mũ ✓ + Áo ✓ -> Safe
    Thiếu một trong hai -> Warning
    Thiếu cả hai      -> Danger
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from utils.detector import PPE_HARDHAT_CLASS_ID, PPE_VEST_CLASS_ID

# Các nhãn trạng thái tuân thủ.
STATUS_SAFE = "Safe"
STATUS_WARNING = "Warning"
STATUS_DANGER = "Danger"


@dataclass
class WorkerStatus:
    """Trạng thái PPE của một công nhân được track."""

    tracker_id: int
    bbox: np.ndarray  # [x_min, y_min, x_max, y_max]
    has_helmet: bool = False
    has_vest: bool = False

    @property
    def status(self) -> str:
        if self.has_helmet and self.has_vest:
            return STATUS_SAFE
        if not self.has_helmet and not self.has_vest:
            return STATUS_DANGER
        return STATUS_WARNING


def _center(bbox: np.ndarray) -> tuple[float, float]:
    """Tâm (cx, cy) của bbox dạng [x_min, y_min, x_max, y_max]."""
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def _point_in_bbox(cx: float, cy: float, bbox: np.ndarray) -> bool:
    x_min, y_min, x_max, y_max = bbox
    return x_min <= cx <= x_max and y_min <= cy <= y_max


def match(
    tracked_persons: sv.Detections,
    ppe_detections: sv.Detections,
) -> list[WorkerStatus]:
    """Gán PPE cho từng công nhân đã được track.

    Args:
        tracked_persons: Detections người, đã có tracker_id.
        ppe_detections: Detections PPE (Hardhat + Safety Vest).

    Returns:
        Danh sách WorkerStatus, mỗi phần tử ứng với một công nhân.
    """
    workers: list[WorkerStatus] = []

    # model.track() có thể chưa gán ID ở một số frame -> tracker_id là None
    # (cả mảng None, hoặc từng phần tử None). Bỏ qua các box chưa có ID.
    track_ids = tracked_persons.tracker_id
    if track_ids is None:
        return workers

    for i in range(len(tracked_persons)):
        if track_ids[i] is None:
            continue
        person_bbox = tracked_persons.xyxy[i]
        tracker_id = int(track_ids[i])
        workers.append(WorkerStatus(tracker_id=tracker_id, bbox=person_bbox))

    for j in range(len(ppe_detections)):
        ppe_bbox = ppe_detections.xyxy[j]
        ppe_class = int(ppe_detections.class_id[j])
        cx, cy = _center(ppe_bbox)

        for worker in workers:
            if _point_in_bbox(cx, cy, worker.bbox):
                if ppe_class == PPE_HARDHAT_CLASS_ID:
                    worker.has_helmet = True
                elif ppe_class == PPE_VEST_CLASS_ID:
                    worker.has_vest = True
                break  # tâm PPE chỉ thuộc về công nhân đầu tiên chứa nó

    return workers
