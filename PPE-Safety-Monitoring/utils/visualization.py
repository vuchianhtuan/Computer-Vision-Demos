"""Vẽ kết quả lên frame video (Phase 7 - Video Display).

Mỗi công nhân được vẽ một bounding box đổi màu theo trạng thái tuân thủ:
    Safe    -> xanh lá
    Warning -> vàng/cam
    Danger  -> đỏ

Kèm nhãn dạng: "CN #ID | Mu: OK/X | Ao: OK/X".
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.matcher import STATUS_DANGER, STATUS_SAFE, STATUS_WARNING, WorkerStatus

# Bảng màu (BGR) cho từng trạng thái.
STATUS_COLORS = {
    STATUS_SAFE: (0, 200, 0),      # xanh lá
    STATUS_WARNING: (0, 165, 255),  # cam
    STATUS_DANGER: (0, 0, 255),    # đỏ
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Kích thước chữ & độ dày nét cho nhãn (to, rõ, dễ đọc trên video).
_FONT_SCALE = 0.9
_TEXT_THICKNESS = 2
_BOX_THICKNESS = 3
_PAD = 6  # đệm quanh chữ trong nền nhãn


def _label(worker: WorkerStatus) -> str:
    helmet = "Mu: OK" if worker.has_helmet else "Mu: X"
    vest = "Ao: OK" if worker.has_vest else "Ao: X"
    return f"CN #{worker.tracker_id} | {helmet} | {vest}"


def draw_workers(
    frame: np.ndarray,
    workers: list[WorkerStatus],
    draw_label: bool = True,
) -> np.ndarray:
    """Vẽ box + nhãn cho từng công nhân lên một bản copy của frame."""
    annotated = frame.copy()

    for worker in workers:
        color = STATUS_COLORS[worker.status]
        x_min, y_min, x_max, y_max = worker.bbox.astype(int)

        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, _BOX_THICKNESS)

        if not draw_label:
            continue

        text = _label(worker)
        (tw, th), baseline = cv2.getTextSize(
            text, _FONT, _FONT_SCALE, _TEXT_THICKNESS
        )
        box_h = th + baseline + 2 * _PAD
        box_w = tw + 2 * _PAD

        # Nền nhãn phía trên box; nếu chạm mép trên thì đẩy xuống trong box.
        top = y_min - box_h
        if top < 0:
            top = y_min
        cv2.rectangle(
            annotated,
            (x_min, top),
            (x_min + box_w, top + box_h),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            text,
            (x_min + _PAD, top + th + _PAD),
            _FONT,
            _FONT_SCALE,
            (255, 255, 255),
            _TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated


def draw_ppe_boxes(frame: np.ndarray, ppe_detections) -> np.ndarray:
    """Vẽ mờ nhẹ các box PPE gốc (mũ/áo) để tăng tính trực quan."""
    annotated = frame.copy()
    for i in range(len(ppe_detections)):
        x_min, y_min, x_max, y_max = ppe_detections.xyxy[i].astype(int)
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (200, 200, 200), 1)
    return annotated
