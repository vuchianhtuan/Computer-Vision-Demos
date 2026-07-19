"""Tính toán thống kê hiển thị trên dashboard (Phase 7).

Đầu vào là danh sách WorkerStatus của frame hiện tại; đầu ra là các con số
tổng hợp: số công nhân, số thiếu mũ/áo, số nguy hiểm, và FPS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from utils.matcher import (
    STATUS_DANGER,
    STATUS_SAFE,
    STATUS_WARNING,
    WorkerStatus,
)

# Ngưỡng kết luận một công nhân "có đội mũ / mặc áo" trong bảng tổng kết:
# chỉ cần detect thấy trang bị ở HƠN 15% số frame là coi như có. Đặt thấp vì
# model hay sót (false negative); miễn xuất hiện ổn định vài lần là tính có.
PPE_PRESENCE_RATIO = 0.15


@dataclass
class FrameStats:
    """Số liệu tổng hợp của một frame."""

    current_workers: int = 0
    missing_helmet: int = 0
    missing_vest: int = 0
    danger_workers: int = 0
    fps: float = 0.0


def compute_stats(workers: list[WorkerStatus], fps: float = 0.0) -> FrameStats:
    """Tổng hợp số liệu từ danh sách công nhân của frame."""
    return FrameStats(
        current_workers=len(workers),
        missing_helmet=sum(1 for w in workers if not w.has_helmet),
        missing_vest=sum(1 for w in workers if not w.has_vest),
        danger_workers=sum(1 for w in workers if w.status == STATUS_DANGER),
        fps=fps,
    )


class FPSTracker:
    """Đo FPS xử lý theo trung bình trượt (exponential moving average)."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._last_time: float | None = None
        self._fps: float = 0.0

    def tick(self) -> float:
        """Gọi mỗi khi xử lý xong một frame; trả về FPS ước lượng."""
        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                instant = 1.0 / dt
                self._fps = (
                    instant
                    if self._fps == 0.0
                    else self.smoothing * self._fps + (1 - self.smoothing) * instant
                )
        self._last_time = now
        return self._fps

    def reset(self) -> None:
        self._last_time = None
        self._fps = 0.0


@dataclass
class WorkerRecord:
    """Tổng hợp thông tin một công nhân (theo tracker_id) xuyên suốt video."""

    tracker_id: int
    frames_seen: int = 0          # số frame ID này xuất hiện
    helmet_frames: int = 0        # số frame được xác định có mũ (sau voting)
    vest_frames: int = 0          # số frame được xác định có áo (sau voting)
    first_frame: int = 0          # frame đầu tiên nhìn thấy
    last_frame: int = 0           # frame cuối cùng nhìn thấy
    last_status: str = STATUS_DANGER  # trạng thái ở lần xuất hiện gần nhất

    @property
    def helmet_ratio(self) -> float:
        return self.helmet_frames / self.frames_seen if self.frames_seen else 0.0

    @property
    def vest_ratio(self) -> float:
        return self.vest_frames / self.frames_seen if self.frames_seen else 0.0


class WorkerRegistry:
    """Tích luỹ thống kê từng công nhân qua toàn bộ video để dựng bảng cuối.

    Mỗi frame gọi update() với danh sách WorkerStatus (đã qua voting). Registry
    đếm số frame xuất hiện, số frame có mũ/áo, và lưu trạng thái gần nhất cho
    mỗi tracker_id.
    """

    def __init__(self) -> None:
        self._records: dict[int, WorkerRecord] = {}

    def update(self, workers: list[WorkerStatus], frame_index: int) -> None:
        for w in workers:
            rec = self._records.get(w.tracker_id)
            if rec is None:
                rec = WorkerRecord(tracker_id=w.tracker_id, first_frame=frame_index)
                self._records[w.tracker_id] = rec
            rec.frames_seen += 1
            rec.helmet_frames += int(w.has_helmet)
            rec.vest_frames += int(w.has_vest)
            rec.last_frame = frame_index
            rec.last_status = w.status

    def records(self) -> list[WorkerRecord]:
        """Danh sách record, sắp xếp theo tracker_id tăng dần."""
        return sorted(self._records.values(), key=lambda r: r.tracker_id)

    def as_rows(self) -> list[dict]:
        """Xuất ra danh sách dict để dựng bảng (st.dataframe)."""
        status_icon = {
            STATUS_SAFE: "🟢 An toàn",
            STATUS_WARNING: "🟠 Cảnh báo",
            STATUS_DANGER: "🔴 Nguy hiểm",
        }
        rows = []
        for r in self.records():
            # Trạng thái tính theo TỔNG HỢP cả video: coi là có mũ/áo nếu detect
            # thấy ở hơn PPE_PRESENCE_RATIO số frame, rồi suy ra Safe/Warning/Danger
            # (thay vì lấy trạng thái tức thời ở frame cuối).
            has_helmet = r.helmet_ratio > PPE_PRESENCE_RATIO
            has_vest = r.vest_ratio > PPE_PRESENCE_RATIO
            if has_helmet and has_vest:
                status = STATUS_SAFE
            elif not has_helmet and not has_vest:
                status = STATUS_DANGER
            else:
                status = STATUS_WARNING
            rows.append(
                {
                    "ID": r.tracker_id,
                    "Trạng thái": status_icon.get(status, status),
                    "Có mũ": "✔" if has_helmet else "✘",
                    "Có áo": "✔" if has_vest else "✘",
                    "Tỉ lệ đội mũ": f"{r.helmet_ratio * 100:.0f}%",
                    "Tỉ lệ mặc áo": f"{r.vest_ratio * 100:.0f}%",
                    "Số frame xuất hiện": r.frames_seen,
                }
            )
        return rows

    def reset(self) -> None:
        self._records.clear()
