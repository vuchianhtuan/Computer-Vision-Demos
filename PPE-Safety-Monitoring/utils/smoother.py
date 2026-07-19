"""Làm mượt kết quả PPE theo thời gian bằng biểu quyết (voting) theo tracker_id.

Vấn đề: model PPE nhấp nháy giữa các frame - cùng một công nhân lúc detect
được mũ/áo, lúc lại sót. Nếu vẽ trực tiếp kết quả từng frame thì trạng thái
Safe/Warning/Danger chớp tắt liên tục.

Giải pháp: với mỗi tracker_id, lưu lại các quan sát has_helmet/has_vest của
vài frame gần nhất (cửa sổ trượt). Trạng thái hiển thị = biểu quyết đa số trong
cửa sổ đó -> ổn định, không giật.

Kết hợp với frame skipping ở app.py: PPE detection chỉ chạy mỗi N frame; ở các
frame còn lại chỉ cần gọi vote() để áp lại trạng thái đã học cho công nhân đang
được track.
"""

from __future__ import annotations

from collections import defaultdict, deque

from utils.matcher import WorkerStatus


class PPEVoter:
    """Biểu quyết đa số theo cửa sổ trượt cho trạng thái PPE của từng công nhân.

    Args:
        window: số quan sát PPE gần nhất được giữ cho mỗi công nhân.
        min_ratio: tỉ lệ tối thiểu số frame "có" để kết luận là đang mang PPE.
            Đặt < 0.5 để thiên về "có" (do sót detect - false negative - hay
            gặp hơn là báo nhầm).
        max_missing: số frame một ID vắng mặt trước khi xoá lịch sử (giải phóng
            bộ nhớ, tránh ID cũ ảnh hưởng khi ByteTrack/BoT-SORT tái sử dụng ID).
    """

    def __init__(
        self,
        window: int = 15,
        min_ratio: float = 0.4,
        max_missing: int = 60,
    ):
        self.window = window
        self.min_ratio = min_ratio
        self.max_missing = max_missing
        self._helmet: dict[int, deque[bool]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._vest: dict[int, deque[bool]] = defaultdict(lambda: deque(maxlen=window))
        self._last_seen: dict[int, int] = {}

    def observe(self, workers: list[WorkerStatus], frame_index: int) -> None:
        """Ghi nhận quan sát PPE thô của một frame CÓ chạy PPE detection."""
        for w in workers:
            self._helmet[w.tracker_id].append(w.has_helmet)
            self._vest[w.tracker_id].append(w.has_vest)
            self._last_seen[w.tracker_id] = frame_index
        self._prune(frame_index)

    def vote(self, workers: list[WorkerStatus]) -> list[WorkerStatus]:
        """Ghi đè has_helmet/has_vest bằng kết quả biểu quyết đã tích luỹ.

        Sửa trực tiếp trên các WorkerStatus truyền vào rồi trả lại. Công nhân
        chưa có lịch sử (ID mới) giữ nguyên giá trị thô cho tới khi tích đủ
        quan sát.
        """
        for w in workers:
            helmet_hist = self._helmet.get(w.tracker_id)
            vest_hist = self._vest.get(w.tracker_id)
            if helmet_hist:
                w.has_helmet = self._majority(helmet_hist)
            if vest_hist:
                w.has_vest = self._majority(vest_hist)
        return workers

    def reset(self) -> None:
        """Xoá toàn bộ lịch sử (gọi khi bắt đầu video mới)."""
        self._helmet.clear()
        self._vest.clear()
        self._last_seen.clear()

    def _majority(self, history: deque[bool]) -> bool:
        return (sum(history) / len(history)) >= self.min_ratio

    def _prune(self, frame_index: int) -> None:
        stale = [
            tid
            for tid, seen in self._last_seen.items()
            if frame_index - seen > self.max_missing
        ]
        for tid in stale:
            self._helmet.pop(tid, None)
            self._vest.pop(tid, None)
            self._last_seen.pop(tid, None)
