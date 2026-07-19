"""Object detectors dùng cho pipeline giám sát PPE.

- PersonDetector: dùng model COCO pretrained (yolov8s.pt), chỉ lấy class person.
- PPEDetector: dùng model fine-tuned (models/best.pt), lấy Hardhat + Safety Vest.

Cả hai trả về supervision.Detections để pipeline phía sau xử lý đồng nhất.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO

# Class id của "person" trong bộ dữ liệu COCO.
COCO_PERSON_CLASS_ID = 0

# Class id trong model PPE (models/kaggle_best.pt).
# Model này có 10 class: {0: Hardhat, 1: Mask, 2: NO-Hardhat, 3: NO-Mask,
# 4: NO-Safety Vest, 5: Person, 6: Safety Cone, 7: Safety Vest, 8: machinery,
# 9: vehicle}. Chỉ dùng 2 nhãn cần thiết, lọc bỏ phần còn lại.
PPE_HARDHAT_CLASS_ID = 0
PPE_VEST_CLASS_ID = 7

# Danh sách class PPE cần giữ khi detect (bỏ các nhãn thừa: NO-*, Person,
# Safety Cone, machinery, vehicle... để không vẽ/khớp nhầm).
PPE_KEEP_CLASS_IDS = [PPE_HARDHAT_CLASS_ID, PPE_VEST_CLASS_ID]


class PersonDetector:
    """Phát hiện + bám đuổi công nhân bằng model COCO pretrained.

    Model yolov8s.pt sẽ được ultralytics tự tải về lần đầu chạy.

    Dùng tracking tích hợp của Ultralytics (ByteTrack) qua model.track():
    ByteTrack khớp theo IoU + confidence (không ReID), nhanh và nhẹ hơn
    BoT-SORT. Cấu hình ở tracker/bytetrack.yaml.
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        device: str = "cpu",
        tracker_config: str = "tracker/bytetrack.yaml",
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.tracker_config = tracker_config

    def detect(self, frame: np.ndarray, confidence: float = 0.3) -> sv.Detections:
        """Chỉ phát hiện người, không gán ID (dùng khi không cần tracking)."""
        result = self.model.predict(
            frame,
            conf=confidence,
            classes=[COCO_PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def track(self, frame: np.ndarray, confidence: float = 0.3) -> sv.Detections:
        """Phát hiện người + gán tracker_id ổn định bằng tracker cấu hình.

        Tracker do tracker_config quyết định (mặc định ByteTrack).
        persist=True để tracker giữ trạng thái xuyên suốt các frame của video.
        Trả về sv.Detections đã có tracker_id (khớp interface pipeline cũ).
        """
        result = self.model.track(
            frame,
            conf=confidence,
            classes=[COCO_PERSON_CLASS_ID],
            device=self.device,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def reset_tracker(self) -> None:
        """Xoá trạng thái tracker để bắt đầu một video mới."""
        # Ultralytics khởi tạo lại tracker khi predictor bị reset.
        if getattr(self.model, "predictor", None) is not None:
            trackers = getattr(self.model.predictor, "trackers", None)
            if trackers:
                for t in trackers:
                    if hasattr(t, "reset"):
                        t.reset()


class PPEDetector:
    """Phát hiện Hardhat và Safety Vest bằng model fine-tuned.

    models/kaggle_best.pt có 10 class (Hardhat, Mask, Person, vehicle, ...).
    Ta chỉ lọc đúng 2 class cần dùng: Hardhat (0) và Safety Vest (7), để không
    vẽ nhầm các nhãn thừa. Class id được import từ hằng số phía trên.
    """

    def __init__(self, model_path: str = "models/kaggle_best.pt", device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame: np.ndarray, confidence: float = 0.3) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=confidence,
            classes=[PPE_HARDHAT_CLASS_ID, PPE_VEST_CLASS_ID],
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)
