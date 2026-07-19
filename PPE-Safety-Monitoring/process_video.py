"""Worker xử lý video PPE - CHẠY TRONG TIẾN TRÌNH RIÊNG.

Vì sao tách riêng: torch/CUDA chạy chung tiến trình với Streamlit gây heap
corruption trên Windows (exit 0xc0000374, không traceback Python). Đã kiểm
chứng: đúng pipeline này chạy độc lập thì trót lọt, chỉ sập khi ở chung process
với Streamlit. Nên app.py (UI) gọi script này qua subprocess: torch không bao
giờ chung không gian bộ nhớ với Streamlit.

Giao tiếp:
- Nhận tham số qua CLI (đường dẫn video, ngưỡng, cấu hình frame-skip/voting).
- In tiến độ ra stdout dạng "PROGRESS <index> <total>" để UI cập nhật thanh %.
- Kết thúc: ghi kết quả (đường dẫn video + bảng thống kê) ra file JSON.

Chạy trực tiếp:
    python process_video.py --source in.mp4 --output out.mp4 --result res.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import supervision as sv
import torch

from utils.detector import PersonDetector, PPEDetector
from utils.matcher import match
from utils.smoother import PPEVoter
from utils.statistics import WorkerRegistry


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Xử lý video giám sát PPE (tiến trình riêng).")
    p.add_argument("--source", required=True, help="Video đầu vào")
    p.add_argument("--output", required=True, help="Video đầu ra (mp4v thô)")
    p.add_argument("--display", help="Video H.264 720p để xem/tải (ffmpeg)")
    p.add_argument("--result", required=True, help="File JSON ghi kết quả")
    p.add_argument("--person-conf", type=float, default=0.3)
    p.add_argument("--ppe-conf", type=float, default=0.3)
    p.add_argument("--ppe-every-n", type=int, default=3)
    p.add_argument("--vote-window", type=int, default=15)
    p.add_argument("--vote-min-ratio", type=float, default=0.4)
    p.add_argument("--show-ppe-boxes", action="store_true")
    p.add_argument("--person-model", default="models/yolov8s.pt")
    p.add_argument("--ppe-model", default="models/kaggle_best.pt")
    return p.parse_args()


def main() -> int:
    from utils.visualization import draw_ppe_boxes, draw_workers

    args = parse_args()
    device = get_device()
    print(f"DEVICE {device}", flush=True)

    person_detector = PersonDetector(model_path=args.person_model, device=device)
    ppe_detector = PPEDetector(model_path=args.ppe_model, device=device)
    person_detector.reset_tracker()

    voter = PPEVoter(window=args.vote_window, min_ratio=args.vote_min_ratio)
    registry = WorkerRegistry()

    video_info = sv.VideoInfo.from_video_path(args.source)
    total_frames = video_info.total_frames or 0

    last_ppe = [sv.Detections.empty()]

    def callback(frame, index):
        # Người: track MỖI frame (giữ ID liền mạch).
        tracked = person_detector.track(frame, confidence=args.person_conf)

        # PPE: chỉ chạy mỗi N frame (frame skipping).
        if index % args.ppe_every_n == 0:
            last_ppe[0] = ppe_detector.detect(frame, confidence=args.ppe_conf)
            raw_workers = match(tracked, last_ppe[0])
            voter.observe(raw_workers, index)

        # Áp trạng thái đã biểu quyết -> mượt, không nhấp nháy.
        workers = match(tracked, sv.Detections.empty())
        workers = voter.vote(workers)
        registry.update(workers, index)

        annotated = frame
        if args.show_ppe_boxes and len(last_ppe[0]) > 0:
            annotated = draw_ppe_boxes(annotated, last_ppe[0])
        annotated = draw_workers(annotated, workers)

        # Báo tiến độ cho UI (đọc qua stdout).
        print(f"PROGRESS {index + 1} {total_frames}", flush=True)
        return annotated

    sv.process_video(
        source_path=args.source,
        target_path=args.output,
        callback=callback,
    )

    # Convert sang H.264 720p để xem/tải trên trình duyệt. Video gốc có thể là
    # 4K -> nặng, phát nghẽn. Scale chiều cao tối đa 720p (giữ tỉ lệ, ép rộng
    # chẵn), nén CRF 23, +faststart để phát ngay khi chưa tải hết.
    display_path = None
    if args.display:
        # Chạy ffmpeg qua subprocess và NUỐT toàn bộ log vào DEVNULL. Nếu để
        # ffmpeg thừa hưởng stdout/stderr của worker (như os.system), đống log
        # libx264 sẽ làm đầy pipe nối tới app.py -> ffmpeg bị chặn ghi -> worker
        # treo ở 100%, không bao giờ in "DONE" (deadlock pipe).
        code = subprocess.run(
            [
                "ffmpeg", "-y", "-i", args.output,
                "-vf", "scale=-2:min(720\\,ih)",
                "-vcodec", "libx264", "-crf", "23", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                args.display,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        if code == 0 and os.path.exists(args.display):
            display_path = args.display

    # Ghi kết quả ra JSON để UI đọc lại (không truyền qua bộ nhớ chung).
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": device,
                "output_path": args.output,
                "display_path": display_path,
                "rows": registry.as_rows(),
            },
            f,
            ensure_ascii=False,
        )

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
