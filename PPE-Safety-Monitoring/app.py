"""Hệ thống Giám sát An toàn Lao động - Streamlit Dashboard (UI thuần).

QUAN TRỌNG - vì sao app.py KHÔNG import torch/ultralytics/supervision:
torch/CUDA chạy chung tiến trình với Streamlit gây heap corruption trên Windows
(thoát với exit 0xc0000374, KHÔNG có traceback Python). Đã kiểm chứng bằng
faulthandler + chạy pipeline độc lập: đúng pipeline này chạy riêng thì trót lọt,
chỉ sập khi ở chung process với Streamlit. Nên toàn bộ phần nặng (detect, track,
PPE, ffmpeg) được tách sang process_video.py và gọi qua SUBPROCESS. app.py chỉ
làm UI: nhận cấu hình, chạy tiến trình con, đọc kết quả (video + JSON) từ đĩa.

Pipeline (trong process_video.py):
    Video -> PersonDetector.track (YOLOv8s + ByteTrack) -> PPEDetector
          -> Center Point Matching -> Voting theo ID -> Visualization
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile

import streamlit as st

# ----------------------------------------------------
# CẤU HÌNH TRANG
# ----------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Giám sát An toàn Lao động",
    page_icon="👷",
    layout="wide",
)

st.title("👷 Hệ thống Giám sát Trang bị Bảo hộ Công nhân")
st.subheader("Phát hiện Mũ bảo hộ & Áo phản quang, bám đuổi công nhân theo thời gian thực")

# Thư mục dự án (để định vị process_video.py dù chạy từ đâu).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(PROJECT_DIR, "process_video.py")


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.header("Cấu hình")

    person_conf = st.slider("Ngưỡng tin cậy - Người", 0.1, 0.9, 0.3, 0.05)
    ppe_conf = st.slider("Ngưỡng tin cậy - PPE", 0.1, 0.9, 0.3, 0.05)
    show_ppe_boxes = st.checkbox("Hiển thị box PPE gốc", value=True)

    st.markdown("---")
    st.subheader("Tối ưu tốc độ & ổn định")
    # Frame skipping: chỉ chạy PPE detection mỗi N frame để giảm tải.
    # Người vẫn được track mọi frame (không được bỏ, nếu không ID sẽ đứt).
    ppe_every_n = st.slider(
        "Chạy PPE mỗi N frame", 1, 10, 3, 1,
        help="1 = mỗi frame. Tăng lên để nhanh hơn; trạng thái giữ bằng voting.",
    )
    # Voting: làm mượt kết quả PPE theo tracker_id qua cửa sổ vài frame gần nhất.
    vote_window = st.slider(
        "Cửa sổ biểu quyết (frame)", 3, 45, 15, 1,
        help="Số lần detect PPE gần nhất dùng để biểu quyết trạng thái mỗi công nhân.",
    )
    vote_min_ratio = st.slider(
        "Ngưỡng biểu quyết 'có PPE'", 0.1, 0.9, 0.4, 0.05,
        help="Tỉ lệ frame tối thiểu detect thấy PPE để kết luận là đang mang. "
        "Thấp hơn 0.5 để bù cho việc hay sót detect.",
    )

    st.markdown("---")
    st.markdown(
        "**Trạng thái tuân thủ:**\n"
        "* 🟢 Safe: đủ mũ + áo\n"
        "* 🟠 Warning: thiếu 1 trong 2\n"
        "* 🔴 Danger: thiếu cả hai"
    )


def run_worker(source: str, output: str, display: str, result: str) -> tuple[int, str]:
    """Chạy process_video.py trong tiến trình con, cập nhật thanh tiến độ.

    Đọc stdout theo dòng: "PROGRESS <i> <total>" -> cập nhật %, "DEVICE ..." /
    "DONE" -> thông tin. Trả về (mã thoát, stderr gộp) để UI báo lỗi nếu có.
    """
    cmd = [
        sys.executable, WORKER_SCRIPT,
        "--source", source,
        "--output", output,
        "--display", display,
        "--result", result,
        "--person-conf", str(person_conf),
        "--ppe-conf", str(ppe_conf),
        "--ppe-every-n", str(ppe_every_n),
        "--vote-window", str(vote_window),
        "--vote-min-ratio", str(vote_min_ratio),
    ]
    if show_ppe_boxes:
        cmd.append("--show-ppe-boxes")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Gộp stderr vào stdout: chỉ còn MỘT pipe để đọc. Nếu tách hai pipe mà chỉ
    # đọc stdout trong vòng lặp, buffer stderr đầy (ffmpeg/torch log nhiều) sẽ
    # chặn tiến trình con -> treo ở 100%, không in DONE. Một pipe = không deadlock.
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    # Giữ lại vài dòng cuối để hiển thị nếu worker lỗi (không phải PROGRESS/DEVICE).
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("PROGRESS"):
            try:
                _, idx, total = line.split()
                idx, total = int(idx), int(total)
                if total:
                    percent = min(int(idx / total * 100), 100)
                    progress_bar.progress(percent)
                    status_text.text(f"Đang xử lý: Frame {idx}/{total} ({percent}%)")
            except ValueError:
                pass
        elif line.startswith("DEVICE"):
            status_text.text(f"Thiết bị suy luận: {line.split(maxsplit=1)[-1].upper()}")
        elif line and line != "DONE":
            tail.append(line)
            tail = tail[-40:]  # chỉ giữ 40 dòng gần nhất

    proc.wait()
    progress_bar.progress(100)
    return proc.returncode, "\n".join(tail)


# ----------------------------------------------------
# GIAO DIỆN UPLOAD & XỬ LÝ
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "Chọn video từ máy tính của bạn", type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns(2)
    with col1:
        st.info("Video Gốc")
        st.video(uploaded_file)

    with col2:
        st.info("Video Kết Quả")
        start = st.button("▶ Bắt đầu phân tích video")

    if start:
        tmp = tempfile.gettempdir()
        output_path = os.path.join(tmp, "output_processed.mp4")
        display_path = os.path.join(tmp, "output_converted.mp4")
        result_path = os.path.join(tmp, "ppe_result.json")

        with st.spinner("Hệ thống AI đang phân tích... Vui lòng đợi."):
            code, err = run_worker(video_path, output_path, display_path, result_path)

        if code != 0:
            st.error(f"Xử lý thất bại (mã lỗi {code}). Chi tiết:")
            st.code(err or "(không có thông tin lỗi)")
            st.session_state.pop("result", None)
        else:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            st.success("Xử lý hoàn tất!")
            st.session_state["result"] = data

    # ------------------------------------------------------------------
    # Hiển thị kết quả từ session_state (chạy cả khi rerun do bấm nút tải).
    # ------------------------------------------------------------------
    result = st.session_state.get("result")
    if result:
        display = result.get("display_path")
        if display and os.path.exists(display):
            st.video(display)
        else:
            st.warning("Không hiển thị trực tiếp được (thiếu ffmpeg?). Tải file kết quả về máy:")

        # Tải bản đã nén (H.264 720p) nếu có, không thì bản gốc.
        download_path = display if (display and os.path.exists(display)) else result.get("output_path")
        if download_path and os.path.exists(download_path):
            with open(download_path, "rb") as f:
                st.download_button(
                    label="⬇️ Tải xuống video kết quả",
                    data=f.read(),
                    file_name="safety_tracked_output.mp4",
                    mime="video/mp4",
                )

        # Bảng thống kê chi tiết theo từng ID công nhân (tổng hợp cả video).
        st.markdown("---")
        st.subheader("📋 Thống kê theo từng công nhân (ID)")
        rows = result.get("rows", [])
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption(
                "Cột 'Có mũ'/'Có áo' = có trang bị ở hơn 15% số frame xuất hiện. "
                "'Tỉ lệ' = phần trăm frame ID đó được xác định có trang bị."
            )
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            st.download_button(
                label="⬇️ Tải bảng thống kê (CSV)",
                data=buf.getvalue().encode("utf-8-sig"),
                file_name="worker_ppe_stats.csv",
                mime="text/csv",
            )
        else:
            st.info("Không phát hiện công nhân nào trong video.")
