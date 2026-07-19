# 👷 Hệ thống Giám sát Trang bị Bảo hộ Công nhân

Hệ thống thị giác máy tính phát hiện công nhân trong video công trường, bám đuổi từng người bằng ID riêng, và kiểm tra việc tuân thủ trang bị bảo hộ (**mũ bảo hộ** + **áo phản quang**) theo thời gian thực. Kết quả hiển thị qua giao diện Streamlit kèm bảng thống kê chi tiết.

## Tính năng chính

- **Phát hiện & bám đuổi công nhân** — YOLOv8s (COCO) + ByteTrack, mỗi công nhân một ID ổn định.
- **Phát hiện PPE** — model fine-tuned nhận diện mũ bảo hộ và áo phản quang.
- **Gán PPE cho từng người** — thuật toán Center Point Matching (tâm hộp PPE nằm trong khung người).
- **Đánh giá tuân thủ** — mỗi công nhân được phân loại: 🟢 An toàn / 🟠 Cảnh báo / 🔴 Nguy hiểm.
- **Chống nhấp nháy** — voting theo ID qua nhiều frame để trạng thái ổn định, không chớp tắt.
- **Bảng thống kê** — tổng hợp theo từng ID: tỉ lệ đội mũ / mặc áo, số frame xuất hiện, xuất CSV.

## Video kết quả

> Video nằm trong thư mục [`videos/`](videos/). GitHub không phát video từ đường dẫn tương đối — bấm vào link để tải/xem.

### Test 1
<video src="videos/test1.mp4" controls width="640"></video>

▶️ [videos/test1.mp4](videos/test1.mp4)

### Test 2
<video src="videos/test2.mp4" controls width="640"></video>

▶️ [videos/test2.mp4](videos/test2.mp4)

## Luồng xử lý

```text
Video ─► Phát hiện người (YOLOv8s) ─► ByteTrack (gán ID)
                                          │
      Phát hiện PPE (mũ + áo) ────────────┤
                                          ▼
                              Center Point Matching
                                          │
                                          ▼
                         Voting theo ID (làm mượt trạng thái)
                                          │
                                          ▼
                    Vẽ kết quả + Bảng thống kê (Streamlit)
```

## Cài đặt & chạy

Yêu cầu: **Python 3.10**, và **ffmpeg** có trên PATH (dùng để nén video kết quả).

```bash
# Tạo môi trường và cài phụ thuộc
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run app.py
```

Ứng dụng tự dùng **GPU (CUDA)** nếu có, không thì chạy CPU.

## Cấu trúc dự án

```text
app.py              # Giao diện Streamlit (UI thuần)
process_video.py    # Worker xử lý video, chạy ở tiến trình riêng
models/             # kaggle_best.pt (PPE), yolov8s.pt (tự tải)
tracker/            # Cấu hình ByteTrack
utils/              # detector, tracker, matcher, smoother, statistics, visualization
videos/             # Video kết quả (test1, test2)
```

> **Lưu ý:** `app.py` gọi `process_video.py` qua tiến trình con để tách PyTorch khỏi Streamlit — tránh lỗi treo/crash khi torch và Streamlit chạy chung tiến trình trên Windows.
