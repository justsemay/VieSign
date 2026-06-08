# Sign Language Web Demo v4

Bản v4 chốt chức năng cuối cùng trước khi viết lại báo cáo.

## Các chức năng chính

- Nhận dạng ký hiệu qua webcam bằng backend Python + MediaPipe + Keras.
- Hiển thị kết quả tiếng Việt có dấu.
- Tự động phát âm tiếng Việt khi hệ thống xác nhận thêm từ mới vào câu.
- Có nút bật/tắt tự động phát âm để thuận tiện khi debug hoặc bảo vệ đồ án.
- Có hướng dẫn sử dụng nhanh ngay dưới camera.
- Có cảnh báo khi MediaPipe chưa thấy pose, chưa thấy bàn tay hoặc chỉ thấy một tay.
- Có chế độ Chuyên gia / Người dùng.
- Có nút Xóa từ cuối và Xóa toàn bộ kết quả.
- Có upload video mẫu ở chế độ Chuyên gia để dự phòng khi webcam lỗi.

## Chạy

```bash
cd backend
python app.py
```

Mở trình duyệt:

```text
http://127.0.0.1:5000
```

## Lưu ý

Copy model `.keras` và file `*_norm_stats.npz` vào `backend/models/` trước khi chạy.
