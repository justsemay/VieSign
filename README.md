# VieSign - Hệ thống hỗ trợ giao tiếp cho người khiếm thính 🤟

![VieSign Banner](link_ảnh_giao_diện_tổng_quan_của_trò.png)

**VieSign** là ứng dụng Web trí tuệ nhân tạo hỗ trợ nhận dạng ngôn ngữ ký hiệu tiếng Việt (VSL) theo thời gian thực. Dự án được xây dựng nhằm thu hẹp khoảng cách giao tiếp giữa người khiếm thính và cộng đồng thông qua việc chuyển đổi cử chỉ tay thành văn bản và giọng nói.

Đồ án tốt nghiệp - Trường Đại học Giao thông Vận tải (UTC).
* **Sinh viên thực hiện:** Đào Minh Quang
* **Mô hình AI:** CNN1D kết hợp BiLSTM

---

## ✨ Tính năng nổi bật

* 📷 **Nhận dạng Real-time:** Phân tích dữ liệu Landmark từ camera và dự đoán ký hiệu ngay trên trình duyệt.
* 🗣️ **Phát âm tự động (Text-to-Speech):** Chuyển đổi văn bản kết quả thành giọng nói tiếng Việt bằng Web Speech API.
* 🎬 **Hỗ trợ Upload Video:** Xử lý và nhận dạng ngôn ngữ ký hiệu từ các video mẫu được tải lên.
* 🧠 **Thuật toán tối ưu:** Cơ chế Hậu xử lý (Post-processing) thông minh giúp lọc nhiễu, chống lặp từ và loại bỏ trạng thái nghỉ (IDLE).
* 🎨 **Giao diện hiện đại:** Thiết kế tone màu Xanh Navy chuyên nghiệp, tối ưu UX/UI cho trải nghiệm mượt mà.

---

## 🛠️ Công nghệ sử dụng

### 1. Trí tuệ nhân tạo (AI) & Xử lý ảnh
* **TensorFlow / Keras:** Xây dựng và huấn luyện mô hình mạng nơ-ron chuỗi thời gian (CNN1D + BiLSTM).
* **MediaPipe:** Trích xuất 225 tọa độ không gian (Landmarks) của cơ thể và hai bàn tay.
* **OpenCV:** Giải mã (Base64) và tiền xử lý khung hình ảnh đầu vào.

### 2. Backend
* **Python 3.9+**
* **Flask:** Viết các Restful API tiếp nhận luồng dữ liệu xử lý mô hình AI.

### 3. Frontend
* **HTML5, CSS3, Vanilla JavaScript**
* **Web Speech API:** Tổng hợp âm thanh tiếng Việt.

---

## 📂 Cấu trúc thư mục lõi

```text
VieSign/
│
├── app.py                      # Controller chính khởi chạy Flask Server
├── requirements.txt            # Danh sách các thư viện cần thiết
├── models/
│   ├── model.h5                # File trọng số mô hình đã huấn luyện
│   └── scaler.pkl              # Tham số chuẩn hóa dữ liệu (Mean/Std)
│
├── services/                   # Các Module xử lý nghiệp vụ AI
│   ├── image_utils.py          # Tiền xử lý ảnh OpenCV
│   ├── mediapipe_service.py    # Trích xuất 225 Landmarks
│   ├── recognition_service.py  # Quản lý bộ đệm 30 frames & Hậu xử lý
│   └── model_service.py        # Tải mô hình và suy luận xác suất
│
├── static/                     # Chứa CSS, JS, Images cho Frontend
└── templates/                  # Chứa file giao diện HTML chính
