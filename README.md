# BƯỚC 1: Dán nội dung này vào file README.md trong thư mục dự án của bạn

# 🔍 Fake News Detector

Dự án phát hiện tin giả (Fake News) sử dụng các kỹ thuật Học máy (Machine Learning) và Xử lý ngôn ngữ tự nhiên (NLP) để phân loại tin tức dựa trên dữ liệu văn bản.

---

## 📋 Mục lục
1. [Giới thiệu](#-giới-thiệu)
2. [Quy trình dự án](#-quy-trình-dự-án)
3. [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
4. [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
5. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)

---

## 💡 Giới thiệu
Trong kỷ nguyên thông tin hiện nay, tin giả lan truyền với tốc độ chóng mặt. Dự án này được xây dựng nhằm mục đích xây dựng một bộ lọc thông minh, tự động phân loại một bài báo/tin tức là "Thật" (True) hay "Giả" (Fake) dựa trên mô hình đã được huấn luyện.

## ⚙️ Quy trình dự án
Dự án được chia thành 4 giai đoạn chính, được thể hiện qua các Jupyter Notebook:
* **`01_data_preprocessing.ipynb`**: Làm sạch dữ liệu, loại bỏ stopwords, xử lý văn bản, và chuẩn hóa dữ liệu đầu vào.
* **`02_model_training.ipynb`**: Xây dựng mô hình bằng thuật toán học máy, sử dụng `TfidfVectorizer` để trích xuất đặc trưng văn bản.
* **`03_model_evaluation.ipynb`**: Đánh giá hiệu quả mô hình thông qua độ chính xác (Accuracy), F1-score và ma trận nhầm lẫn (Confusion Matrix).
* **`04_inference_test.ipynb`**: Kiểm thử mô hình với các dữ liệu tin tức mới.

## 🛠️ Công nghệ sử dụng
* **Ngôn ngữ:** Python
* **Thư viện Data Science:** `pandas`, `numpy`, `scikit-learn`
* **Xử lý ngôn ngữ (NLP):** `nltk`
* **Triển khai ứng dụng:** `Flask` (để tạo API dự đoán trực tiếp)
* **Lưu trữ mô hình:** `pickle` (các file `.pkl`)

## 🚀 Hướng dẫn cài đặt

1. **Clone repository về máy:**
   ```bash
   git clone [https://github.com/aduylee/fake_news_detector.git](https://github.com/aduylee/fake_news_detector.git)
   cd fake_news_detector

## 📁 Cấu trúc thư mục
```text
FAKE_NEWS/
├── data/
│   ├── cleaned_data.csv
│   ├── Fake.csv
│   └── True.csv
├── models/
│   └── fake_news_model.pkl
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_inference_test.ipynb
├── src/
├── templates/
├── app_flask.py
├── requirements.txt
└── README.md
