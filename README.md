# 🧠 Hệ Thống Phân Loại Khối U Não MRI

## Tổng quan Dự án

Dự án này tập trung vào việc xây dựng một hệ thống thông minh có khả năng phân loại khối u não từ ảnh chụp cộng hưởng từ (MRI) thành 4 loại chính: **Glioma (U thần kinh đệm)**, **Meningioma (U màng não)**, **No Tumor (Không có khối u)** và **Pituitary (U tuyến yên)**. Hệ thống ứng dụng các kỹ thuật học sâu tiên tiến, cụ thể là kiến trúc ResNet50V2 kết hợp với cơ chế Chú ý (Attention Mechanism), nhằm đạt được độ chính xác cao trong chẩn đoán, hỗ trợ các chuyên gia y tế.

---

## 📂 Cấu trúc Dự án

Dưới đây là cấu trúc thư mục và các tệp chính của dự án:
Chắc chắn rồi! Dưới đây là phiên bản README.md được định dạng Markdown, giúp hiển thị đẹp và chuyên nghiệp hơn trên các nền tảng như GitHub, GitLab, v.v. Tôi đã thêm vào các định dạng như tiêu đề, danh sách, khối code, và làm nổi bật một số thông tin quan trọng.

Markdown

# 🧠 Hệ Thống Phân Loại Khối U Não MRI

## Tổng quan Dự án

Dự án này tập trung vào việc xây dựng một hệ thống thông minh có khả năng phân loại khối u não từ ảnh chụp cộng hưởng từ (MRI) thành 4 loại chính: **Glioma (U thần kinh đệm)**, **Meningioma (U màng não)**, **No Tumor (Không có khối u)** và **Pituitary (U tuyến yên)**. Hệ thống ứng dụng các kỹ thuật học sâu tiên tiến, cụ thể là kiến trúc ResNet50V2 kết hợp với cơ chế Chú ý (Attention Mechanism), nhằm đạt được độ chính xác cao trong chẩn đoán, hỗ trợ các chuyên gia y tế.

---

## 📂 Cấu trúc Dự án

Dưới đây là cấu trúc thư mục và các tệp chính của dự án:

.
├── dataset_1/                  # Tập dữ liệu gốc thứ nhất
├── dataset_2/                  # Tập dữ liệu gốc thứ hai
├── dataset_3/                  # Tập dữ liệu gốc thứ ba
├── notebooks/
│   ├── EDA.ipynb               # Phân tích khám phá dữ liệu (EDA)
│   ├── Preprocessing_dataset_1.ipynb # Tiền xử lý cho tập dữ liệu 1
│   ├── Preprocessing_dataset_2.ipynb # Tiền xử lý cho tập dữ liệu 2
│   ├── Preprocessing_dataset_3.ipynb # Tiền xử lý cho tập dữ liệu 3
│   ├── phanloai_dataset_1.ipynb # Chia dữ liệu tập dataset_1 thành train/val/test
│   ├── phanloai_dataset_2.ipynb # Chia dữ liệu tập dataset_2 thành train/val/test
│   └── ResNet50V2_voi_Attention_96_.ipynb # Notebook huấn luyện mô hình chính, chạy trên colab
├── app.py
├── best_resnet_attention_model.h5 
└── README.md                   

---

## 🚀 Các Giai đoạn Xử lý Chính

Dự án được thực hiện qua các giai đoạn chính sau:

1.  **Phân tích Khám phá Dữ liệu (EDA)** (`notebooks/EDA.ipynb`)
    * Phân tích sự phân bố dữ liệu của các lớp.
    * Kiểm tra kích thước ảnh và định dạng ban đầu.
    * Phân tích sơ bộ về độ sáng, độ tương phản của ảnh.
    * (Tùy chọn) Trích xuất các đặc trưng cơ bản ban đầu để hiểu rõ hơn về dữ liệu.

2.  **Tiền xử lý Dữ liệu** (`notebooks/Preprocessing_dataset_*.ipynb`)
    * **Resize ảnh:** Đưa tất cả ảnh về kích thước thống nhất (ví dụ: 224x224 pixels) để phù hợp với đầu vào của mô hình.
    * **Cân bằng dữ liệu:** Áp dụng các kỹ thuật tăng cường dữ liệu (Data Augmentation) cho tập huấn luyện để cân bằng số lượng mẫu giữa các lớp, giúp mô hình học tốt hơn và tránh thiên vị.
    * **Chuẩn hóa giá trị pixel:** Chuyển đổi giá trị pixel về khoảng [0, 1] để ổn định quá trình huấn luyện.
    * **Phân chia dữ liệu:** Chia mỗi bộ dữ liệu thành các tập huấn luyện (train), kiểm định (validation) và kiểm tra (test) theo tỷ lệ phù hợp.

3.  **Xây dựng và Huấn luyện Mô hình** (`notebooks/ResNet50V2_voi_Attention_96_.ipynb`)
    * **Thử nghiệm kiến trúc:** Đã xem xét và thử nghiệm với các kiến trúc CNN phổ biến như MobileNetV2, EfficientNetB3, và ResNet50V2.
    * **Lựa chọn cuối cùng:** ResNet50V2 được chọn làm mô hình nền nhờ khả năng trích xuất đặc trưng mạnh mẽ.
    * **Tích hợp Cơ chế Chú ý (Attention):** Một khối chú ý được thêm vào kiến trúc ResNet50V2 để giúp mô hình tập trung vào các vùng ảnh quan trọng, cải thiện hiệu suất phân loại.
    * **Học Chuyển giao (Transfer Learning):** Tận dụng trọng số đã được tiền huấn luyện trên bộ dữ liệu ImageNet cho phần ResNet50V2.
    * **Tùy chỉnh lớp phân loại:** Các lớp kết nối đầy đủ (custom layers) được thêm vào cuối mô hình để phân loại ra 4 loại khối u mục tiêu.
    * **Kết quả:** Mô hình ResNet50V2 kết hợp Attention đạt độ chính xác trên tập kiểm tra khoảng **96%** (cần xác nhận lại con số chính xác này dựa trên bộ dữ liệu nào bạn muốn báo cáo ở đây).

4.  **Đánh giá Mô hình**
    * Sử dụng các chỉ số đo lường tiêu chuẩn: Accuracy, Precision, Recall, F1-score.
    * Phân tích chi tiết Ma trận nhầm lẫn (Confusion Matrix) để hiểu rõ các loại lỗi của mô hình.
    * Đánh giá độ chính xác phân loại cho từng lớp riêng biệt.
    * Trực quan hóa một số dự đoán mẫu (đúng và sai) để đánh giá định tính.

5.  **Triển khai Ứng dụng Minh họa** (`app/app.py`)
    * Xây dựng giao diện người dùng web tương tác bằng thư viện **Streamlit**.
    * Cho phép người dùng tải lên (upload) ảnh MRI não.
    * Hiển thị kết quả phân loại (tên lớp dự đoán).
    * Cung cấp xác suất dự đoán cho từng lớp, thể hiện mức độ "tự tin" của mô hình.
    * Tích hợp **Grad-CAM** để trực quan hóa vùng ảnh mà mô hình tập trung chú ý khi đưa ra quyết định, tăng tính giải thích được cho mô hình.

---

## 🛠️ Cài đặt và Sử dụng

Để chạy ứng dụng demo và tái tạo môi trường, bạn cần thực hiện các bước sau:

1.  **Clone repository này về máy:**
    ```bash
    git clone [URL_REPOSITORY_CUA_BAN]
    cd [TEN_THU_MUC_REPOSITORY]
    ```

2.  **Cài đặt các thư viện cần thiết:**
    Tạo một môi trường ảo (khuyến khích) và cài đặt các gói từ file `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Các thư viện chính bao gồm: `tensorflow`, `streamlit`, `opencv-python`, `numpy`, `Pillow`, `matplotlib`, `scikit-learn`, `seaborn`.

3.  **Chạy ứng dụng demo:**
    Đảm bảo bạn đã có file mô hình `best_resnet_attention_model.h5`.
    ```bash
    streamlit run app.py
    ```
    Sau đó, mở trình duyệt và truy cập vào địa chỉ được Streamlit cung cấp (thường là `http://localhost:8501`).

---

## 🛠️ Thiết lập Môi trường Ảo và Cài đặt

Để đảm bảo tính nhất quán của các thư viện và tránh xung đột giữa các dự án, bạn nên tạo một môi trường ảo riêng cho dự án này. Dưới đây là hướng dẫn sử dụng `venv` (khuyến nghị nếu bạn dùng Python gốc) hoặc `conda` (nếu bạn dùng Anaconda/Miniconda).

### 1. Sử dụng `venv` (Python 3)

`venv` là một module có sẵn trong Python 3, cho phép bạn tạo các môi trường ảo nhẹ.

**a. Tạo môi trường ảo:**

Mở Terminal (macOS/Linux) hoặc Command Prompt/PowerShell (Windows), di chuyển đến thư mục gốc của dự án của bạn và chạy lệnh sau:

* Trên macOS/Linux:
    ```bash
    python3 -m venv venv_mri_tumor
    ```
* Trên Windows:
    ```bash
    python -m venv venv_mri_tumor
    ```
    *(Lệnh này sẽ tạo một thư mục có tên `venv_mri_tumor` chứa môi trường ảo của bạn.)*

**b. Kích hoạt môi trường ảo:**

Sau khi tạo, bạn cần kích hoạt môi trường ảo:

* Trên macOS/Linux (Bash/Zsh):
    ```bash
    source venv_mri_tumor/bin/activate
    ```
* Trên Windows (Command Prompt):
    ```bash
    venv_mri_tumor\Scripts\activate.bat
    ```
* Trên Windows (PowerShell):
    ```powershell
    .\venv_mri_tumor\Scripts\Activate.ps1
    ```
    *(Sau khi kích hoạt, bạn sẽ thấy tên môi trường ảo (ví dụ: `(venv_mri_tumor)`) xuất hiện ở đầu dòng lệnh của bạn.)*

**c. Cài đặt các thư viện cần thiết:**

Khi môi trường ảo đã được kích hoạt, hãy cài đặt các thư viện từ file `requirements.txt` (file này nên nằm trong thư mục gốc của dự án):

    ```bash
    pip install -r requirements.txt
    ```
**d. Hủy kích hoạt môi trường ảo:**
Khi bạn làm việc xong, bạn có thể hủy kích hoạt môi trường ảo bằng lệnh:

    ```bash
    deactivate
    ```
### 2. Sử dụng conda (Anaconda/Miniconda)
Nếu bạn đang sử dụng Anaconda hoặc Miniconda, bạn có thể tạo môi trường ảo bằng conda.

**a. Tạo môi trường ảo:**

Mở Anaconda Prompt (hoặc Terminal/Command Prompt đã cấu hình cho conda) và chạy lệnh sau. Bạn có thể chỉ định phiên bản Python khi tạo môi trường (ví dụ: python=3.9).
    ```bash
    conda create --name venv_mri_tumor python=3.9
    ```
**b. Kích hoạt môi trường ảo:**
    ```bash
    conda activate venv_mri_tumor
    ```
**c. Cài đặt các thư viện cần thiết:**

Khi môi trường ảo đã được kích hoạt, hãy cài đặt các thư viện từ file requirements.txt:
    ```bash
        pip install -r requirements.txt
    ```
**d. Hủy kích hoạt môi trường ảo:**

Khi bạn làm việc xong, bạn có thể hủy kích hoạt môi trường ảo bằng lệnh:

    ```bash
        conda deactivate
    ```


## 📊 Kết quả Nổi bật


* **Độ chính xác tổng thể trên tập kiểm tra (ví dụ: Bộ dữ liệu 3):** **96.3%**
* **Độ chính xác theo từng lớp (ví dụ: Bộ dữ liệu 3):**
    * Glioma: 100%
    * Meningioma: 95.24% 
    * No\_Tumor: 91.84%
    * Pituitary: 96.30%

---

## ℹ️ Thông tin về các loại khối u não

* **Glioma (U thần kinh đệm):**
    * Là khối u phát sinh từ tế bào hỗ trợ (tế bào đệm) trong não.
    * Có thể là lành tính hoặc ác tính.
    * Bao gồm các loại như u sao bào, u thần kinh đệm và u nguyên bào đệm.
* **Meningioma (U màng não):**
    * Phát triển từ màng não (màng bao quanh não và tủy sống).
    * Thường là lành tính và phát triển chậm.
    * Có thể gây áp lực lên não và các dây thần kinh.
* **Pituitary (U tuyến yên):**
    * Phát triển ở tuyến yên, một tuyến nhỏ nằm ở đáy não.
    * Thường là lành tính.
    * Có thể ảnh hưởng đến sự sản xuất hormone và gây ra các vấn đề về nội tiết.

---

## 📚 Tham khảo và Nguồn dữ liệu

* Các bộ dữ liệu được tổng hợp từ nhiều nguồn công khai trên internet (ví dụ: Kaggle, Figshare), đã trải qua quá trình tiền xử lý và cân bằng để phục vụ cho việc huấn luyện mô hình.
* Kiến trúc mô hình học sâu được xây dựng và huấn luyện dựa trên thư viện **TensorFlow** và **Keras API**.
* Thuật toán **Attention Mechanism** được tích hợp nhằm cải thiện hiệu suất và khả năng tập trung của mô hình phân loại.

---

## ⚠️ Lưu ý Quan trọng

Kết quả phân loại từ mô hình này được cung cấp với mục đích nghiên cứu và tham khảo. **Nó không thể và không nên thay thế cho việc chẩn đoán y tế chuyên nghiệp** từ các bác sĩ có chuyên môn.

---

## 💻 Môi trường Huấn luyện

Mô hình được huấn luyện chủ yếu trên nền tảng **Google Colaboratory** với sự hỗ trợ của GPU.

