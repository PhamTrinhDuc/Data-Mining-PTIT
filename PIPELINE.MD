# Báo cáo cuối kì — Data Mining: Customer Churn Prediction

## Thông tin chung

- **Dataset:** Telco Customer Churn (IBM/Kaggle)
- **Bài toán:** Phân loại khách hàng rời bỏ dịch vụ (Churn Prediction)
- **Công cụ:** Python, scikit-learn, pandas, matplotlib, seaborn

---

## Cấu trúc báo cáo

### Chương 1 — Giới thiệu *(~2 trang)*

- 1.1 Đặt vấn đề: tại sao churn prediction quan trọng
- 1.2 Mục tiêu của đề tài
- 1.3 Phạm vi và giới hạn
- 1.4 Cấu trúc báo cáo

### Chương 2 — Cơ sở lý thuyết *(~4 trang)*

- 2.1 Tổng quan về Data Mining
  - Định nghĩa, phân biệt DM vs ML vs Data Science
  - Quy trình CRISP-DM
- 2.2 Các kỹ thuật sử dụng trong bài
  - Clustering (KMeans, Elbow method, Silhouette score)
  - Classification (Random Forest, feature importance)
  - Imbalanced dataset và các chiến lược xử lý
- 2.3 Các độ đo đánh giá
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC — tại sao không chỉ dùng accuracy

### Chương 3 — Dataset & EDA *(~5 trang)*

- 3.1 Mô tả dataset
  - Nguồn gốc, kích thước, các features
  - Phân loại feature theo nhóm (thông tin KH, tài khoản, dịch vụ)
- 3.2 Phân tích chất lượng dữ liệu
  - Missing values, duplicate, kiểu dữ liệu
- 3.3 Phân tích đơn biến (Univariate)
  - Phân phối target — imbalanced dataset
  - Phân phối các numeric features (tenure, MonthlyCharges, TotalCharges)
- 3.4 Phân tích đa biến (Multivariate)
  - Churn rate theo từng categorical feature
  - Correlation heatmap
  - Phân tích tenure segments — "vùng nguy hiểm"
- 3.5 Tổng hợp insights từ EDA

### Chương 4 — Tiền xử lý dữ liệu *(~3 trang)*

- 4.1 Xử lý missing values và lý do lựa chọn chiến lược
- 4.2 Encoding categorical features
  - Binary encoding (Yes/No)
  - Ordinal encoding (Contract — có thứ tự)
  - One-hot encoding (InternetService, PaymentMethod — không có thứ tự)
- 4.3 Chuẩn hóa (StandardScaler) — tại sao fit chỉ trên train
- 4.4 Train/test split — tại sao dùng stratify

### Chương 5 — Clustering: Phân nhóm khách hàng *(~4 trang)*

> Đây là phần cốt lõi DM — khác biệt với bài ML thông thường

- 5.1 Mục tiêu và ý nghĩa của clustering trong bài toán này
- 5.2 Chọn số cluster tối ưu (Elbow + Silhouette)
- 5.3 Kết quả phân nhóm — profile từng cluster
- 5.4 Phân tích churn rate theo cluster
- 5.5 Đặt tên và giải thích đặc trưng từng nhóm khách hàng
- 5.6 Ý nghĩa kinh doanh của kết quả clustering

### Chương 6 — Classification: Dự đoán churn *(~4 trang)*

- 6.1 Lựa chọn model — lý do chọn Random Forest
- 6.2 Các tham số model và giải thích
- 6.3 Kết quả đánh giá (F1, ROC-AUC, Confusion Matrix)
- 6.4 Phân tích Feature Importance
  - Top features quan trọng nhất
  - Giải thích ý nghĩa kinh doanh của từng feature
- 6.5 Phân tích Risk Level — nhóm nguy cơ cao/trung bình/thấp
- 6.6 So sánh: nếu chỉ dùng accuracy thì sẽ sai ở đâu

### Chương 7 — Simulate & Retrain Pipeline *(~4 trang)*

- 7.1 Đặt vấn đề: tại sao cần retrain định kỳ
- 7.2 Chiến lược simulate dữ liệu mới theo tháng
  - Cách sinh data phản ánh pattern từ EDA
  - Data drift theo tháng (churn rate tăng dần)
- 7.3 Retrain pipeline — chiến lược incremental
- 7.4 Kết quả: so sánh F1/AUC qua các tháng
- 7.5 Model selection — tiêu chí chọn model cuối
- 7.6 So sánh Feature Importance trước/sau retrain

### Chương 8 — Kết luận *(~2 trang)*

- 8.1 Tổng hợp DM insights từ toàn bộ pipeline
- 8.2 Gợi ý kinh doanh (actionable recommendations)
- 8.3 Hạn chế của đề tài
- 8.4 Hướng phát triển

### Tài liệu tham khảo

---

## Phân công nhiệm vụ (5 người — deadline 1-2 tuần)

> Mỗi người phụ trách phần code + viết nội dung tương ứng

### Người 1 — Giới thiệu + Cơ sở lý thuyết + Dataset

Phụ trách: Chương 1, 2, 3 (phần 3.1–3.2)

Công việc cụ thể:
- Viết chương 1 (đặt vấn đề, mục tiêu)
- Viết chương 2 (lý thuyết DM, CRISP-DM, các kỹ thuật dùng trong bài)
- Chạy file `01_dataset.py`, viết mô tả dataset cho báo cáo
- Tổng hợp tài liệu tham khảo

Ghi chú: phần lý thuyết cần nhấn mạnh DM khác ML ở chỗ nào — cô sẽ hỏi điều này.

---

### Người 2 — EDA

Phụ trách: Chương 3 (phần 3.3–3.5)

Công việc cụ thể:
- Chạy file `02_eda.py`, đọc hiểu toàn bộ kết quả
- Viết phân tích cho từng biểu đồ EDA (numeric, categorical, correlation, tenure)
- Tổng hợp 5 insights chính từ EDA thành văn phong báo cáo
- Chuẩn bị hình ảnh đẹp cho báo cáo (xuất từ code)

Ghi chú: đây là phần dài và quan trọng — mỗi biểu đồ cần có 2–3 câu giải thích insight, không chỉ mô tả hình.

---

### Người 3 — Tiền xử lý + Clustering

Phụ trách: Chương 4, 5

Công việc cụ thể:
- Chạy file `03_preprocessing.py` và `04_clustering.py`
- Viết chương 4: giải thích từng quyết định encode/scale
- Viết chương 5: phân tích elbow/silhouette, profile cluster, đặt tên cluster
- Vẽ lại hoặc chỉnh sửa biểu đồ cluster cho đẹp (nếu cần)

Ghi chú: clustering là phần cốt lõi DM của bài — viết kỹ phần "profile và đặt tên cluster", đây là chỗ cô sẽ chú ý nhất.

---

### Người 4 — Classification + Feature Importance

Phụ trách: Chương 6

Công việc cụ thể:
- Chạy file `05_classification.py`
- Viết chương 6: giải thích lý do chọn Random Forest, các tham số
- Phân tích kết quả Confusion Matrix — loại lỗi nào nguy hiểm hơn
- Viết phần Feature Importance với giải thích ý nghĩa kinh doanh
- Viết phần Risk Level Analysis

Ghi chú: kết quả phải trình bày theo góc nhìn DM — không chỉ "F1 = 0.85" mà phải "feature X quan trọng vì... → gợi ý kinh doanh là..."

---

### Người 5 — Simulate + Retrain + Kết luận

Phụ trách: Chương 7, 8

Công việc cụ thể:
- Chạy file `06_simulate.py` và `07_retrain.py`
- Viết chương 7: giải thích chiến lược simulate, phân tích data drift, kết quả retrain
- Viết chương 8: tổng hợp insights, gợi ý kinh doanh, hạn chế và hướng phát triển
- Tổng hợp toàn bộ báo cáo, kiểm tra format thống nhất

Ghi chú: phần retrain cần nêu rõ tại sao pipeline này khác với "train 1 lần rồi thôi" — đây là điểm DM mạnh nhất của bài.

---

## Gợi ý timeline (2 tuần)

| Tuần | Việc cần làm |
|---|---|
| Tuần 1 — ngày 1–3 | Mỗi người chạy code phần mình, đọc hiểu output |
| Tuần 1 — ngày 4–7 | Viết nháp nội dung chương phụ trách |
| Tuần 2 — ngày 1–3 | Review chéo nội dung, bổ sung phần thiếu |
| Tuần 2 — ngày 4–5 | Người 5 tổng hợp, chỉnh format, xuất file cuối |
| Tuần 2 — ngày 6–7 | Buffer: sửa theo feedback nhóm, chuẩn bị thuyết trình |

---

