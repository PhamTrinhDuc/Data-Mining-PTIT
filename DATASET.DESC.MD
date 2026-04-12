# 📊 Dataset Telco Customer Churn

## 1. Giới thiệu
Dataset **Telco Customer Churn** là một bộ dữ liệu mẫu của IBM, thường được sử dụng để nghiên cứu và thực hành các mô hình dự đoán khách hàng rời bỏ dịch vụ (churn). Đây là một trong những bộ dữ liệu phổ biến trong lĩnh vực *customer analytics* và *machine learning*.

- **Số lượng bản ghi:** 7,043 khách hàng  
- **Số lượng thuộc tính:** 21 cột (features)  
- **Biến mục tiêu:** `Churn` (Yes/No) – cho biết khách hàng có rời bỏ dịch vụ trong tháng vừa qua hay không  

---

## 2. Các nhóm dữ liệu

### 🧑‍💻 Thông tin nhân khẩu học
| Trường | Kiểu dữ liệu | Ý nghĩa |
|--------|--------------|---------|
| `customerID` | String | Mã định danh khách hàng |
| `gender` | Categorical (Male/Female) | Giới tính |
| `SeniorCitizen` | Integer (0/1) | Khách hàng là người cao tuổi (1) hay không (0) |
| `Partner` | Yes/No | Có vợ/chồng/đối tác |
| `Dependents` | Yes/No | Có người phụ thuộc (con cái, người thân) |

### 📡 Thông tin dịch vụ
| Trường | Kiểu dữ liệu | Ý nghĩa |
|--------|--------------|---------|
| `PhoneService` | Yes/No | Có sử dụng dịch vụ điện thoại |
| `MultipleLines` | Yes/No/No phone service | Có nhiều đường dây điện thoại |
| `InternetService` | DSL/Fiber optic/No | Loại dịch vụ Internet |
| `OnlineSecurity` | Yes/No/No internet service | Dịch vụ bảo mật trực tuyến |
| `OnlineBackup` | Yes/No/No internet service | Dịch vụ sao lưu trực tuyến |
| `DeviceProtection` | Yes/No/No internet service | Dịch vụ bảo vệ thiết bị |
| `TechSupport` | Yes/No/No internet service | Hỗ trợ kỹ thuật |
| `StreamingTV` | Yes/No/No internet service | Dịch vụ TV trực tuyến |
| `StreamingMovies` | Yes/No/No internet service | Dịch vụ phim trực tuyến |

### 💳 Thông tin tài khoản
| Trường | Kiểu dữ liệu | Ý nghĩa |
|--------|--------------|---------|
| `tenure` | Integer | Số tháng gắn bó với công ty |
| `Contract` | Month-to-month/One year/Two year | Loại hợp đồng |
| `PaperlessBilling` | Yes/No | Có sử dụng hóa đơn điện tử |
| `PaymentMethod` | Categorical | Phương thức thanh toán (Electronic check, Mailed check, Bank transfer, Credit card) |
| `MonthlyCharges` | Float | Chi phí hàng tháng |
| `TotalCharges` | Float | Tổng chi phí đã trả |

### 🎯 Biến mục tiêu
| Trường | Kiểu dữ liệu | Ý nghĩa |
|--------|--------------|---------|
| `Churn` | Yes/No | Khách hàng có rời bỏ dịch vụ hay không |

---

## 3. Mục đích sử dụng
- Xây dựng mô hình dự đoán churn (Logistic Regression, Random Forest, XGBoost, v.v.).  
- Phân tích yếu tố ảnh hưởng đến việc khách hàng rời bỏ dịch vụ.  
- Thực hành các kỹ thuật tiền xử lý dữ liệu, phân tích thống kê, và trực quan hóa.  

---

## 4. Ý nghĩa
Bộ dữ liệu này giúp doanh nghiệp viễn thông (telco) hiểu rõ hơn về hành vi khách hàng, từ đó xây dựng các chương trình giữ chân khách hàng hiệu quả. Nó cũng là một bộ dữ liệu chuẩn để học tập và thử nghiệm các thuật toán học máy trong lĩnh vực phân tích khách hàng.

