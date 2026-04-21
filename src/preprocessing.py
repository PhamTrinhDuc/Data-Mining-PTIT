"""
File 3: Tiền xử lý & Chuẩn hóa
==================================
[DM Note] Tiền xử lý trong DM cần giải thích lý do của từng quyết định.
          Không chỉ "encode rồi scale" — mà phải ghi lại tại sao chọn
          cách encode đó, tại sao scale bằng method đó.
          
          Đây là phần tạo ra "clean data" để đưa vào cả:
            - Clustering (File 4)  — tìm nhóm khách hàng
            - Classification (File 5) — dự đoán churn
"""
import os 
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Load data gốc
# ─────────────────────────────────────────────
PROJET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(PROJET_DIR, 'data/data_raw.csv')
df = pd.read_csv(DATA_RAW_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print("=" * 60)
print("TIỀN XỬ LÝ & CHUẨN HÓA")
print("=" * 60)
print(f"\nInput: {df.shape[0]:,} dòng × {df.shape[1]} cột")

# ─────────────────────────────────────────────
# BƯỚC 1: Loại bỏ cột không cần thiết
# ─────────────────────────────────────────────
print("\n── Bước 1: Loại bỏ cột không cần thiết ──")

# [DM] customerID là định danh, không mang thông tin dự đoán
# Giữ lại để tra cứu nhưng không đưa vào model
customer_ids = df['customerID'].copy()
df = df.drop(columns=['customerID'])
print(f"  Đã loại: customerID (chỉ là định danh, không có ý nghĩa dự đoán)")

# ─────────────────────────────────────────────
# BƯỚC 2: Xử lý Missing Values
# ─────────────────────────────────────────────
print("\n── Bước 2: Xử lý Missing Values ──")

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]
print(f"  Số cột bị missing: {len(missing_before)}")

for col, cnt in missing_before.items():
    print(f"  • {col}: {cnt} giá trị missing")

# [DM] Chiến lược điền missing:
# TotalCharges bị missing vì tenure=0 (khách hàng mới chưa phát sinh phí)
# → Điền bằng 0 hợp lý hơn mean/median vì đây là missing có nghĩa kinh doanh
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(f"\n  Chiến lược: TotalCharges=NaN → điền 0")
print(f"  Lý do: tenure=0 nghĩa là khách hàng mới, chưa có tổng phí")
print(f"  (Không dùng mean/median vì missing KHÔNG phải ngẫu nhiên)")

print(f"\n  Missing sau xử lý: {df.isnull().sum().sum()} giá trị")

# ─────────────────────────────────────────────
# BƯỚC 3: Encode Target (Churn)
# ─────────────────────────────────────────────
print("\n── Bước 3: Encode Target ──")

# [DM] Target phải encode trước để tính correlation, kiểm tra imbalance
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print(f"  Churn: 'Yes' → 1, 'No' → 0")
print(f"  Phân phối: {df['Churn'].value_counts().to_dict()}")
print(f"  Tỉ lệ churn: {df['Churn'].mean()*100:.1f}%")

# ─────────────────────────────────────────────
# BƯỚC 4: Encode Categorical Features
# ─────────────────────────────────────────────
print("\n── Bước 4: Encode Categorical Features ──")

# --- 4a. Binary encoding (Yes/No) ---
# [DM] Với cột chỉ có 2 giá trị Yes/No → encode thành 0/1
# Đơn giản, không tạo thêm cột, giữ tính tuyến tính
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    unique_vals = df[col].unique()
    if set(unique_vals).issubset({'Male', 'Female'}):
        df[col] = (df[col] == 'Male').astype(int)
        print(f"  {col}: Male=1, Female=0")
    else:
        df[col] = (df[col] == 'Yes').astype(int)
        print(f"  {col}: Yes=1, No=0")

# --- 4b. Ordinal encoding (có thứ tự) ---
# [DM] Contract có thứ tự ngầm: month-to-month < 1 year < 2 year
# → Encode theo thứ tự cam kết tăng dần — giữ được quan hệ tuyến tính
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['Contract'] = df['Contract'].map(contract_map)
print(f"\n  Contract (ordinal): Month-to-month=0, One year=1, Two year=2")
print(f"  Lý do: có thứ tự tự nhiên về mức độ cam kết")

# --- 4c. One-hot encoding (không có thứ tự) ---
# [DM] InternetService và PaymentMethod không có thứ tự rõ ràng
# → One-hot để tránh model hiểu nhầm có quan hệ số học giữa các category
onehot_cols = ['InternetService', 'PaymentMethod']
df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols, drop_first=False)
# Chuyển boolean sang int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
print(f"\n  One-hot encoding: {onehot_cols}")
print(f"  Lý do: không có thứ tự tự nhiên giữa các loại dịch vụ/thanh toán")

# --- 4d. Encode "No internet service" / "No phone service" ---
# [DM] Nhiều cột dịch vụ có giá trị "No internet service" = thực chất là "No"
# → Gộp lại thành binary để đơn giản hóa
service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in service_cols:
    # "No internet service" / "No phone service" → No → 0
    df[col] = df[col].replace({'No internet service': 'No',
                               'No phone service': 'No'})
    df[col] = (df[col] == 'Yes').astype(int)
    print(f"  {col}: Yes=1, No/No service=0")

# ─────────────────────────────────────────────
# BƯỚC 5: Kiểm tra sau encoding
# ─────────────────────────────────────────────
print("\n── Bước 5: Kiểm tra sau encoding ──")
print(f"  Shape sau encoding: {df.shape}")
print(f"  Còn object dtype: {df.select_dtypes(include='object').columns.tolist()}")

remaining_obj = df.select_dtypes(include='object').columns.tolist()
if remaining_obj:
    print(f"  ⚠️ Còn cột chưa encode: {remaining_obj}")
else:
    print(f"  ✅ Tất cả cột đã được encode sang numeric")

# ─────────────────────────────────────────────
# BƯỚC 6: Tách features và target
# ─────────────────────────────────────────────
print("\n── Bước 6: Tách X và y ──")

y = df['Churn']
X = df.drop(columns=['Churn'])

print(f"  X (features): {X.shape}  — {list(X.columns)}")
print(f"  y (target):   {y.shape}  — phân phối: {y.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# BƯỚC 7: Train/Test Split
# ─────────────────────────────────────────────
print("\n── Bước 7: Train/Test Split ──")

# [DM] stratify=y đảm bảo tỉ lệ churn trong train và test giống nhau
# → Quan trọng vì dataset mất cân bằng (imbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # reproducibility
    stratify=y          # giữ tỉ lệ churn
)

print(f"  Train: {X_train.shape[0]:,} mẫu — churn rate: {y_train.mean()*100:.1f}%")
print(f"  Test:  {X_test.shape[0]:,} mẫu  — churn rate: {y_test.mean()*100:.1f}%")
print(f"  → stratify=y đảm bảo tỉ lệ churn nhất quán giữa train và test ✅")

# ─────────────────────────────────────────────
# BƯỚC 8: Chuẩn hóa (Scaling)
# ─────────────────────────────────────────────
print("\n── Bước 8: Chuẩn hóa (StandardScaler) ──")

# [DM] Tại sao cần scale?
# tenure: 0-72 tháng | MonthlyCharges: 18-118 | TotalCharges: 0-8000+
# → Các features có đơn vị khác nhau hoàn toàn
# → Model như Logistic Regression, KMeans bị ảnh hưởng bởi scale
# → StandardScaler: mean=0, std=1 (phù hợp với distribution bình thường)

# [DM QUAN TRỌNG] Chỉ fit scaler trên TRAIN, transform cả train và test
# → Nếu fit trên cả dataset = data leakage (dùng thông tin của test khi train)
scaler = StandardScaler()

# Các cột cần scale (numeric liên tục, không phải binary)
scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])   # fit + transform
X_test_scaled[scale_cols]  = scaler.transform(X_test[scale_cols])         # chỉ transform

print(f"  Scale columns: {scale_cols}")
print(f"  Method: StandardScaler (z-score normalization)")
print(f"  ⚠️  fit_transform chỉ trên TRAIN → tránh data leakage")
print(f"\n  Trước scaling — tenure (train):")
print(f"    mean={X_train['tenure'].mean():.1f}, std={X_train['tenure'].std():.1f}")
print(f"  Sau scaling — tenure (train):")
print(f"    mean={X_train_scaled['tenure'].mean():.4f}, std={X_train_scaled['tenure'].std():.4f}")

# [DM] Tạo thêm bản X_cluster dùng cho clustering (File 4)
# Clustering cần scale TẤT CẢ features (kể cả binary) để KMeans không bị bias
scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(X)  # Dùng toàn bộ data cho clustering
print(f"\n  X_cluster (dùng cho KMeans): fit_transform toàn bộ X ({X.shape})")
print(f"  Lý do: KMeans dùng Euclidean distance — mọi feature cần cùng scale")

# ─────────────────────────────────────────────
# BƯỚC 9: Lưu tất cả artifacts
# ─────────────────────────────────────────────
print("\n── Bước 9: Lưu artifacts ──")


PREPROCESSED_DIR = os.path.join(PROJET_DIR, 'data/processed')
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
# Lưu data đã xử lý
X_train_scaled.to_csv(os.path.join(PREPROCESSED_DIR, "X_train.csv"), index=False)
X_test_scaled.to_csv(os.path.join(PREPROCESSED_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PREPROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PREPROCESSED_DIR, "y_test.csv"), index=False)
np.save(os.path.join(PREPROCESSED_DIR, "X_cluster.npy"), X_cluster)
X.to_csv(os.path.join(PREPROCESSED_DIR, "X_full.csv"), index=False)
y.to_csv(os.path.join(PREPROCESSED_DIR, "y_full.csv"), index=False)

# Lưu scaler để dùng ở bước retrain (File 7)
with open(os.path.join(PREPROCESSED_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(PREPROCESSED_DIR, "scaler_cluster.pkl"), 'wb') as f:
    pickle.dump(scaler_cluster, f)

# Lưu tên features để trace kết quả
feature_names = list(X.columns)
with open(os.path.join(PREPROCESSED_DIR, "feature_names.pkl"), 'wb') as f:
    pickle.dump(feature_names, f)

print(f"  ✅ X_train.csv    — {X_train_scaled.shape}")
print(f"  ✅ X_test.csv     — {X_test_scaled.shape}")
print(f"  ✅ y_train.csv    — {y_train.shape}")
print(f"  ✅ y_test.csv     — {y_test.shape}")
print(f"  ✅ X_cluster.npy  — {X_cluster.shape}  (dùng cho KMeans)")
print(f"  ✅ X_full.csv     — {X.shape}           (dùng cho clustering)")
print(f"  ✅ scaler.pkl     — StandardScaler (dùng cho retrain pipeline)")
print(f"  ✅ feature_names.pkl — {len(feature_names)} features")

# ─────────────────────────────────────────────
# TỔNG KẾT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TỔNG KẾT TIỀN XỬ LÝ")
print("=" * 60)
print(f"""
  Input:  {df.shape[0]:,} dòng, {df.shape[1]+1} cột (gồm target)
  Output: {X_train_scaled.shape[0]:,} train / {X_test_scaled.shape[0]:,} test
          {X_train_scaled.shape[1]} features sau encoding

  Quyết định quan trọng:
  • TotalCharges=NaN → 0  (không phải ngẫu nhiên, là khách hàng mới)
  • Contract → ordinal    (có thứ tự cam kết tự nhiên)
  • InternetService → one-hot (không có thứ tự)
  • StandardScaler fit chỉ trên train (tránh data leakage)
  • stratify=y khi split (vì dataset mất cân bằng)

  → File 4: Clustering — phân nhóm khách hàng (DM core)
""")