"""
File 1: Dataset — Load & Mô tả dữ liệu
========================================
Dataset: Telco Customer Churn (IBM Sample)
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Target: Cột 'Churn' — khách hàng có rời bỏ dịch vụ không (Yes/No)

[DM Note] Bước này không chỉ "load data" như ML thông thường.
          Mục tiêu là hiểu SỐ lượng, CHẤT lượng và CẤU TRÚC của dataset
          trước khi bắt tay phân tích — đây là bước đặt câu hỏi ban đầu.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────
# Nếu chưa có file, download từ Kaggle hoặc dùng URL trực tiếp
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(URL)

print("=" * 60)
print("TỔNG QUAN DATASET")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. Kích thước và cấu trúc
# ─────────────────────────────────────────────
print(f"\n📦 Kích thước: {df.shape[0]:,} dòng × {df.shape[1]} cột")
print(f"   → Mỗi dòng = 1 khách hàng")

# [DM] Hiểu dtype giúp biết feature nào cần encode, normalize
print("\n📋 Kiểu dữ liệu từng cột:")
print(df.dtypes.to_string())

# ─────────────────────────────────────────────
# 3. Xem mẫu dữ liệu
# ─────────────────────────────────────────────
print("\n📄 5 dòng đầu tiên:")
print(df.head().to_string())

# ─────────────────────────────────────────────
# 4. Phân loại features theo nhóm
# ─────────────────────────────────────────────
# [DM] Phân nhóm feature là bước "hiểu dữ liệu" — không có trong ML thuần
print("\n" + "=" * 60)
print("PHÂN LOẠI FEATURES")
print("=" * 60)

customer_info = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']
account_info  = ['tenure', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                 'MonthlyCharges', 'TotalCharges']
services      = ['PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
target        = ['Churn']

print(f"\n👤 Thông tin khách hàng ({len(customer_info)} features): {customer_info}")
print(f"💳 Thông tin tài khoản  ({len(account_info)} features): {account_info}")
print(f"📡 Dịch vụ đăng ký      ({len(services)} features): {services}")
print(f"🎯 Target               ({len(target)} feature): {target}")

# ─────────────────────────────────────────────
# 5. Thống kê mô tả — numeric features
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("THỐNG KÊ MÔ TẢ — NUMERIC FEATURES")
print("=" * 60)

# [DM] Nhìn vào min/max/std để phát hiện outlier sớm
# TotalCharges có thể bị lưu dưới dạng string (lỗi thường gặp của dataset này)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print(df[numeric_cols].describe().round(2).to_string())

# ─────────────────────────────────────────────
# 6. Phân phối target — câu hỏi quan trọng nhất
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHÂN PHỐI TARGET (CHURN)")
print("=" * 60)

churn_counts = df['Churn'].value_counts()
churn_pct    = df['Churn'].value_counts(normalize=True) * 100

print(f"\n  Không rời bỏ (No):  {churn_counts['No']:,} ({churn_pct['No']:.1f}%)")
print(f"  Có rời bỏ   (Yes): {churn_counts['Yes']:,} ({churn_pct['Yes']:.1f}%)")

# [DM] Imbalanced dataset là phát hiện quan trọng!
# Tỉ lệ ~73/27 → sẽ ảnh hưởng đến chọn metric và model
if churn_pct['Yes'] < 35:
    print("\n  ⚠️  [DM INSIGHT] Dataset bị mất cân bằng (imbalanced)!")
    print("     → Dùng F1-score + ROC-AUC thay vì chỉ accuracy")
    print("     → Cân nhắc class_weight='balanced' khi train model")

# ─────────────────────────────────────────────
# 7. Tổng kết & lưu
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TỔNG KẾT BƯỚC 1")
print("=" * 60)
print(f"  • {df.shape[0]:,} khách hàng, {df.shape[1]} features")
print(f"  • {len(numeric_cols)} numeric · {df.shape[1] - len(numeric_cols) - 1} categorical · 1 target")
print(f"  • Tỉ lệ churn: {churn_pct['Yes']:.1f}% — dataset mất cân bằng")
print(f"  • TotalCharges: cần ép kiểu numeric (có giá trị rỗng)")
print("\n  → Chuyển sang File 2: EDA\n")

# Lưu ra file để các bước sau dùng
df.to_csv("data_raw.csv", index=False)
print("  ✅ Đã lưu: data_raw.csv")