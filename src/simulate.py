"""
File 6: Simulate Collect Data — Giả lập thu thập dữ liệu mới
==============================================================
[DM Note] Đây là phần "Mining" tự động trong pipeline.
          Trong thực tế: hệ thống sẽ liên tục thu thập data mới
          từ các nguồn (CRM, log hệ thống, billing...) mỗi tháng.

          Ở đây ta SIMULATE bằng cách:
            1. Sinh khách hàng mới mỗi "tháng" với phân phối
               giống data gốc (có nhiễu thực tế)
            2. Lưu từng batch vào file riêng (giống collect thật)
            3. File 7 sẽ đọc các batch này và retrain model

          Mục tiêu: chứng minh pipeline có thể TỰ ĐỘNG CẬP NHẬT
          khi có data mới — không phải train 1 lần rồi thôi.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Load data gốc để học phân phối
# ─────────────────────────────────────────────
df_raw = pd.read_csv('data_raw.csv')
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce').fillna(0)

print("=" * 60)
print("SIMULATE COLLECT DATA — GIẢ LẬP THU THẬP DỮ LIỆU MỚI")
print("=" * 60)
print(f"\nData gốc: {df_raw.shape[0]:,} khách hàng")
print("[DM] Simulate = học phân phối data gốc → sinh data mới có nhiễu")

# Tạo thư mục lưu batch data
os.makedirs('monthly_batches', exist_ok=True)

# ─────────────────────────────────────────────
# Hàm sinh khách hàng mới
# ─────────────────────────────────────────────

def simulate_new_customers(n: int, month: int, seed: int = None) -> pd.DataFrame:
    """
    Sinh n khách hàng mới cho tháng `month`.
    
    [DM] Chiến lược simulate thực tế:
    - Categorical features: lấy mẫu theo tỉ lệ thực từ data gốc
    - Numeric features: fit normal distribution từ data gốc + thêm nhiễu
    - Churn label: dùng luật đơn giản phản ánh pattern đã biết từ EDA
    - Theo tháng: churn rate tăng dần (giả lập market xấu đi)
    """
    if seed is not None:
        np.random.seed(seed + month * 100)

    n_customers = n
    df_new = pd.DataFrame()

    # ── Categorical features — lấy mẫu theo tỉ lệ thực ──
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']

    for col in cat_cols:
        # Tính tỉ lệ thực từ data gốc
        probs = df_raw[col].value_counts(normalize=True)
        df_new[col] = np.random.choice(
            probs.index.tolist(),
            size=n_customers,
            p=probs.values
        )

    # ── Numeric features — normal distribution từ data gốc ──

    # tenure: khách hàng mới có tenure thấp hơn (0-24 tháng thay vì 0-72)
    # [DM] Đây là điểm "thực tế" — khách hàng mới join tenure thấp
    tenure_mean = min(df_raw['tenure'].mean(), 18)  # khách mới tenure thấp hơn
    tenure_std  = df_raw['tenure'].std() * 0.6
    df_new['tenure'] = np.clip(
        np.random.normal(tenure_mean, tenure_std, n_customers),
        0, 72
    ).astype(int)

    # MonthlyCharges: giả lập giá tăng nhẹ theo tháng (inflation)
    mc_mean = df_raw['MonthlyCharges'].mean() * (1 + month * 0.005)
    mc_std  = df_raw['MonthlyCharges'].std()
    df_new['MonthlyCharges'] = np.clip(
        np.random.normal(mc_mean, mc_std, n_customers),
        18, 120
    ).round(2)

    # TotalCharges = tenure × MonthlyCharges (có nhiễu nhỏ)
    df_new['TotalCharges'] = (
        df_new['tenure'] * df_new['MonthlyCharges'] *
        np.random.uniform(0.95, 1.05, n_customers)
    ).round(2)

    # SeniorCitizen: binary, lấy tỉ lệ từ data gốc
    senior_rate = df_raw['SeniorCitizen'].mean()
    df_new['SeniorCitizen'] = np.random.binomial(1, senior_rate, n_customers)

    # customerID mới
    df_new['customerID'] = [f"NEW-M{month:02d}-{i:04d}" for i in range(n_customers)]

    # ── Simulate Churn label ──
    # [DM] Dùng luật dựa trên pattern từ EDA (không phải random)
    # → Churn xảy ra khi: tenure thấp + hợp đồng ngắn + giá cao
    churn_prob = np.zeros(n_customers)

    # Tenure ngắn → churn cao
    churn_prob += np.where(df_new['tenure'] < 6,  0.35, 0)
    churn_prob += np.where(df_new['tenure'] < 12, 0.15, 0)
    churn_prob += np.where(df_new['tenure'] > 36, -0.15, 0)

    # Hợp đồng month-to-month → churn cao
    churn_prob += np.where(df_new['Contract'] == 'Month-to-month', 0.25, 0)
    churn_prob += np.where(df_new['Contract'] == 'Two year', -0.20, 0)

    # Phí cao → churn cao (không thấy giá trị)
    high_charge = df_new['MonthlyCharges'] > df_raw['MonthlyCharges'].quantile(0.75)
    churn_prob += np.where(high_charge, 0.15, 0)

    # Không có TechSupport/OnlineSecurity → churn cao
    churn_prob += np.where(df_new['TechSupport'] == 'No', 0.10, 0)
    churn_prob += np.where(df_new['OnlineSecurity'] == 'No', 0.10, 0)

    # Fiber optic → churn cao hơn DSL (theo EDA)
    churn_prob += np.where(df_new['InternetService'] == 'Fiber optic', 0.10, 0)

    # [DM] Thêm drift theo tháng: churn rate tăng dần (market xấu dần)
    # → Đây là lý do cần RETRAIN — model cũ không biết trend mới
    monthly_drift = month * 0.015
    churn_prob += monthly_drift

    # Clip về [0.05, 0.85] và thêm nhiễu Gaussian
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn_prob += np.random.normal(0, 0.05, n_customers)
    churn_prob = np.clip(churn_prob, 0, 1)

    # Sinh nhãn từ xác suất
    df_new['Churn'] = np.where(
        np.random.uniform(0, 1, n_customers) < churn_prob,
        'Yes', 'No'
    )

    return df_new

# ─────────────────────────────────────────────
# Simulate 6 tháng
# ─────────────────────────────────────────────
print("\n── Simulate 6 tháng dữ liệu mới ──")
print(f"\n  {'Tháng':<10} {'Khách mới':>10} {'Churn rate':>12} {'File'}")
print("  " + "-" * 55)

N_PER_MONTH  = 200    # ~200 khách hàng mới mỗi tháng
N_MONTHS     = 6
batch_summary = []

for month in range(1, N_MONTHS + 1):
    df_batch = simulate_new_customers(n=N_PER_MONTH, month=month, seed=42)

    churn_rate = (df_batch['Churn'] == 'Yes').mean() * 100
    fname      = f'monthly_batches/month_{month:02d}.csv'

    df_batch.to_csv(fname, index=False)

    batch_summary.append({
        'month': month,
        'n_customers': len(df_batch),
        'churn_rate': churn_rate,
        'file': fname
    })

    print(f"  Tháng {month:<5} {len(df_batch):>10,} {churn_rate:>11.1f}%   {fname}")

# ─────────────────────────────────────────────
# Kiểm tra drift — so sánh với data gốc
# ─────────────────────────────────────────────
print("\n── Kiểm tra Data Drift theo tháng ──")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Simulate Data — Phân tích Data Drift theo tháng', fontsize=14)

# Churn rate theo tháng
ax = axes[0, 0]
months     = [s['month'] for s in batch_summary]
churn_rates= [s['churn_rate'] for s in batch_summary]
orig_churn = (df_raw['Churn'] == 'Yes').mean() * 100

ax.plot(months, churn_rates, 'ro-', linewidth=2, markersize=8, label='Simulated monthly')
ax.axhline(y=orig_churn, color='blue', linestyle='--', label=f'Original ({orig_churn:.1f}%)')
ax.set_xlabel('Tháng')
ax.set_ylabel('Churn rate (%)')
ax.set_title('Churn Rate Drift theo tháng')
ax.legend()
ax.grid(True, alpha=0.3)

# [DM] Data drift là lý do phải retrain!
# Nếu churn rate tăng dần → model cũ sẽ underestimate nguy cơ

# MonthlyCharges distribution: tháng 1 vs tháng 6
ax = axes[0, 1]
df_m1 = pd.read_csv('monthly_batches/month_01.csv')
df_m6 = pd.read_csv('monthly_batches/month_06.csv')
ax.hist(df_m1['MonthlyCharges'], bins=30, alpha=0.6, label='Tháng 1', color='#4C8EDA')
ax.hist(df_m6['MonthlyCharges'], bins=30, alpha=0.6, label='Tháng 6', color='#E05C5C')
ax.set_xlabel('Monthly Charges ($)')
ax.set_ylabel('Count')
ax.set_title('MonthlyCharges: Tháng 1 vs Tháng 6')
ax.legend()

# Tenure distribution so với gốc
ax = axes[1, 0]
all_new = pd.concat([pd.read_csv(f"monthly_batches/month_{m:02d}.csv")
                     for m in range(1, N_MONTHS+1)])
ax.hist(df_raw['tenure'], bins=30, alpha=0.5, label='Data gốc', color='gray', density=True)
ax.hist(all_new['tenure'], bins=30, alpha=0.6, label='Data mới', color='#2ECC71', density=True)
ax.set_xlabel('Tenure (tháng)')
ax.set_ylabel('Density')
ax.set_title('Tenure: Data gốc vs Data mới\n(data mới có nhiều khách hàng mới hơn)')
ax.legend()

# Contract distribution
ax = axes[1, 1]
orig_contract = df_raw['Contract'].value_counts(normalize=True) * 100
new_contract  = all_new['Contract'].value_counts(normalize=True) * 100
x = np.arange(len(orig_contract))
w = 0.35
ax.bar(x - w/2, orig_contract.values, w, label='Data gốc', color='#4C8EDA', alpha=0.8)
ax.bar(x + w/2, [new_contract.get(c, 0) for c in orig_contract.index],
       w, label='Data mới', color='#E05C5C', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(orig_contract.index, rotation=10)
ax.set_ylabel('Tỉ lệ (%)')
ax.set_title('Phân phối Contract: gốc vs mới')
ax.legend()

plt.tight_layout()
plt.savefig('simulate_drift_analysis.png', dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: simulate_drift_analysis.png")

# ─────────────────────────────────────────────
# Lưu summary
# ─────────────────────────────────────────────
df_summary = pd.DataFrame(batch_summary)
df_summary.to_csv('monthly_batches/batch_summary.csv', index=False)

total_new = sum(s['n_customers'] for s in batch_summary)
avg_drift = churn_rates[-1] - churn_rates[0]

print(f"""
{"=" * 60}
TỔNG KẾT SIMULATE
{"=" * 60}

  Đã tạo: {N_MONTHS} batch × ~{N_PER_MONTH} khách hàng = {total_new:,} records
  Lưu tại: monthly_batches/month_01.csv → month_06.csv

  [DM] Data Drift phát hiện:
  • Churn rate tháng 1: {churn_rates[0]:.1f}%
  • Churn rate tháng 6: {churn_rates[-1]:.1f}%
  • Drift: +{avg_drift:.1f}% → model cũ sẽ ngày càng kém chính xác

  → Đây là lý do phải RETRAIN định kỳ (File 7)
  → File 7: Retrain pipeline — cập nhật model với data mới
""")