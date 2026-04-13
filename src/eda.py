"""
File 2: EDA — Exploratory Data Analysis
=========================================
[DM Note] EDA trong Data Mining không chỉ là "nhìn data cho biết".
          Mục tiêu là ĐẶT CÂU HỎI và TÌM PATTERN ẩn trong dữ liệu.
          
          Câu hỏi dẫn dắt EDA này:
            Q1: Feature nào liên quan nhất đến churn?
            Q2: Khách hàng churn trông như thế nào (profile)?
            Q3: Có pattern bất thường nào không?
"""
import os 
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
df = pd.read_csv(os.path.join(PROJECT_DIR, "data/data_raw.csv"))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Bảng màu nhất quán cho toàn bộ EDA
COLOR_NO  = '#4C8EDA'   # xanh = ở lại
COLOR_YES = '#E05C5C'   # đỏ  = rời bỏ
PALETTE   = {'No': COLOR_NO, 'Yes': COLOR_YES}

print("=" * 60)
print("EDA — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────────
# PHẦN A: Kiểm tra chất lượng dữ liệu
# ─────────────────────────────────────────────
print("\n── A. CHẤT LƯỢNG DỮ LIỆU ──")

# [DM] Missing values là vấn đề phổ biến — phải kiểm tra trước khi phân tích
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    print(f"\n⚠️  Có {len(missing)} cột bị missing:")
    for col, cnt in missing.items():
        pct = cnt / len(df) * 100
        print(f"   • {col}: {cnt} giá trị ({pct:.2f}%)")
        
    # [DM] Kiểm tra: các dòng missing TotalCharges có pattern gì không?
    # → Nếu tenure=0 thì chưa có phí → đây là pattern có ý nghĩa kinh doanh!
    missing_rows = df[df['TotalCharges'].isnull()]
    print(f"\n   Phân tích dòng bị missing TotalCharges:")
    print(f"   tenure trung bình: {missing_rows['tenure'].mean():.1f} tháng")
    print(f"   → Kết luận: khách hàng mới (tenure=0), chưa phát sinh phí")
else:
    print("  ✅ Không có missing values")

# Kiểm tra duplicate
n_dup = df.duplicated().sum()
print(f"\n  Duplicate rows: {n_dup}")

# ─────────────────────────────────────────────
# PHẦN B: Phân phối numeric features
# ─────────────────────────────────────────────
print("\n── B. PHÂN PHỐI NUMERIC FEATURES ──")

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('B. Phân phối Numeric Features — phân tách theo Churn', fontsize=14, y=1.02)

for i, col in enumerate(numeric_cols):
    # Histogram phân tách theo churn
    ax = axes[0, i]
    for churn_val, color in [('No', COLOR_NO), ('Yes', COLOR_YES)]:
        subset = df[df['Churn'] == churn_val][col].dropna()
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=churn_val, density=True)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend()

    # Boxplot so sánh
    ax2 = axes[1, i]
    data_no  = df[df['Churn'] == 'No'][col].dropna()
    data_yes = df[df['Churn'] == 'Yes'][col].dropna()
    ax2.boxplot([data_no, data_yes], labels=['No churn', 'Churn'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title(f'{col} — boxplot')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'eda_B_numeric.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: eda_B_numeric.png")

# [DM] In insight từ numeric
for col in numeric_cols:
    mean_no  = df[df['Churn']=='No'][col].mean()
    mean_yes = df[df['Churn']=='Yes'][col].mean()
    diff_pct = (mean_yes - mean_no) / mean_no * 100
    direction = "cao hơn" if diff_pct > 0 else "thấp hơn"
    print(f"\n  [{col}]")
    print(f"   No churn: mean={mean_no:.1f} | Churn: mean={mean_yes:.1f}")
    print(f"   → Khách hàng churn có {col} {direction} {abs(diff_pct):.1f}%")

# ─────────────────────────────────────────────
# PHẦN C: Churn rate theo từng categorical feature
# ─────────────────────────────────────────────
print("\n── C. CHURN RATE THEO CATEGORICAL FEATURES ──")

# [DM] Đây là phần quan trọng nhất của EDA trong DM
# → Tìm feature nào "phân tách" khách hàng churn rõ ràng nhất

cat_features = [
    'Contract', 'InternetService', 'PaymentMethod',
    'TechSupport', 'OnlineSecurity', 'PaperlessBilling',
    'SeniorCitizen', 'Partner', 'Dependents'
]

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('C. Churn Rate theo Categorical Features', fontsize=14)
axes = axes.flatten()

for i, col in enumerate(cat_features):
    ax = axes[i]
    
    # Tính churn rate từng category
    churn_rate = (df.groupby(col)['Churn']
                    .apply(lambda x: (x == 'Yes').mean() * 100)
                    .reset_index())
    churn_rate.columns = [col, 'churn_rate']
    churn_rate = churn_rate.sort_values('churn_rate', ascending=True)
    
    # Tô màu theo mức độ nguy hiểm
    colors = ['#E05C5C' if r > 40 else '#F5A623' if r > 25 else '#4C8EDA' 
              for r in churn_rate['churn_rate']]
    
    bars = ax.barh(churn_rate[col].astype(str), churn_rate['churn_rate'], color=colors)
    ax.set_xlabel('Churn rate (%)')
    ax.set_title(col, fontweight='bold')
    ax.axvline(x=df['Churn'].eq('Yes').mean()*100, color='gray',
               linestyle='--', alpha=0.7, label='Average')
    
    # Hiện % trên bar
    for bar, rate in zip(bars, churn_rate['churn_rate']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}%', va='center', fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'eda_C_categorical.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: eda_C_categorical.png")

# [DM] In top insight từ categorical
print("\n  Top insights từ categorical features:")
avg_churn = df['Churn'].eq('Yes').mean() * 100

for col in cat_features:
    churn_by_cat = (df.groupby(col)['Churn']
                      .apply(lambda x: (x == 'Yes').mean() * 100))
    max_cat  = churn_by_cat.idxmax()
    max_rate = churn_by_cat.max()
    min_cat  = churn_by_cat.idxmin()
    min_rate = churn_by_cat.min()
    
    if max_rate - min_rate > 20:  # Chỉ in nếu sự chênh lệch đủ lớn
        print(f"\n  [{col}] — chênh lệch {max_rate - min_rate:.1f}%")
        print(f"   Cao nhất: '{max_cat}' → {max_rate:.1f}%")
        print(f"   Thấp nhất: '{min_cat}' → {min_rate:.1f}%")

# ─────────────────────────────────────────────
# PHẦN D: Correlation heatmap
# ─────────────────────────────────────────────
print("\n── D. CORRELATION ANALYSIS ──")

# [DM] Encode nhị phân để tính correlation với target
df_corr = df.copy()
df_corr['Churn_bin'] = (df_corr['Churn'] == 'Yes').astype(int)
df_corr['SeniorCitizen'] = df_corr['SeniorCitizen'].astype(int)

# One-hot encode các cột categorical đơn giản
simple_binary = {
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 0},
}
for col, mapping in simple_binary.items():
    df_corr[col] = df_corr[col].map(mapping)

# Contract encode (month-to-month = rủi ro cao nhất)
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df_corr['Contract_encoded'] = df_corr['Contract'].map(contract_map)

corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
             'Partner', 'Dependents', 'PaperlessBilling', 'TechSupport',
             'OnlineSecurity', 'Contract_encoded', 'Churn_bin']

corr_matrix = df_corr[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Chỉ hiện nửa dưới
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=ax, square=True, linewidths=0.5)
ax.set_title('D. Correlation Heatmap — Tương quan giữa các features và Churn', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'eda_D_correlation.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: eda_D_correlation.png")

# [DM] In correlation với target
churn_corr = corr_matrix['Churn_bin'].drop('Churn_bin').sort_values(key=abs, ascending=False)
print("\n  Correlation với Churn (từ cao đến thấp):")
for feat, corr in churn_corr.items():
    bar = '█' * int(abs(corr) * 20)
    sign = '+' if corr > 0 else '-'
    print(f"   {feat:<22} {sign}{bar} {corr:+.3f}")

# ─────────────────────────────────────────────
# PHẦN E: Tenure segments — phát hiện "vùng nguy hiểm"
# ─────────────────────────────────────────────
print("\n── E. PHÂN TÍCH TENURE SEGMENTS ──")

# [DM] Chia tenure thành nhóm để tìm "vùng nguy hiểm" 
# → Đây là bước tìm pattern thay vì chỉ dùng raw number
df['tenure_group'] = pd.cut(df['tenure'],
    bins=[0, 6, 12, 24, 36, 48, 72],
    labels=['0-6 tháng', '7-12 tháng', '13-24 tháng',
            '25-36 tháng', '37-48 tháng', '49+ tháng'])

tenure_churn = (df.groupby('tenure_group', observed=True)['Churn']
                  .apply(lambda x: (x == 'Yes').mean() * 100)
                  .reset_index())
tenure_churn.columns = ['tenure_group', 'churn_rate']

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(tenure_churn['tenure_group'], tenure_churn['churn_rate'],
              color=['#E05C5C' if r > 35 else '#F5A623' if r > 20 else '#4C8EDA'
                     for r in tenure_churn['churn_rate']])
ax.axhline(y=avg_churn, color='gray', linestyle='--', label=f'Avg churn ({avg_churn:.1f}%)')
ax.set_xlabel('Nhóm tenure')
ax.set_ylabel('Churn rate (%)')
ax.set_title('E. Churn Rate theo nhóm Tenure — Tìm "vùng nguy hiểm"', fontsize=13)
ax.legend()
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'eda_E_tenure.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: eda_E_tenure.png")

print("\n  Churn rate theo tenure segment:")
for _, row in tenure_churn.iterrows():
    flag = " ← ⚠️  VÙNG NGUY HIỂM" if row['churn_rate'] > 40 else ""
    print(f"   {str(row['tenure_group']):<15}: {row['churn_rate']:.1f}%{flag}")

# ─────────────────────────────────────────────
# TỔNG KẾT EDA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TỔNG KẾT EDA — CÁC PHÁT HIỆN CHÍNH")
print("=" * 60)
print("""
  [DM Insights]

  1. Dataset mất cân bằng (~27% churn) → dùng F1, ROC-AUC

  2. Tenure ngắn (0-12 tháng) = nguy hiểm nhất
     → Khách hàng mới chưa gắn bó với dịch vụ

  3. Contract month-to-month có churn rate cao gấp 3-5x
     so với hợp đồng 1-2 năm
     → Không có cam kết = dễ bỏ

  4. Không có TechSupport / OnlineSecurity → churn cao
     → Khách hàng không thấy giá trị của dịch vụ

  5. MonthlyCharges cao + tenure ngắn = profile churn điển hình
     → Tốn tiền nhiều nhưng chưa thấy lợi ích

  → Tiếp theo: File 3 — Tiền xử lý & Chuẩn hóa
""")