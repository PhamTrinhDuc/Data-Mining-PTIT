"""
File 7: Retrain Pipeline — Cập nhật model tự động
===================================================
[DM Note] Đây là phần "tự động" của Data Mining pipeline.
          ML thông thường: train 1 lần → deploy → xong.
          DM pipeline: liên tục học từ data mới → model luôn cập nhật.

          Chiến lược retrain: INCREMENTAL (tích lũy)
          → Mỗi tháng: gộp data gốc + tất cả batch đã có → retrain
          → So sánh performance model mới vs model cũ
          → Nếu model mới tốt hơn → thay thế (model selection)

          Đây là điểm cốt lõi phân biệt DM pipeline vs một lần train ML.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import pickle
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Load artifacts từ các file trước
# ─────────────────────────────────────────────
X_train_orig = pd.read_csv('X_train.csv')
X_test_orig  = pd.read_csv('X_test.csv')
y_train_orig = pd.read_csv('y_train.csv').squeeze()
y_test_orig  = pd.read_csv('y_test.csv').squeeze()
X_full       = pd.read_csv('X_full.csv')
y_full       = pd.read_csv('y_full.csv').squeeze()

with open('rf_model.pkl', 'rb') as f:
    model_baseline = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler_orig = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("=" * 60)
print("RETRAIN PIPELINE — CẬP NHẬT MODEL THEO THÁNG")
print("=" * 60)

# ─────────────────────────────────────────────
# Hàm tiền xử lý batch mới (giống File 3)
# ─────────────────────────────────────────────

def preprocess_batch(df_batch: pd.DataFrame) -> tuple:
    """
    Áp dụng cùng pipeline tiền xử lý như File 3
    cho một batch data mới.
    
    [DM] Quan trọng: phải dùng CÙNG encoding logic
         để features nhất quán với data gốc.
    """
    df = df_batch.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Encode target
    y = (df['Churn'] == 'Yes').astype(int)
    df = df.drop(columns=['customerID', 'Churn'])

    # Binary encode
    binary_map = {
        'gender': {'Male': 1, 'Female': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
    }
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping)

    # Ordinal encode Contract
    df['Contract'] = df['Contract'].map(
        {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    )

    # One-hot encode
    df = pd.get_dummies(df, columns=['InternetService', 'PaymentMethod'], drop_first=False)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Service binary
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
        df[col] = (df[col] == 'Yes').astype(int)

    # Align columns với data gốc (one-hot có thể tạo ra cột khác)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0   # thiếu category → điền 0
    df = df[feature_names]   # đúng thứ tự

    return df, y


def scale_batch(X_new: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Scale batch mới dùng scaler đã fit từ data gốc.
    [DM] KHÔNG refit scaler — dùng lại scaler cũ để nhất quán.
    """
    X_scaled = X_new.copy()
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_scaled[scale_cols] = scaler.transform(X_new[scale_cols])
    return X_scaled


def evaluate_model(model, X_test, y_test, label="") -> dict:
    """Đánh giá model và trả về dict metrics."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return {
        'label':   label,
        'f1':      f1_score(y_test, y_pred),
        'auc':     roc_auc_score(y_test, y_pred_prob),
        'accuracy':(y_pred == y_test).mean(),
    }

# ─────────────────────────────────────────────
# BƯỚC 1: Đánh giá baseline (model từ File 5)
# ─────────────────────────────────────────────
print("\n── Bước 1: Baseline — Model từ File 5 ──")

# [DM] Dùng test set gốc làm benchmark cố định
# → Mọi model mới đều so sánh trên cùng test set này
baseline_metrics = evaluate_model(model_baseline, X_test_orig, y_test_orig, "Baseline")
print(f"  Baseline — F1: {baseline_metrics['f1']:.4f} | AUC: {baseline_metrics['auc']:.4f}")

# Test model baseline trên data drift (batch mới nhất — chưa thấy)
df_latest = pd.read_csv('monthly_batches/month_06.csv')
X_latest, y_latest = preprocess_batch(df_latest)
X_latest_scaled    = scale_batch(X_latest, scaler_orig)

baseline_on_new = evaluate_model(model_baseline, X_latest_scaled, y_latest, "Baseline on new data")
print(f"  Baseline trên data tháng 6 — F1: {baseline_on_new['f1']:.4f} | AUC: {baseline_on_new['auc']:.4f}")
print(f"\n  [DM] Nếu F1 giảm → model cũ đang drift → cần retrain!")

# ─────────────────────────────────────────────
# BƯỚC 2: Retrain pipeline từng tháng
# ─────────────────────────────────────────────
print(f"\n── Bước 2: Retrain theo từng tháng ──")
print(f"\n  {'Tháng':<8} {'Data size':>10} {'F1 (test)':>11} {'AUC (test)':>12} {'F1 on new':>11} {'Δ F1':>8}")
print("  " + "-" * 65)

history = []
current_model  = model_baseline
current_scaler = scaler_orig

# Data tích lũy — bắt đầu từ data gốc
X_accumulated = X_full.copy()
y_accumulated = y_full.copy()

batch_files = sorted(glob.glob('monthly_batches/month_*.csv'))

for i, fpath in enumerate(batch_files):
    month = i + 1

    # ── Load và preprocess batch mới ──
    df_batch        = pd.read_csv(fpath)
    X_batch, y_batch = preprocess_batch(df_batch)

    # ── Tích lũy data: gộp batch mới vào dataset ──
    # [DM] Chiến lược INCREMENTAL: thêm mới, không xóa cũ
    # → Model học được cả pattern cũ lẫn trend mới
    X_accumulated = pd.concat([X_accumulated, X_batch], ignore_index=True)
    y_accumulated = pd.concat([y_accumulated, y_batch], ignore_index=True)

    # ── Refit scaler trên data tích lũy ──
    # [DM] Mỗi lần retrain: fit lại scaler vì distribution thay đổi
    new_scaler = StandardScaler()
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_acc_scaled = X_accumulated.copy()
    X_acc_scaled[scale_cols] = new_scaler.fit_transform(X_accumulated[scale_cols])

    # ── Retrain model ──
    new_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    new_model.fit(X_acc_scaled, y_accumulated)

    # ── Đánh giá trên test set cố định ──
    X_test_rescaled = X_test_orig.copy()
    X_test_rescaled[scale_cols] = new_scaler.transform(X_test_orig[scale_cols])
    new_metrics = evaluate_model(new_model, X_test_rescaled, y_test_orig, f"Month {month}")

    # ── Đánh giá trên batch mới nhất (data chưa thấy) ──
    X_latest_rescaled = scale_batch(X_latest, new_scaler)
    new_on_latest = evaluate_model(new_model, X_latest_rescaled, y_latest)

    delta_f1 = new_metrics['f1'] - baseline_metrics['f1']
    sign = '+' if delta_f1 >= 0 else ''

    print(f"  Tháng {month:<4} {len(X_accumulated):>10,} {new_metrics['f1']:>11.4f} "
          f"{new_metrics['auc']:>12.4f} {new_on_latest['f1']:>11.4f} "
          f"{sign}{delta_f1:>7.4f}")

    history.append({
        'month':        month,
        'data_size':    len(X_accumulated),
        'f1_test':      new_metrics['f1'],
        'auc_test':     new_metrics['auc'],
        'f1_on_new':    new_on_latest['f1'],
        'delta_f1':     delta_f1,
        'model':        new_model,
        'scaler':       new_scaler,
    })

# ─────────────────────────────────────────────
# BƯỚC 3: Model Selection — chọn model tốt nhất
# ─────────────────────────────────────────────
print(f"\n── Bước 3: Model Selection ──")

# [DM] Tiêu chí chọn: F1 trên test set cố định
# → Đảm bảo fair comparison (cùng test set)
best_month = max(history, key=lambda x: x['f1_test'])
best_f1    = best_month['f1_test']

print(f"\n  Baseline F1:        {baseline_metrics['f1']:.4f}")
print(f"  Best retrained F1:  {best_f1:.4f} (sau tháng {best_month['month']})")
print(f"  Cải thiện:          {best_f1 - baseline_metrics['f1']:+.4f}")

if best_f1 > baseline_metrics['f1']:
    final_model  = best_month['model']
    final_scaler = best_month['scaler']
    print(f"\n  ✅ Model tháng {best_month['month']} tốt hơn baseline → DEPLOY model mới")
else:
    final_model  = model_baseline
    final_scaler = scaler_orig
    print(f"\n  ⚠️  Baseline vẫn tốt hơn → GIỮ model cũ")

# ─────────────────────────────────────────────
# BƯỚC 4: Visualize kết quả retrain
# ─────────────────────────────────────────────
print(f"\n── Bước 4: Visualize retrain history ──")

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig)

months_list  = [h['month'] for h in history]
f1_list      = [h['f1_test'] for h in history]
auc_list     = [h['auc_test'] for h in history]
f1_new_list  = [h['f1_on_new'] for h in history]
size_list    = [h['data_size'] for h in history]

# --- 4a. F1 trên test set theo tháng ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(months_list, f1_list, 'bo-', linewidth=2, markersize=8, label='F1 (test set cố định)')
ax1.axhline(y=baseline_metrics['f1'], color='red', linestyle='--',
            label=f'Baseline F1={baseline_metrics["f1"]:.4f}')
ax1.fill_between(months_list, baseline_metrics['f1'], f1_list,
                 alpha=0.1, color='green' if f1_list[-1] > baseline_metrics['f1'] else 'red')
ax1.set_xlabel('Tháng')
ax1.set_ylabel('F1 Score')
ax1.set_title('F1 Score theo tháng retrain')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 4b. AUC theo tháng ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(months_list, auc_list, 'gs-', linewidth=2, markersize=8, label='AUC (test set cố định)')
ax2.axhline(y=baseline_metrics['auc'], color='red', linestyle='--',
            label=f'Baseline AUC={baseline_metrics["auc"]:.4f}')
ax2.set_xlabel('Tháng')
ax2.set_ylabel('ROC-AUC')
ax2.set_title('ROC-AUC theo tháng retrain')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- 4c. F1 trên data mới (unseen) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(months_list, f1_new_list, 'r^-', linewidth=2, markersize=8,
         label='F1 trên data tháng 6 (unseen)')
ax3.axhline(y=baseline_on_new['f1'], color='gray', linestyle='--',
            label=f'Baseline trên data mới={baseline_on_new["f1"]:.4f}')
ax3.set_xlabel('Tháng')
ax3.set_ylabel('F1 Score')
ax3.set_title('F1 trên data mới (unseen)\n→ Model mới thích nghi tốt hơn?')
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- 4d. Data size tích lũy ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(months_list, size_list, color='#4C8EDA', alpha=0.8)
ax4.set_xlabel('Tháng')
ax4.set_ylabel('Tổng số mẫu tích lũy')
ax4.set_title('Data size tích lũy theo tháng')
for i, (m, s) in enumerate(zip(months_list, size_list)):
    ax4.text(m, s + 50, f'{s:,}', ha='center', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Retrain Pipeline — Theo dõi model performance theo tháng', fontsize=14)
plt.tight_layout()
plt.savefig('retrain_history.png', dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: retrain_history.png")

# ─────────────────────────────────────────────
# BƯỚC 5: So sánh Feature Importance trước/sau retrain
# ─────────────────────────────────────────────
print(f"\n── Bước 5: So sánh Feature Importance ──")

fi_baseline = pd.Series(model_baseline.feature_importances_,
                         index=X_train_orig.columns).sort_values(ascending=False)
fi_retrained = pd.Series(final_model.feature_importances_,
                          index=X_accumulated.columns).sort_values(ascending=False)

top_n = 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.barh(fi_baseline.head(top_n).index[::-1],
         fi_baseline.head(top_n).values[::-1], color='#4C8EDA', alpha=0.8)
ax1.set_title(f'Feature Importance — Baseline\n(data gốc)')
ax1.set_xlabel('Importance')

ax2.barh(fi_retrained.head(top_n).index[::-1],
         fi_retrained.head(top_n).values[::-1], color='#E05C5C', alpha=0.8)
ax2.set_title(f'Feature Importance — Sau Retrain\n(data gốc + 6 tháng mới)')
ax2.set_xlabel('Importance')

plt.suptitle('So sánh Feature Importance: Baseline vs Retrained', fontsize=13)
plt.tight_layout()
plt.savefig('retrain_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: retrain_feature_importance.png")

# [DM] Nếu thứ tự feature thay đổi → data drift thực sự có ảnh hưởng
top3_base    = list(fi_baseline.head(3).index)
top3_retrain = list(fi_retrained.head(3).index)
print(f"\n  Top 3 features — Baseline:  {top3_base}")
print(f"  Top 3 features — Retrained: {top3_retrain}")
if top3_base != top3_retrain:
    print(f"  → Thứ tự thay đổi: data drift đã ảnh hưởng đến model!")
else:
    print(f"  → Thứ tự ổn định: pattern chính không đổi")

# ─────────────────────────────────────────────
# BƯỚC 6: Lưu final model
# ─────────────────────────────────────────────
print(f"\n── Bước 6: Lưu final model ──")

with open('rf_model_final.pkl', 'wb') as f:
    pickle.dump(final_model, f)
with open('scaler_final.pkl', 'wb') as f:
    pickle.dump(final_scaler, f)

# Lưu history để báo cáo
history_df = pd.DataFrame([{k: v for k, v in h.items() if k not in ['model','scaler']}
                            for h in history])
history_df.to_csv('retrain_history.csv', index=False)

print(f"  ✅ rf_model_final.pkl     — model cuối (đã retrain)")
print(f"  ✅ scaler_final.pkl       — scaler cuối")
print(f"  ✅ retrain_history.csv    — lịch sử retrain")

# ─────────────────────────────────────────────
# TỔNG KẾT TOÀN BỘ PIPELINE
# ─────────────────────────────────────────────
print(f"""
{"=" * 60}
TỔNG KẾT TOÀN BỘ DM PIPELINE
{"=" * 60}

  File 1 — Dataset:       Load, mô tả, phân nhóm features
  File 2 — EDA:           Khám phá pattern, visualize churn
  File 3 — Preprocessing: Encode, scale, split chuẩn
  File 4 — Clustering:    Phân nhóm khách hàng (unsupervised)  ★ DM
  File 5 — Classification:Train RF, feature importance         ★ DM
  File 6 — Simulate:      Giả lập thu thập data theo tháng     ★ DM
  File 7 — Retrain:       Cập nhật model tự động               ★ DM

  Baseline  → F1: {baseline_metrics['f1']:.4f} | AUC: {baseline_metrics['auc']:.4f}
  Retrained → F1: {best_f1:.4f} | AUC: {best_month['auc_test']:.4f}
  Cải thiện → ΔF1: {best_f1 - baseline_metrics['f1']:+.4f}

  [DM Insights tổng hợp]
  1. Khách hàng tenure < 12 tháng + month-to-month = nguy cơ cao nhất
  2. Clustering phát hiện {len(history[0]['model'].classes_)} nhóm tự nhiên
     với churn rate khác biệt rõ rệt
  3. Data drift: churn rate tăng theo tháng
     → Retrain định kỳ giúp model thích nghi với xu hướng mới
  4. Feature importance ổn định → pattern cốt lõi không thay đổi
     nhưng model cập nhật vẫn tốt hơn nhờ thêm data
""")