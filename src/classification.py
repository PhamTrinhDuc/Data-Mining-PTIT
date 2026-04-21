"""
File 5: Classification — Dự đoán Churn
========================================
[DM Note] Bước này xây dựng model dự đoán, nhưng trọng tâm DM
          KHÔNG phải accuracy cao nhất — mà là:
            1. Giải thích được tại sao model đưa ra dự đoán đó
            2. Feature nào quan trọng → insight về hành vi khách hàng
            3. Kết quả phải "actionable" — cô có thể ra quyết định kinh doanh

          Model chọn: Random Forest
          Lý do: tự nhiên cho ra feature importance, ít cần tune,
                 xử lý tốt dữ liệu hỗn hợp (numeric + binary)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_recall_curve, f1_score)
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Load data từ File 3 + cluster label từ File 4
# ─────────────────────────────────────────────
PROJET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSED_DIR  = os.path.join(PROJET_DIR, 'data/processed')
CLUSTER_DIR = os.path.join(PROJET_DIR, 'data/clustering')
X_train = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'y_train.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'y_test.csv')).squeeze()

# [DM] Thêm cluster label vào feature set
# → Cluster là "nhóm nguy hiểm" → thông tin hữu ích cho model
X_full_clustered = pd.read_csv(os.path.join(CLUSTER_DIR, 'X_full_clustered.csv'))
cluster_all = X_full_clustered['cluster'].values

# Gán cluster cho train/test (dùng index giống File 3: 80/20 split)
n_train = len(X_train)
X_train['cluster'] = cluster_all[:n_train]
X_test['cluster']  = cluster_all[n_train:]

with open(os.path.join(PREPROCESSED_DIR, 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)
feature_names_with_cluster = list(X_train.columns)

print("=" * 60)
print("CLASSIFICATION — RANDOM FOREST")
print("=" * 60)
print(f"\nTrain: {X_train.shape[0]:,} mẫu | Test: {X_test.shape[0]:,} mẫu")
print(f"Features: {X_train.shape[1]} (gồm cluster label từ File 4)")
print(f"Churn rate — Train: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

# ─────────────────────────────────────────────
# BƯỚC 1: Train Random Forest
# ─────────────────────────────────────────────
print("\n── Bước 1: Train Random Forest ──")

# [DM] Tham số quan trọng:
# class_weight='balanced': tự động điều chỉnh weight vì dataset mất cân bằng
#   → Không dùng balanced thì model sẽ bias sang "No churn" (73%)
# n_estimators=200: 200 cây → ổn định hơn 100, không quá tốn thời gian
# max_depth=10: giới hạn độ sâu → tránh overfitting
# min_samples_leaf=20: mỗi leaf cần ≥20 mẫu → tránh học thuộc lòng noise
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    class_weight='balanced',   # quan trọng với imbalanced dataset
    random_state=42,
    n_jobs=-1                  # dùng tất cả CPU cores
)

rf.fit(X_train, y_train)
print(f"  ✅ Train xong — {rf.n_estimators} cây, max_depth={rf.max_depth}")
print(f"  class_weight='balanced' → tránh bias về majority class")

# ─────────────────────────────────────────────
# BƯỚC 2: Dự đoán & Đánh giá
# ─────────────────────────────────────────────
print("\n── Bước 2: Đánh giá model ──")

y_pred      = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]   # xác suất churn

# [DM] Tại sao không chỉ dùng accuracy?
# Nếu model luôn đoán "No churn" → accuracy = 73% nhưng vô dụng hoàn toàn
# → Dùng F1 (precision + recall) và ROC-AUC thay thế
print(f"\n  Accuracy:  {(y_pred == y_test).mean()*100:.2f}%")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"  F1 (churn):{f1_score(y_test, y_pred):.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No churn', 'Churn']))

# [DM] Phân tích Confusion Matrix — hiểu loại lỗi nào nặng hơn
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  Confusion Matrix:")
print(f"                Predicted No  Predicted Yes")
print(f"  Actual No:       {tn:5d}         {fp:5d}   (False Positive: báo động nhầm)")
print(f"  Actual Yes:      {fn:5d}         {tp:5d}   (False Negative: bỏ sót churn)")
print(f"\n  [DM] Trong bài toán churn:")
print(f"   False Negative ({fn}) nguy hiểm hơn False Positive ({fp})")
print(f"   → Bỏ sót khách hàng sắp rời > Cảnh báo nhầm khách hàng ở lại")

# ─────────────────────────────────────────────
# BƯỚC 3: Feature Importance — Trái tim của DM
# ─────────────────────────────────────────────
print("\n── Bước 3: Feature Importance ──")

# [DM] Đây là phần làm bài thành DM thực sự
# Không chỉ biết model "đúng bao nhiêu %" mà còn biết "dựa vào gì"
importances = pd.Series(rf.feature_importances_,
                        index=feature_names_with_cluster)
importances = importances.sort_values(ascending=False)

print("\n  Top 10 features quan trọng nhất:")
for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
    bar = '█' * int(imp * 200)
    print(f"   {i:2d}. {feat:<35} {bar} {imp:.4f}")

# [DM] Giải thích ý nghĩa kinh doanh
print(f"""
  [DM Interpretation — Feature Importance]
  Top 3 features có ý nghĩa kinh doanh:

  1. '{importances.index[0]}': {importances.iloc[0]:.4f}
     → Quan trọng nhất: ảnh hưởng trực tiếp đến quyết định rời bỏ

  2. '{importances.index[1]}': {importances.iloc[1]:.4f}
     → Cam kết dài hạn giúp giữ chân khách hàng hiệu quả

  3. '{importances.index[2]}': {importances.iloc[2]:.4f}
     → Khách hàng mới chưa gắn bó = dễ bỏ nhất
""")

# ─────────────────────────────────────────────
# BƯỚC 4: Visualize kết quả
# ─────────────────────────────────────────────
print("── Bước 4: Visualize ──")

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig)

# --- 4a. Feature Importance (top 15) ---
ax1 = fig.add_subplot(gs[0, :])
top15 = importances.head(15)
colors = ['#E05C5C' if i < 3 else '#F5A623' if i < 7 else '#4C8EDA'
          for i in range(len(top15))]
bars = ax1.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
ax1.set_xlabel('Feature Importance (Mean Decrease Impurity)')
ax1.set_title('Feature Importance — Top 15 (đỏ = quan trọng nhất)', fontsize=13)
for bar, val in zip(bars, top15.values[::-1]):
    ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()

# --- 4b. Confusion Matrix ---
ax2 = fig.add_subplot(gs[1, 0])
im = ax2.imshow(cm, cmap='Blues')
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Predicted No', 'Predicted Yes'])
ax2.set_yticklabels(['Actual No', 'Actual Yes'])
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i,j] > cm.max()/2 else 'black'
        ax2.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                 fontsize=14, fontweight='bold', color=color)
ax2.set_title('Confusion Matrix')
plt.colorbar(im, ax=ax2)

# --- 4c. ROC Curve ---
ax3 = fig.add_subplot(gs[1, 1])
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)
ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC={auc:.3f})')
ax3.plot([0,1], [0,1], 'r--', linewidth=1, label='Random guess (AUC=0.5)')
ax3.fill_between(fpr, tpr, alpha=0.1)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.suptitle('Classification Results — Random Forest', fontsize=14)
plt.tight_layout()
plt.savefig('clf_01_results.png', dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: clf_01_results.png")

# ─────────────────────────────────────────────
# BƯỚC 5: Churn Probability Analysis
# ─────────────────────────────────────────────
print("\n── Bước 5: Phân tích xác suất churn ──")

# [DM] Thay vì chỉ dự đoán Yes/No, xem phân phối xác suất
# → Phát hiện "nhóm nguy hiểm trung bình" mà threshold thông thường bỏ qua
df_result = X_test.copy()
df_result['actual_churn']   = y_test.values
df_result['churn_prob']     = y_pred_prob
df_result['predicted_churn']= y_pred

# Phân nhóm theo mức độ rủi ro
df_result['risk_level'] = pd.cut(df_result['churn_prob'],
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=['Thấp (<30%)', 'Trung bình (30-50%)',
            'Cao (50-70%)', 'Rất cao (>70%)'])

risk_analysis = df_result.groupby('risk_level', observed=True).agg(
    count=('actual_churn', 'size'),
    actual_churn_rate=('actual_churn', 'mean')
).reset_index()
risk_analysis['actual_churn_rate'] *= 100

print("\n  Phân tích theo mức độ rủi ro:")
print(f"  {'Risk level':<25} {'Count':>7} {'Actual churn rate':>18}")
print("  " + "-" * 55)
for _, row in risk_analysis.iterrows():
    print(f"  {str(row['risk_level']):<25} {int(row['count']):>7,} {row['actual_churn_rate']:>17.1f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram xác suất churn
ax1.hist(df_result[df_result['actual_churn']==0]['churn_prob'],
         bins=40, alpha=0.6, color='#4C8EDA', label='No churn', density=True)
ax1.hist(df_result[df_result['actual_churn']==1]['churn_prob'],
         bins=40, alpha=0.6, color='#E05C5C', label='Churn', density=True)
ax1.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
ax1.set_xlabel('Predicted Churn Probability')
ax1.set_ylabel('Density')
ax1.set_title('Phân phối xác suất churn')
ax1.legend()

# Risk level bar
bars = ax2.bar(risk_analysis['risk_level'],
               risk_analysis['actual_churn_rate'],
               color=['#4C8EDA','#F5A623','#E05C5C','#8B0000'])
ax2.set_ylabel('Actual Churn Rate (%)')
ax2.set_title('Tỉ lệ churn thực tế theo Risk Level')
ax2.tick_params(axis='x', rotation=15)
for bar, rate in zip(bars, risk_analysis['actual_churn_rate']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{rate:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('clf_02_risk_analysis.png', dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: clf_02_risk_analysis.png")

# ─────────────────────────────────────────────
# BƯỚC 6: Lưu model
# ─────────────────────────────────────────────
print("\n── Bước 6: Lưu model ──")

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Lưu kết quả dự đoán để phân tích thêm
df_result.to_csv('prediction_results.csv', index=False)

print("  ✅ rf_model.pkl           — Random Forest model")
print("  ✅ prediction_results.csv — kết quả dự đoán kèm xác suất")

# ─────────────────────────────────────────────
# TỔNG KẾT
# ─────────────────────────────────────────────
auc_final = roc_auc_score(y_test, y_pred_prob)
f1_final  = f1_score(y_test, y_pred)

print(f"""
{"=" * 60}
TỔNG KẾT — [DM INSIGHTS TỪ CLASSIFICATION]
{"=" * 60}

  Model: Random Forest (200 cây, max_depth=10)
  ROC-AUC: {auc_final:.4f} | F1-score: {f1_final:.4f}

  [DM Insights — những gì DATA nói với chúng ta]

  1. Feature quan trọng nhất: '{importances.index[0]}'
     → Đây là yếu tố ảnh hưởng nhất đến quyết định rời bỏ

  2. Cluster label (từ File 4) có trong top features
     → Xác nhận: phân nhóm khách hàng giúp ích cho dự đoán

  3. Khách hàng có churn_prob > 70%: cần can thiệp ngay
     → Gợi ý: ưu đãi gia hạn hợp đồng, tặng tháng miễn phí

  4. Khách hàng có churn_prob 30-50%: theo dõi sát
     → Gợi ý: cải thiện trải nghiệm dịch vụ, hỗ trợ kỹ thuật

  → File 6: Simulate collect data & Retrain pipeline
""")