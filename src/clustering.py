"""
File 4: Clustering — Phân nhóm khách hàng
==========================================
[DM Note] Đây là phần CỐT LÕI của Data Mining trong bài này.
          ML thông thường KHÔNG có bước này.

          Mục tiêu: Tìm ra các nhóm khách hàng TỰ NHIÊN trong data
          (không dùng nhãn Churn) → sau đó phân tích xem nhóm nào
          có nguy cơ churn cao → đây là KNOWLEDGE DISCOVERY.

          Pipeline:
            1. Tìm số cluster tối ưu (Elbow + Silhouette)
            2. Fit KMeans
            3. Phân tích profile từng cluster
            4. Gán cluster vào data gốc → dùng cho Classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Load data đã xử lý từ File 3
# ─────────────────────────────────────────────
X_full    = pd.read_csv('X_full.csv')
y_full    = pd.read_csv('y_full.csv').squeeze()
X_cluster = np.load('X_cluster.npy')           # đã scale toàn bộ features

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("=" * 60)
print("CLUSTERING — PHÂN NHÓM KHÁCH HÀNG")
print("=" * 60)
print(f"\nInput: {X_cluster.shape[0]:,} khách hàng × {X_cluster.shape[1]} features")
print("[DM] Clustering là unsupervised — KHÔNG dùng nhãn Churn khi fit")

# ─────────────────────────────────────────────
# BƯỚC 1: Tìm số cluster tối ưu
# ─────────────────────────────────────────────
print("\n── Bước 1: Tìm số cluster tối ưu ──")

# [DM] Dùng 2 tiêu chí kết hợp:
#   - Elbow method: inertia giảm mạnh rồi "gãy" → chọn điểm gãy
#   - Silhouette score: đo mức độ cluster tách biệt nhau (cao = tốt)
K_range = range(2, 9)
inertias    = []
silhouettes = []

print("  Đang tính Elbow + Silhouette cho k=2..8 ...")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cluster, labels, sample_size=2000))
    print(f"   k={k}: inertia={km.inertia_:,.0f}, silhouette={silhouettes[-1]:.4f}")

# Plot Elbow + Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Số cluster (k)')
ax1.set_ylabel('Inertia (Within-cluster SSE)')
ax1.set_title('Elbow Method — Tìm điểm gãy')
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_range), silhouettes, 'rs-', linewidth=2, markersize=8)
ax2.set_xlabel('Số cluster (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score — Cao hơn = tốt hơn')
ax2.grid(True, alpha=0.3)

# Đánh dấu k tối ưu
best_k = list(K_range)[np.argmax(silhouettes)]
ax2.axvline(x=best_k, color='green', linestyle='--', label=f'Best k={best_k}')
ax2.legend()

plt.suptitle('Bước 1: Tìm số cluster tối ưu', fontsize=13)
plt.tight_layout()
plt.savefig('cluster_01_optimal_k.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  ✅ Lưu: cluster_01_optimal_k.png")
print(f"  → Silhouette cao nhất tại k={best_k} → chọn k={best_k}")

# ─────────────────────────────────────────────
# BƯỚC 2: Fit KMeans với k tối ưu
# ─────────────────────────────────────────────
print(f"\n── Bước 2: Fit KMeans (k={best_k}) ──")

# [DM] n_init=20: chạy 20 lần với centroid khởi tạo khác nhau
# → Tránh local minima (KMeans nhạy với centroid ban đầu)
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_cluster)

# Gán nhãn cluster vào dataframe gốc
df_analysis = X_full.copy()
df_analysis['cluster'] = cluster_labels
df_analysis['Churn']   = y_full.values

# Đếm size từng cluster
cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
print(f"\n  Phân phối cluster:")
for c, size in cluster_sizes.items():
    pct = size / len(cluster_labels) * 100
    print(f"   Cluster {c}: {size:,} khách hàng ({pct:.1f}%)")

sil_final = silhouette_score(X_cluster, cluster_labels, sample_size=2000)
print(f"\n  Silhouette score cuối: {sil_final:.4f}")
print(f"  (> 0.5 = tốt, 0.3-0.5 = chấp nhận được, < 0.3 = kém)")

# ─────────────────────────────────────────────
# BƯỚC 3: Phân tích profile từng cluster
# ─────────────────────────────────────────────
print(f"\n── Bước 3: Phân tích profile từng cluster ──")

# [DM] Đây là bước QUAN TRỌNG NHẤT — "Cluster này là ai?"
# Không chỉ đặt tên số 0,1,2 mà phải mô tả đặc trưng

key_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'Contract', 'TechSupport', 'OnlineSecurity',
                'InternetService_Fiber optic', 'InternetService_No',
                'SeniorCitizen', 'Partner']

# Tính churn rate và mean từng feature theo cluster
churn_by_cluster = df_analysis.groupby('cluster')['Churn'].mean() * 100
profile = df_analysis.groupby('cluster')[key_features].mean()
profile['churn_rate_%'] = churn_by_cluster
profile['size'] = cluster_sizes

print("\n  Profile từng cluster (mean values):")
print(profile[['size', 'churn_rate_%', 'tenure', 'MonthlyCharges',
               'Contract', 'TechSupport', 'OnlineSecurity']].round(2).to_string())

# [DM] Đặt tên cluster dựa trên đặc trưng nổi bật
# → Đây là phần "interpretation" — biến số thành insight
cluster_names = {}
for c in range(best_k):
    churn_rate = churn_by_cluster[c]
    tenure_avg = profile.loc[c, 'tenure']
    contract   = profile.loc[c, 'Contract']  # 0=month, 1=1yr, 2=2yr

    if churn_rate > 40:
        if tenure_avg < 20:
            cluster_names[c] = f"C{c}: Khách hàng mới — nguy cơ CAO ⚠️"
        else:
            cluster_names[c] = f"C{c}: Hợp đồng ngắn — nguy cơ CAO ⚠️"
    elif churn_rate > 20:
        cluster_names[c] = f"C{c}: Khách hàng trung bình — nguy cơ TRUNG BÌNH"
    else:
        cluster_names[c] = f"C{c}: Khách hàng lâu năm — nguy cơ THẤP ✅"

print("\n  [DM] Đặt tên cluster theo đặc trưng:")
for c, name in cluster_names.items():
    print(f"   {name} — churn rate: {churn_by_cluster[c]:.1f}%")

# ─────────────────────────────────────────────
# BƯỚC 4: Visualize — Churn rate theo cluster
# ─────────────────────────────────────────────
print("\n── Bước 4: Visualize cluster ──")

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig)

# --- 4a. Churn rate per cluster (bar) ---
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#E05C5C' if r > 35 else '#F5A623' if r > 20 else '#4C8EDA'
          for r in churn_by_cluster]
bars = ax1.bar([f'Cluster {c}' for c in churn_by_cluster.index],
               churn_by_cluster.values, color=colors)
ax1.axhline(y=y_full.mean()*100, color='gray', linestyle='--', label='Average churn')
ax1.set_ylabel('Churn rate (%)')
ax1.set_title('Churn Rate theo Cluster')
ax1.legend()
for bar, rate in zip(bars, churn_by_cluster):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{rate:.1f}%', ha='center', fontweight='bold')

# --- 4b. Tenure distribution per cluster (box) ---
ax2 = fig.add_subplot(gs[0, 1])
tenure_data = [df_analysis[df_analysis['cluster']==c]['tenure'].values
               for c in range(best_k)]
bp = ax2.boxplot(tenure_data, labels=[f'C{c}' for c in range(best_k)],
                 patch_artist=True)
cluster_colors = ['#E05C5C', '#4C8EDA', '#2ECC71', '#F5A623'][:best_k]
for patch, color in zip(bp['boxes'], cluster_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel('Tenure (tháng)')
ax2.set_title('Phân phối Tenure theo Cluster')

# --- 4c. MonthlyCharges vs Tenure (scatter, màu theo cluster) ---
ax3 = fig.add_subplot(gs[1, 0])
for c in range(best_k):
    mask = df_analysis['cluster'] == c
    ax3.scatter(df_analysis[mask]['tenure'],
                df_analysis[mask]['MonthlyCharges'],
                c=cluster_colors[c], alpha=0.3, s=10, label=f'C{c}')
ax3.set_xlabel('Tenure (tháng)')
ax3.set_ylabel('Monthly Charges ($)')
ax3.set_title('Tenure vs MonthlyCharges — màu theo Cluster')
ax3.legend()

# --- 4d. PCA 2D visualization ---
ax4 = fig.add_subplot(gs[1, 1])
# [DM] PCA để giảm chiều xuống 2D → visualize cluster trong không gian 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)
explained = pca.explained_variance_ratio_ * 100

for c in range(best_k):
    mask = cluster_labels == c
    ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=cluster_colors[c], alpha=0.3, s=8, label=f'C{c}')
ax4.set_xlabel(f'PC1 ({explained[0]:.1f}% variance)')
ax4.set_ylabel(f'PC2 ({explained[1]:.1f}% variance)')
ax4.set_title(f'PCA 2D — Cluster visualization\n(giải thích {sum(explained):.1f}% variance)')
ax4.legend()

plt.suptitle('Cluster Analysis — Phân nhóm khách hàng', fontsize=14)
plt.tight_layout()
plt.savefig('cluster_02_analysis.png', dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: cluster_02_analysis.png")

# ─────────────────────────────────────────────
# BƯỚC 5: Heatmap profile các cluster
# ─────────────────────────────────────────────
viz_features = ['tenure', 'MonthlyCharges', 'Contract',
                'TechSupport', 'OnlineSecurity', 'SeniorCitizen',
                'Partner', 'Dependents']

# Normalize từng feature về [0,1] để so sánh trực quan
profile_viz = df_analysis.groupby('cluster')[viz_features].mean()
profile_norm = (profile_viz - profile_viz.min()) / (profile_viz.max() - profile_viz.min())

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(profile_norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(viz_features)))
ax.set_xticklabels(viz_features, rotation=45, ha='right')
ax.set_yticks(range(best_k))
ax.set_yticklabels([f'Cluster {c}\n(churn {churn_by_cluster[c]:.0f}%)'
                    for c in range(best_k)])

# Hiện giá trị gốc trên cell
for i in range(best_k):
    for j in range(len(viz_features)):
        val = profile_viz.iloc[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color='black')

plt.colorbar(im, ax=ax, label='Normalized (0=thấp, 1=cao)')
ax.set_title('Heatmap Profile — Đặc trưng từng Cluster', fontsize=13)
plt.tight_layout()
plt.savefig('cluster_03_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Lưu: cluster_03_heatmap.png")

# ─────────────────────────────────────────────
# BƯỚC 6: Lưu artifacts
# ─────────────────────────────────────────────
print("\n── Bước 6: Lưu artifacts ──")

# Lưu cluster labels vào X_train / X_test để dùng làm feature trong Classification
X_full_with_cluster = X_full.copy()
X_full_with_cluster['cluster'] = cluster_labels
X_full_with_cluster.to_csv('X_full_clustered.csv', index=False)

# Lưu model KMeans để dùng trong retrain pipeline
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Lưu profile để báo cáo
profile.to_csv('cluster_profile.csv')

print("  ✅ X_full_clustered.csv — data + cluster label")
print("  ✅ kmeans_model.pkl     — KMeans model")
print("  ✅ cluster_profile.csv  — profile từng cluster")

# ─────────────────────────────────────────────
# TỔNG KẾT — INSIGHT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TỔNG KẾT — [DM INSIGHTS TỪ CLUSTERING]")
print("=" * 60)

for c in range(best_k):
    churn = churn_by_cluster[c]
    t     = profile.loc[c, 'tenure']
    mc    = profile.loc[c, 'MonthlyCharges']
    cont  = profile.loc[c, 'Contract']
    size  = int(profile.loc[c, 'size'])
    print(f"""
  {cluster_names[c]}
    Size: {size:,} khách hàng
    Tenure trung bình: {t:.1f} tháng
    MonthlyCharges TB: ${mc:.1f}
    Contract TB:       {cont:.2f} (0=month-to-month, 2=2 năm)
    Churn rate:        {churn:.1f}%""")

print(f"""
  → Cluster label sẽ được thêm vào làm feature cho Classification
    (giúp model biết khách hàng này thuộc "nhóm nguy hiểm" nào)

  → File 5: Classification — dự đoán churn từng khách hàng
""")