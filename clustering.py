import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Buat folder output jika belum ada
output_dir = "dm_output"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("dataset/makanan_indonesia.csv")

# Encode kolom kategorikal
label_cols = ['status_kesehatan', 'jenis']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Mapping kembali jenis ke label (manual mapping sesuai permintaan)
jenis_map = {
    0: 'makanan berat',
    1: 'makanan ringan',
    2: 'cemilan',
    3: 'minuman'
}
df['jenis_label'] = df['jenis'].map(jenis_map)

# Mapping kembali status_kesehatan ke label untuk visualisasi
status_map = {
    0: 'tidak sehat',
    1: 'sedang',
    2: 'sehat'
}
df['status_kesehatan_label'] = df['status_kesehatan'].map(status_map)

# Gunakan kolom fitur numerik
features = ['kalori', 'status_kesehatan', 'jenis']
X = df[features]

# Metode Elbow (elbow_analysis.png)
inertia = []
range_n = range(1, len(X.drop_duplicates()) + 1)
for k in range_n:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range_n, inertia, 'bo-')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.title('Metode Elbow')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/elbow_analysis.png", dpi=300)
plt.close()

# KMeans dengan k = 3
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Mapping cluster ke label status kesehatan
cluster_health_mean = df.groupby('cluster')['status_kesehatan'].mean().sort_values()
ordered_clusters = cluster_health_mean.index.tolist()
cluster_label_map = {
    ordered_clusters[0]: 'tidak sehat',
    ordered_clusters[1]: 'sedang',
    ordered_clusters[2]: 'sehat'
}
df['cluster_label'] = df['cluster'].map(cluster_label_map)

# PCA untuk visualisasi cluster (cluster_composition.png)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster_label', palette='Set2')
plt.title('Visualisasi Cluster berdasarkan Status Kesehatan')
plt.tight_layout()
plt.savefig(f"{output_dir}/cluster_composition.png", dpi=300)
plt.close()

# Visualisasi distribusi status_kesehatan, jenis, dan keterangan_kalori (class_distribution.png)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(data=df, x='status_kesehatan_label', ax=axes[0])
axes[0].set_title('Status Kesehatan')
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x='jenis_label', ax=axes[1])
axes[1].set_title('Jenis Makanan')
axes[1].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x='keterangan_kalori', ax=axes[2])
axes[2].set_title('Keterangan Kalori')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{output_dir}/class_distribution.png", dpi=300)
plt.close()

# Distribusi kalori (calorie_distribution.png)
plt.figure(figsize=(6, 4))
sns.histplot(df['kalori'], bins=10, kde=True)
plt.title('Distribusi Kalori')
plt.tight_layout()
plt.savefig(f"{output_dir}/calorie_distribution.png", dpi=300)
plt.close()

# Pairplot fitur dengan cluster label (feature_visualization.png)
pairplot = sns.pairplot(df[features + ['cluster_label']], hue='cluster_label', palette='Set2')
pairplot.savefig(f"{output_dir}/feature_visualization.png", dpi=300)
plt.close()