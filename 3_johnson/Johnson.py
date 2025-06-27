import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../Dataset/3_johnson/2019-05-15_compound-annotation.csv')  # 修改为你的路径
smiles_list = df['SMILES'].tolist()

# 分子描述符提取
def mol_to_desc(mol):
    return [desc[1](mol) for desc in Descriptors._descList]

valid_smiles = []
desc_vectors = []
for s in smiles_list:
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        try:
            desc = mol_to_desc(mol)
            if all(np.isfinite(desc)):
                desc_vectors.append(desc)
                valid_smiles.append(s)
        except:
            continue

X = np.array(desc_vectors)
X_scaled = StandardScaler().fit_transform(X)

# t-SNE降维
X_reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

# 聚类优化搜索
best_score = -np.inf
best_result = {}

# for n_clusters in [60, 70, 80, 90, 100]:
kmeans = KMeans(n_clusters=60, random_state=42)
labels = kmeans.fit_predict(X_reduced)

sil = silhouette_score(X_reduced, labels)
ch = calinski_harabasz_score(X_reduced, labels)
db = davies_bouldin_score(X_reduced, labels)

if sil > 0.3 and ch > 10000 and db < 1.0:
    score = sil * (ch / 10000) / db
    if score > best_score:
        best_score = score
        best_result = {
            # 'n_clusters': n_clusters,
            'labels': labels,
            'kmeans': kmeans,
            'sil': sil,
            'ch': ch,
            'db': db
        }

# 可视化与代表分子输出
if best_result:
    # n_clusters = best_result['n_clusters']
    labels = best_result['labels']
    kmeans = best_result['kmeans']
    sil = best_result['sil']
    ch = best_result['ch']
    db = best_result['db']

    # t-SNE 聚类可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab20', s=10)
    plt.title(f't-SNE of Johnson Molecules (KMeans)\nSilhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.savefig('johnson_tsne_kmeans_optimized.png')
    # plt.show()

    # 代表性分子（每簇中心点最近分子）
    centers, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_reduced)
    representative_smiles = [valid_smiles[i] for i in centers]
    rep_df = pd.DataFrame({'SMILES': representative_smiles})
    rep_df.to_csv('representative_molecules_tsne.csv', index=False)
