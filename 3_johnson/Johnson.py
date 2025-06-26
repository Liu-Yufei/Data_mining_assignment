import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from rdkit.Chem import rdMolDescriptors

# 1. 读取SMILES数据
df = pd.read_csv('../Dataset/3_johnson/2019-05-15_compound-annotation.csv')  # 假设有一列名为'smiles'
smiles_list = df['SMILES'].tolist()

# 2. 计算分子指纹
morgan_gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=2048)
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

fps = [smiles_to_fp(s) for s in smiles_list]
fps = [fp for fp in fps if fp is not None]
import numpy as np
X = np.array([list(fp) for fp in fps])

# 3. 聚类
n_clusters = 30  # 可调整
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# 4. 聚类指标
sil = silhouette_score(X, labels)
ch = calinski_harabasz_score(X, labels)
db = davies_bouldin_score(X, labels)
print(f'Silhouette: {sil:.3f}, Calinski-Harabasz: {ch:.1f}, Davies-Bouldin: {db:.3f}')

# 5. t-SNE降维可视化
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X)
plt.figure(figsize=(8,6))
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, cmap='tab20', s=5)
plt.title('t-SNE of Johnson Molecules')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.show()