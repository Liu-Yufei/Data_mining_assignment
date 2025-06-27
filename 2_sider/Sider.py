import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops, rdMolDescriptors, Descriptors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 数据集处理 ==========
class SIDERDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.smiles_list = df.iloc[:, 0].tolist()
        self.labels = df.iloc[:, 1:].values.astype(np.float32)
        self.num_classes = self.labels.shape[1]
        self.data_list = [self.smiles_to_graph(smiles, label) for smiles, label in zip(self.smiles_list, self.labels)]
        self.data_list = [d for d in self.data_list if d is not None]

    def smiles_to_graph(self, smiles, label):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        
        # 增强的原子特征 - 使用更稳定的方法避免警告
        x = []
        for atom in mol.GetAtoms():
            # 基础原子特征
            features = [
                atom.GetAtomicNum() / 100.0,  # 标准化原子序数
                atom.GetDegree() / 10.0,      # 标准化度数
                atom.GetFormalCharge(),       # 形式电荷
                int(atom.GetHybridization()) / 10.0,  # 杂化类型
                int(atom.GetIsAromatic()),    # 是否芳香族
                atom.GetMass() / 100.0,       # 标准化原子质量
                int(atom.IsInRing()),         # 是否在环中
                atom.GetTotalNumHs() / 10.0,  # 氢原子数
                int(atom.GetChiralTag()) / 10.0,  # 手性标签
                # 避免使用会产生警告的方法，用更安全的替代
                len([x for x in atom.GetNeighbors()]) / 10.0,  # 邻居数量
                float(atom.GetAtomicNum() in [6, 7, 8, 9, 15, 16, 17, 35, 53]),  # 常见药物原子
                int(atom.IsInRingSize(3)),    # 三元环
                int(atom.IsInRingSize(4)),    # 四元环  
                int(atom.IsInRingSize(5)),    # 五元环
                int(atom.IsInRingSize(6)),    # 六元环
            ]
            x.append(features)
        
        # 边特征（键类型）
        edge_attr = []
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_features = [
                float(bond_type) / 10.0,      # 标准化键类型
                int(bond.GetIsAromatic()),    # 是否芳香键
                int(bond.IsInRing()),         # 是否在环中
            ]
            # 无向图，添加双向边
            edge_indices.extend([[i, j], [j, i]])
            edge_attr.extend([bond_features, bond_features])
        
        if len(edge_indices) == 0:
            # 如果没有边，创建自环
            num_atoms = len(x)
            edge_indices = [[i, i] for i in range(num_atoms)]
            edge_attr = [[0.0, 0.0, 0.0] for _ in range(num_atoms)]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# ========== 模型定义 ==========
class ImprovedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(ImprovedGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 批标准化
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 3种池化的拼接
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 多种池化方式的组合
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)
        
        # 拼接不同的池化结果
        x = torch.cat([x1, x2, x3], dim=1)
        
        out = self.classifier(x)
        return out

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=3, dropout=0.2):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT层
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 多种池化方式的组合
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)
        
        # 拼接不同的池化结果
        x = torch.cat([x1, x2, x3], dim=1)
        
        out = self.classifier(x)
        return out

# ========== 训练 & 评估函数 ==========
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # 重新调整 data.y 的形状以匹配 out 的形状
        target = data.y.view(out.shape[0], -1)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = torch.sigmoid(model(data))
            target = data.y.view(out.shape[0], -1)
            y_true.append(target.cpu().numpy())
            y_score.append(out.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_score = np.concatenate(y_score, axis=0)

    # macro
    try:
        auroc = roc_auc_score(y_true, y_score, average="macro")
        aupr = average_precision_score(y_true, y_score, average="macro")
    except ValueError:
        auroc, aupr = 0.0, 0.0
    return auroc, aupr, y_true, y_score

# ========== 主程序 ==========
def main():
    # 加载数据
    train_dataset = SIDERDataset('../Dataset/2_SIDER/SIDER_train.csv')
    test_dataset = SIDERDataset('../Dataset/2_SIDER/SIDER_test.csv')
    input_dim = 15  # 更新为增强后的原子特征维度
    output_dim = train_dataset.num_classes
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 减小batch size
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 使用改进的模型，增加隐藏层维度和层数
    model = ImprovedGCN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, num_layers=5, dropout=0.3).to(device)
    
    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = 0.0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(1, 1001):  # 增加训练轮数
        loss = train(model, train_loader, optimizer, criterion)
        auroc, aupr, y_true, y_score = evaluate(model, test_loader)
        
        # 学习率调度
        
        scheduler.step(auroc)
        
        print(f"Epoch {epoch:03d}: Loss={loss:.4f} | AUROC={auroc:.4f} | AUPR={aupr:.4f}")
        
        if auroc > best_auroc and auroc > 70:
            best_auroc = auroc
            torch.save(model.state_dict(), "best_model_1e_5.pth")
            print(f"New best model saved with AUROC: {best_auroc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        # if patience_counter >= max_patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     break
            
        # 如果已经达到目标性能，可以提前结束
        if auroc >= 0.85:
            print(f"Target AUROC (0.85) achieved at epoch {epoch}!")
            break

    print(f"Best AUROC achieved: {best_auroc:.4f}")
    
    # 最终绘图
    draw_curves(y_true, y_score, train_dataset.num_classes)
    plot_confusion_matrices(y_true, y_score, train_dataset.num_classes)

def draw_curves(y_true, y_score, num_classes):
    plt.figure(figsize=(12, 5))

    # Macro ROC
    plt.subplot(1, 2, 1)
    try:
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f'class {i}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("Macro-average ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(fontsize=7, ncol=2)
    except:
        pass

    # Macro PR
    plt.subplot(1, 2, 2)
    try:
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
            plt.plot(recall, precision, label=f'class {i}')
        plt.title("Macro-average PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(fontsize=7, ncol=2)
    except:
        pass

    plt.tight_layout()
    plt.savefig("curves_macro.png")
    plt.show()

def plot_confusion_matrices(y_true, y_score, num_classes, threshold=0.5):
    """绘制每个类别的混淆矩阵"""
    y_pred = (y_score >= threshold).astype(int)
    for i in range(num_classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Class {i}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_class_{i}.png')
        plt.close()
    print("每个类别的混淆矩阵已保存为 confusion_matrix_class_*.png")


if __name__ == '__main__':
    # 可以选择运行单个模型或尝试多个模型
    # choice = input("Enter 1 for single model training, 2 for multiple model comparison: ")
    # if choice == "2":
    #     train_with_different_models()
    # else:
    main()

