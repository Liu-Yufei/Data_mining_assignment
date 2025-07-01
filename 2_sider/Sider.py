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
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class for SIDER
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
        
        x = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum() / 100.0,  
                atom.GetDegree() / 10.0,      
                atom.GetFormalCharge(),       
                int(atom.GetHybridization()) / 10.0,  
                int(atom.GetIsAromatic()),    
                atom.GetMass() / 100.0,       
                int(atom.IsInRing()),         
                atom.GetTotalNumHs() / 10.0,  
                int(atom.GetChiralTag()) / 10.0,  

                len([x for x in atom.GetNeighbors()]) / 10.0,  
                float(atom.GetAtomicNum() in [6, 7, 8, 9, 15, 16, 17, 35, 53]),
                int(atom.IsInRingSize(3)), 
                int(atom.IsInRingSize(4)), 
                int(atom.IsInRingSize(5)), 
                int(atom.IsInRingSize(6)), 
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
                float(bond_type) / 10.0,  
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),     
            ]

            edge_indices.extend([[i, j], [j, i]])
            edge_attr.extend([bond_features, bond_features])
        
        if len(edge_indices) == 0:

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


class ImprovedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(ImprovedGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x3 = global_add_pool(x, batch)
        
        
        x = torch.cat([x1, x2, x3], dim=1)
        
        out = self.classifier(x)
        return out


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
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

    try:
        auroc = roc_auc_score(y_true, y_score, average="micro")
        aupr = average_precision_score(y_true, y_score, average="micro")
    except ValueError:
        auroc, aupr = 0.0, 0.0
    return auroc, aupr, y_true, y_score


def draw_curves(y_true, y_score, num_classes):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    try:
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        plt.plot(fpr, tpr, label='micro-average ROC')
    except:
        pass
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Micro-average ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(fontsize=9)

    plt.subplot(1, 2, 2)
    try:
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
        plt.plot(recall, precision, label='micro-average PR')
    except:
        pass
    plt.title("Micro-average PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("curves_micro.png")
    plt.show()

def model_train():
    train_dataset = SIDERDataset('../Dataset/2_SIDER/SIDER_train.csv')
    test_dataset = SIDERDataset('../Dataset/2_SIDER/SIDER_test.csv')
    input_dim = 15
    output_dim = train_dataset.num_classes
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 减小batch size
    test_loader = DataLoader(test_dataset, batch_size=32)
    model = ImprovedGCN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, num_layers=5, dropout=0.3).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = 0.0
    max_patience = 15
    
    for epoch in range(1, 1001):
        loss = train(model, train_loader, optimizer, criterion)
        auroc, aupr, y_true, y_score = evaluate(model, test_loader)
        
        scheduler.step(auroc)
        
        print(f"Epoch {epoch:03d}: Loss={loss:.4f} | AUROC={auroc:.4f} | AUPR={aupr:.4f}")
        
        if auroc > best_auroc and auroc > 70:
            best_auroc = auroc
            torch.save(model.state_dict(), "best_model_1.pth")
            print(f"New best model saved with AUROC: {best_auroc:.4f}")
    
    draw_curves(y_true, y_score, train_dataset.num_classes)

def model_test():
    train_dataset = SIDERDataset('../Dataset/2_SIDER/SIDER_train.csv')
    test_dataset = SIDERDataset('../Dataset/2_SIDER/SIDER_test.csv')
    input_dim = 15
    output_dim = train_dataset.num_classes
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 减小batch size
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = ImprovedGCN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, num_layers=5, dropout=0.3).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    auroc, aupr, y_true, y_score = evaluate(model, test_loader)
    print(f" AUROC={auroc:.4f} | AUPR={aupr:.4f}")

    draw_curves(y_true, y_score, train_dataset.num_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=1, help='0: train, 1: test')
    args = parser.parse_args()
    if args.mode == 0:
        model_train()
    else:
        model_test()