import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import numpy as np

def one_hot_encoding(value, choices):
    return [int(value == c) for c in choices[:-1]] + [int(value not in choices[:-1])]


# 原子特征简单构造
def atom_features(atom):
    features = []

    # One-hot 原子类型（C, O, N, ...），最多20种
    atom_type = one_hot_encoding(atom.GetSymbol(),
                                  ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H', 'B', 'Si', 'Se', 'Other'])
    features += atom_type

    # 价键数、是否芳香等
    features += [
        atom.GetAtomicNum() / 100,               # 归一化
        atom.GetDegree() / 10,
        atom.GetFormalCharge() / 10,
        int(atom.GetIsAromatic())
    ]
    return torch.tensor(features, dtype=torch.float)


# 分子转图结构
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = []
    edge_index = [[], []]

    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [start, end]
        edge_index[1] += [end, start]

    data = Data(
        x=torch.stack(node_feats),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )
    return data

# 自定义数据集
class LipophilicityDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.data_list = []
        for _, row in self.df.iterrows():
            data = mol_to_graph(row['smiles'])
            if data is not None:
                data.y = torch.tensor([row['exp']], dtype=torch.float)
                self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# 模型定义
class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))

        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.squeeze()


# 训练函数
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(pred, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# 测试函数
def test(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            preds.append(pred.cpu().numpy())
            labels.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    rmse = root_mean_squared_error(labels, preds)
    return rmse

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = LipophilicityDataset('./Dataset/1_Lipophilicity/LIPO_train.csv')
    test_dataset = LipophilicityDataset('./Dataset/1_Lipophilicity/LIPO_test.csv')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GIN(in_dim=18).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_rmse = float('inf')
    for epoch in range(1, 501):
        loss = train(model, train_loader, optimizer, device)
        rmse = test(model, test_loader, device)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, RMSE = {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), "best_model.pt")

    print(f"Best RMSE: {best_rmse:.4f}")

if __name__ == "__main__":
    main()
