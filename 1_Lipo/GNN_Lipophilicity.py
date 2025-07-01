import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from rdkit import Chem
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import argparse
# Feature Encoding
def one_hot_encoding(value, choices):
    return [int(value == c) for c in choices[:-1]] + [int(value not in choices[:-1])]

def atom_features(atom):
    features = []

    atom_type = one_hot_encoding(atom.GetSymbol(),
                                  ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H', 'B', 'Si', 'Se', 'Other'])
    features += atom_type

    features += [
        atom.GetAtomicNum() / 100,
        atom.GetDegree() / 10,
        atom.GetFormalCharge() / 10,
        int(atom.GetIsAromatic())
    ]
    return torch.tensor(features, dtype=torch.float)


def bond_features(bond):
    bond_type = one_hot_encoding(bond.GetBondType(), [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC])
    return torch.tensor(bond_type + [
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ], dtype=torch.float)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = []
    edge_index = [[], []]
    edge_feats = []

    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)

        # 双向边
        edge_index[0] += [start, end]
        edge_index[1] += [end, start]
        edge_feats += [feat, feat]  # 双向边复制特征

    data = Data(
        x=torch.stack(node_feats),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.stack(edge_feats)
    )
    return data


# Dataset
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

# Model
class GINE(torch.nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=64):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)

        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.squeeze()


# Training
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(pred, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Evaluation
def test(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            preds.append(pred.cpu().numpy())
            labels.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    rmse = root_mean_squared_error(labels, preds)
    return rmse

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = LipophilicityDataset('../Dataset/1_Lipophilicity/LIPO_train.csv')
    test_dataset = LipophilicityDataset('../Dataset/1_Lipophilicity/LIPO_test.csv')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GINE(in_dim=18, edge_dim=6).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    best_rmse = float('inf')
    for epoch in range(1, 501):
        loss = train(model, train_loader, optimizer, device)
        rmse = test(model, test_loader, device)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, RMSE = {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), "1_LIPO_best_model_1.pt")

    print(f"Best RMSE: {best_rmse:.4f}")

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = LipophilicityDataset('../Dataset/1_Lipophilicity/LIPO_train.csv')
    test_dataset = LipophilicityDataset('../Dataset/1_Lipophilicity/LIPO_test.csv')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GINE(in_dim=18, edge_dim=6).to(device)
    model.load_state_dict(torch.load("1_LIPO_best_model.pt", map_location=device))
    model.eval()
    rmse = test(model, test_loader, device)
    print(f"Test RMSE: {rmse:.4f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='0: train, 1: test')
    args = parser.parse_args()
    if args.mode == 0:
        train_model()
    else:
        test_model()