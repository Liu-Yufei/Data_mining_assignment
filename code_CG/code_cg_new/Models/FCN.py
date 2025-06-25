import torch.nn as nn
class FCNN(nn.Module):
    def __init__(self,output_dim, input_dim=180):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc4(x)
        features = self.relu(x)
        x = self.dropout(features)
        predicts = self.fc5(x)
        return features, predicts
    

# 定义深度学习模型
class DeepNN(nn.Module):
    def __init__(self, output_dim, input_dim=180, hidden_dim=128):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        features = self.fc3(x)
        x = self.bn3(features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return features, x

# input_dim = 180
# hidden_dim = 128
# output_dim = 5

# model = FCNN(input_dim, hidden_dim, output_dim)