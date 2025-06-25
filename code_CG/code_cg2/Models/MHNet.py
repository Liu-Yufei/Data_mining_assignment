import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义多头注意力机制
# class MultiHeadAttention(nn.Module):
#     def __init__(self, input_dim=10, num_heads=2, output_dim=2):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
#         self.fc = nn.Linear(237 * 2, output_dim)

#     def forward(self, x):
#         x = x.permute(2, 0, 1)  # 将输入从 (batch_size, channels, seq_len) 转换为 (seq_len, batch_size, channels)
#         x, _ = self.attention(x, x, x)
#         x = x.permute(1, 2, 0)  # 将输出从 (seq_len, batch_size, channels) 转换为 (batch_size, channels, seq_len)
#         features = x[:, -1, :]
#         x = self.fc(features)
#         return x
        # return features, x

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads,dropout=0.1)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # 将输入从 (batch_size, channels, seq_len) 转换为 (seq_len, batch_size, channels)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)  # 将输出从 (seq_len, batch_size, channels) 转换为 (batch_size, channels, seq_len)
        return x

# # 定义卷积神经网络模型
# class CNN1DWithAttention(nn.Module):
#     def __init__(self):
#         super(CNN1DWithAttention, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.attention = MultiHeadAttention(input_dim=64, num_heads=4)
#         self.fc1 = nn.Linear(64 * (max_length // 2 // 2), 128)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.attention(x)
#         x = x.view(-1, 64 * (max_length // 2 // 2))
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
    
    # 定义深度模型
# class CNN_BiLSTM(nn.Module):
#     def __init__(self, input_dim=10, hidden_dim=128, output_dim=2, n_layers=4, batch_size=32):
#         super(CNN_BiLSTM, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         # self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
#         # self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True) # 64
#         self.selfatt = MultiHeadAttention(input_dim=118, num_heads=2)
#         self.mlp = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,32), nn.Sigmoid())
#         self.fc = nn.Linear(18880, output_dim)
#         self.fc_feature = nn.Linear(30208, batch_size)

#     def forward(self, x):
#         b, t, d = x.size()
#         n=1
#         x = self.pool1(torch.relu(self.conv1(x)))
#         x = self.pool2(torch.relu(self.conv2(x)))
#         # x = self.pool3(torch.relu(self.conv3(x)))
#         x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_length, num_channels)
#         x_lstm, _ = self.lstm(x) # [32,118,256]
#         x_att = self.selfatt(x_lstm) # [32,118,256]
#         # features = features[:, -1, :]  # 取LSTM最后一个时间步的输出
#         # x_cat = torch.cat((x, x_att), dim=-1)
#         x_mlp = self.mlp(x_att) # [32,118,32]  [32,32]
#         features = x_att.reshape((b,n,-1)).mean(1)
#         features = self.fc_feature(features)
#         x_cat = torch.cat((x,x_mlp),dim=-1).reshape((b,n,-1)).mean(1)
#         x = self.fc(x_cat)
#         return x, features
    
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2, n_layers=2,batch_size=32, seq_length = 474):
        super(CNN_BiLSTM, self).__init__()
        self.conv1_sma = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=1, padding=0),
            nn.Dropout(0.2),
            nn.ReLU()
            )
        self.conv1_med = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.conv1_lar = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.conv1_merg = nn.Sequential(
            nn.Conv1d(in_channels=32*3, out_channels=32, kernel_size=1, padding=0),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.norm = nn.BatchNorm1d(num_features=32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=False, batch_first=True)
        self.selfatt = MultiHeadAttention(input_dim=236, num_heads=4)
        self.mlp = nn.Sequential(nn.Linear(32,64), 
                                #  nn.Dropout(0.2),
                                 nn.ReLU(), 
                                #  nn.BatchNorm1d(num_features=256), 
                                 nn.Linear(64,32), 
                                #  nn.Dropout(0.2),
                                 nn.ReLU(), 
                                #  nn.BatchNorm1d(num_features=256), 
                                #  nn.Linear(128,32), 
                                #  nn.Sigmoid()
                                )
        self.fc = nn.Linear(32*256, output_dim)
        
        self.fc_features = nn.Linear(32*256,128)

    def forward(self, x):
        b, t, d = x.size()
        n=1
        # x = torch.cat([self.conv1_sma(x), self.conv1_med(x), self.conv1_lar(x)], dim=1)
        # x = self.conv1_merg(x)
        x = self.conv1_med(x)

        # x = self.pool(torch.relu(self.conv1(x))) # 取局部特征
        x = self.pool(x) # 取局部特征
        # x = self.norm(x)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_length, num_channels)

        x_lstm, _ = self.lstm(x) # 取全局特征
        x_mlp = self.mlp(x_lstm)
        # x = x+x_mlp
        x = (x+x_mlp)
        # x = (x+x_lstm)

        # x = self.fc(torch.mean(x,dim=1)) # 输出层 (n*1*1)
        features = torch.mean(x,dim=1)
        x = x.reshape((b,n,-1)).mean(1)
        x = self.fc(x)
        x = F.softmax(x) # [,[]]
        # features = 0
        return x, features
# lr:0.00001




# class CNN_BiLSTM(nn.Module):
#     def __init__(self, input_dim=10, hidden_dim=256, output_dim=2, n_layers=2,batch_size=32, seq_length = 474):
#         super(CNN_BiLSTM, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=2, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, padding=1)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
#         # self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=False, batch_first=True)
#         self.selfatt = MultiHeadAttention(input_dim=64, num_heads=2)
#         self.mlp = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Linear(256,128), nn.Sigmoid())
#         # self.mlp = nn.Sequential(nn.Linear(64,128), nn.ReLU(), nn.Linear(128,64), nn.Sigmoid())
#         self.fc = nn.Linear(64*128, output_dim)

#     def forward(self, x):
#         b, t, d = x.size()
#         n=1
#         x = self.pool1(torch.relu(self.conv1(x)))
#         x = self.pool2(torch.relu(self.conv2(x)))
#         x = self.pool3(torch.relu(self.conv3(x)))
#         x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_length, num_channels)
#         # x_lstm, _ = self.lstm(x)
#         x_att = self.selfatt(x)
#         # features = features[:, -1, :]  # 取LSTM最后一个时间步的输出
#         # x_mlp = self.mlp(x_att)
#         # x_mlp = self.mlp(x_lstm)
#         x_mlp = self.mlp(x_att)
#         # x = torch.cat((x, x_mlp), dim=-1).reshape((b,n,-1)).mean(1)
#         x = (x+x_mlp).reshape((b,n,-1)).mean(1)

#         x = self.fc(x)
#         x = F.softmax(x)
#         features = 0
#         return x, features