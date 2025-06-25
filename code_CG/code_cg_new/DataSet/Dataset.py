import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_data(args):
    # 读取训练集和测试集
    train_data = pd.read_csv(args.data_path+'/train.csv')
    test_data = pd.read_csv(args.data_path+'/test.csv')


    # 提取特征和标签
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    # 检查每个类别的样本数量
    unique, counts = np.unique(y_train, return_counts=True)
    print("Class distribution before SMOTE:", dict(zip(unique, counts)))

    # # 使用 SMOTE 增强数据
    # # 这里将 k_neighbors 参数调整为 2，以避免因样本不足导致的问题
    # sm = SMOTE(random_state=42,)
    # X_train, y_train = sm.fit_resample(X_train, y_train)

    # # 再次检查每个类别的样本数量
    # unique, counts = np.unique(y_train, return_counts=True)
    # print("Class distribution after SMOTE:", dict(zip(unique, counts)))

    # # 数据增强
    # sm = SMOTE(random_state=42)
    # X_train, y_train = sm.fit_resample(X_train, y_train)

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 创建加权采样器
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    samples_weights = weights[y_train]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


