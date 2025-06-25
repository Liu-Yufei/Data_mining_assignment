import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config
from scipy.interpolate import interp1d
# from features_get import *
def get_data(path,task_name,dict):
    with open(path, 'r') as file:
        # df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
        df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=dict[task_name])

        data = df.to_numpy()
    return data


class ImageFolder(data.Dataset):
    def __init__(self, args, split,dict):
        super().__init__()
        self.feature_len = args.feature_len
        self.scaler = StandardScaler()
        self.task_name = args.task_name
        self.data_df = pd.read_csv(args.data_path+'/%s.csv'%split)
        self.dict = dict

    def process_feat(self, feat, length):
        new_feat = np.zeros((feat.shape[0],length )).astype(np.float32)

        r = np.linspace(0, len(feat), length+1, dtype=np.int32)
        for i in range(length):
            if r[i]!=r[i+1]:
                # new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
                new_feat[:,i] = np.mean(feat[:,r[i]:r[i+1]], 0)
            else:
                # new_feat[i,:] = feat[r[i], :]
                new_feat[:,i] = feat[ :,r[i]]
        return new_feat
    
    def __getitem__(self, index):
        data_path = self.data_df.iloc[index, 0]
        label = self.data_df.iloc[index, 1]
        # 读区bat文件
        task_name = self.task_name

        data = get_data(data_path,task_name,self.dict).reshape(len(self.dict[task_name]), -1) # need to change reshape num by task name

        # # 长度补全
        if self.feature_len>0 and len(data[1]) < self.feature_len: # 如果最大长度>0按照最大长度补全
            data = np.pad(data, ((0, 0), (0, self.feature_len - data.shape[1])), 'constant')
        data = self.process_feat(data, self.feature_len)
        # 标准化数据
        data = self.scaler.fit_transform(data.T).T

        # 插值
        # x = np.linspace(0, 1, data.shape[1])
        # f = interp1d(x, data, kind='linear', axis=1)
        # x_new = np.linspace(0, 1, self.feature_len)
        # data = f(x_new)
        # data = self.scaler.fit_transform(data.T).T

        # feature = get_all_feature(data)

        
        # 转换为张量
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor, ''.join(data_path.split('/')[-2:]).strip('.dat')
    
    def __len__(self):
        return len(self.data_df)
    
if __name__ == '__main__':
    dict = {'CAGvsCNAG':[1,6,7,8,9]} # 用于确定要选择哪些通道

    args = config.args
    dataset = ImageFolder(args, 'train',dict)
    print(dataset[10])