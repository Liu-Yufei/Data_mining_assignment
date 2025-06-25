import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
# JK 373 399
# CNAG 306 460
# CAG-IM 348 474
# CAG 237 450
def clean():
    path = '/home/lyf/Data/data_5cls/CAG'
    cases = os.listdir(path)
    max_len = 0
    min_len = 400
    for case in cases:
        with open(os.path.join(path, case), 'r') as file:
            df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2,usecols=range(10))
            data = df.to_numpy()
            max_len = max(data.shape[0], max_len)
            min_len = min(data.shape[0], min_len)
    print(max_len)
    print(min_len)

def get_data(path):
    cases = os.listdir(path)
    data_paths = []
    for case in cases:
        data_paths.append(os.path.join(path, case))
    return data_paths

def random2list(l1,l2):
    cob = list(zip(l1,l2))
    random.shuffle(cob)
    l1[:], l2[:] = zip(*cob)
    return l1, l2

def save_csv(data_paths, labels, save_path):
    data = pd.DataFrame(data_paths)
    data['label'] = labels
    train_data_list = []
    test_data_list = []
    for y in np.unique(data['label']):
        group = data[data['label']==y]
        train, test = train_test_split(group, test_size=0.1)
        train_data_list.append(train)
        test_data_list.append(test)

    train_data = pd.concat(train_data_list).sample(frac=1).reset_index(drop=True)  # 打乱顺序
    test_data = pd.concat(test_data_list).sample(frac=1).reset_index(drop=True)  # 打乱顺序
    train_data.to_csv(save_path+'/train.csv', index=False)
    test_data.to_csv(save_path+'/test.csv', index=False)

def split_data():
    data_path = '/home/lyf/Data/data_5cls/'
    # save_path = '/home/lyf/Data/CAG_CAG-IM/Conv_LSTM'
    save_path = '/home/lyf/Data/CAG_CNAG/Conv_LSTM'
    datas = []
    labels = []
    for i, case in enumerate(['CAG', 'CNAG']):
        data = get_data(os.path.join(data_path,case))
        datas.extend(data)
        labels.extend([i]*len(data))
    
    save_csv(datas, labels, save_path)
split_data()
    


# clean()
    # 统计长度

    # 