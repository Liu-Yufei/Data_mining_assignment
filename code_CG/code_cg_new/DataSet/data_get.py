import os
import numpy as np
import pandas as pd
from features_get import get_all_feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

def chagename(cg_path,health_path):
    num = 0
    for filename in os.listdir(cg_path):
        if os.path.isfile(os.path.join(cg_path, filename)):
            new_file_path = str(num).zfill(3)
            num = num + 1
            new_file_path = new_file_path + ".dat"
            old_file_path = os.path.join(cg_path, filename)
            new_file_path = os.path.join(cg_path, new_file_path)
            os.rename(old_file_path, new_file_path)
        else:
            for filenames in os.listdir(os.path.join(cg_path,filename)):
                if os.path.isfile(os.path.join(cg_path,filename,filenames)):
                    new_file_path = str(num).zfill(3)
                    num = num + 1
                    new_file_path = new_file_path + ".dat"
                    old_file_path = os.path.join(cg_path, filename,filenames)
                    new_file_path = os.path.join(cg_path, filename, new_file_path)
                    os.rename(old_file_path, new_file_path)
    for filename in os.listdir(health_path):
        if os.path.isfile(os.path.join(health_path, filename)):
            new_file_path = str(num).zfill(3)
            num = num + 1
            new_file_path = new_file_path + ".dat"
            old_file_path = os.path.join(health_path, filename)
            new_file_path = os.path.join(health_path, new_file_path)
            os.rename(old_file_path, new_file_path)
        else:
            for filenames in os.listdir(os.path.join(health_path,filename)):
                if os.path.isfile(os.path.join(health_path,filename,filenames)):
                    new_file_path = str(num).zfill(3)
                    num = num + 1
                    new_file_path = new_file_path + ".dat"
                    old_file_path = os.path.join(health_path, filename,filenames)
                    new_file_path = os.path.join(health_path, filename, new_file_path)
                    os.rename(old_file_path, new_file_path)

def clean_data(cg_path,health_path):
    # num = 0
    #遍历文件夹
    for filename in os.listdir(cg_path):
        with open(os.path.join(cg_path, filename), 'r') as file:
            original_content = file.read() #
            cut_index = original_content.find("\"2")
            print(filename)
            if cut_index != -1:
                new_content = original_content[:cut_index]
            # 将新内容写回文件
                with open(os.path.join(cg_path, filename), 'w') as file:
                    file.write(new_content)
                print("文件内容已更新。")
    for filename in os.listdir(health_path):
        with open(os.path.join(health_path, filename), 'r') as file:
            original_content = file.read() #
            cut_index = original_content.find("\"2")
            print(filename)
            if cut_index != -1:
                new_content = original_content[:cut_index]
            # 将新内容写回文件
                with open(os.path.join(health_path, filename), 'w') as file:
                    file.write(new_content)
                print("文件内容已更新。")

def clean_data_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path):
    for filename in os.listdir(cag_path):
        with open(os.path.join(cag_path, filename), 'r') as file:
            original_content = file.read() #
            cut_index = original_content.find("\"2")
            print(filename)
            if cut_index != -1:
                new_content = original_content[:cut_index]
            # 将新内容写回文件
                with open(os.path.join(cag_path, filename), 'w') as file:
                    file.write(new_content)
                print("文件内容已更新。")
    for filename in os.listdir(cag_im_path):
        with open(os.path.join(cag_im_path, filename), 'r') as file:
            original_content = file.read()
            cut_index = original_content.find("\"2")
            print(filename)
            if cut_index != -1:
                new_content = original_content[:cut_index]
            # 将新内容写回文件
                with open(os.path.join(cag_im_path, filename), 'w') as file:
                    file.write(new_content)
                print("文件内容已更新。")
    for filename in os.listdir(cag_cancer_path):
        with open(os.path.join(cag_cancer_path, filename), 'r') as file:
            original_content = file.read()
            cut_index = original_content.find("\"2")
            print(filename)
            if cut_index != -1:
                new_content = original_content[:cut_index]
            # 将新内容写回文件
                with open(os.path.join(cag_cancer_path, filename), 'w') as file:
                    file.write(new_content)
                print("文件内容已更新。")
    for filename in os.listdir(jk_path):
        with open(os.path.join(jk_path, filename), 'r') as file:
            original_content = file.read()
            cut_index = original_content.find("\"2")
            print(filename)
            if cut_index != -1:
                new_content = original_content[:cut_index]
            # 将新内容写回文件
                with open(os.path.join(jk_path, filename), 'w') as file:
                    file.write(new_content)
                print("文件内容已更新。")
    for filename in os.listdir(cnag_path):
        with open(os.path.join(cnag_path, filename), 'r') as file:
            original_content = file.read()
            cut_index = original_content.find("\"2")
            print(filename)
            if cut_index != -1:
                new_content = original_content[:cut_index]
            # 将新内容写回文件
                with open(os.path.join(cnag_path, filename), 'w') as file:
                    file.write(new_content)
                print("文件内容已更新。")
    

def get_feature_2(cg_path,health_path):
    # data_3d = []
    y = []
    feature_3d = []
    for filename in os.listdir(cg_path):
        if os.path.isfile(os.path.join(cg_path, filename)):
            with open(os.path.join(cg_path, filename), 'r') as file:
                # df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2,nrows=237)
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                # df.astype(float)
                data = df.to_numpy()
                feature = get_all_feature(data)
                # print(feature.shape)
                # data_3d.append(data)
                feature_3d.append(feature)
                y.append(1)
    for filename in os.listdir(health_path):
        if os.path.isfile(os.path.join(health_path, filename)):
            with open(os.path.join(health_path, filename), 'r') as file:
                # df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2,nrows=237)
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                # df.astype(float)
                data = df.to_numpy()
                feature = get_all_feature(data)
                # feature = feature.flatten() # 拉直的做法
                feature_3d.append(feature)
                y.append(0)
    Y = np.dstack(y)# (1, 1, 554) (row*col*num)
    features = np.dstack(feature_3d) # (18, 11, 554) (row*col*num)
    features = np.transpose(features, (2, 0, 1))
    shape = (features.shape[0], -1)
    features = features.reshape(shape)
    Y = np.transpose(Y, (2, 0, 1))
    Y = Y.reshape(-1)
    # print(Y.shape)
    # print(features.shape)
    return Y,features

def filter_feature(Y,features, index = 3): # 删除Y中=index的数据 以及features中对应的数据
    Y = Y.tolist()
    features = features.tolist()
    for i in range(len(Y)-1,-1,-1):
        if Y[i] == index:
            Y.pop(i)
            features.pop(i)
        elif Y[i] > index:
            Y[i] -= 1

    return np.array(Y),np.array(features)
    
        

def get_feature_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path):
    y = []
    feature_3d = []
    for filename in os.listdir(jk_path):
        if os.path.isfile(os.path.join(jk_path, filename)):
            with open(os.path.join(jk_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                feature = get_all_feature(data)
                feature_3d.append(feature)
                y.append(0)
    for filename in os.listdir(cag_path):
        if os.path.isfile(os.path.join(cag_path, filename)):
            with open(os.path.join(cag_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                feature = get_all_feature(data)
                feature_3d.append(feature)
                y.append(1)
    for filename in os.listdir(cag_im_path):
        if os.path.isfile(os.path.join(cag_im_path, filename)):
            with open(os.path.join(cag_im_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                feature = get_all_feature(data)
                feature_3d.append(feature)
                y.append(2)
    for filename in os.listdir(cag_cancer_path):
        if os.path.isfile(os.path.join(cag_cancer_path, filename)):
            with open(os.path.join(cag_cancer_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                feature = get_all_feature(data)
                feature_3d.append(feature)
                y.append(3)
    for filename in os.listdir(cnag_path):
        if os.path.isfile(os.path.join(cnag_path, filename)):
            with open(os.path.join(cnag_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                feature = get_all_feature(data)
                feature_3d.append(feature)
                y.append(4)
    Y = np.dstack(y)# (1, 1, 554) (row*col*num)
    print(Y.shape)
    features = np.dstack(feature_3d) # (18, 11, 554) (row*col*num)
    features = np.transpose(features, (2, 0, 1))
    shape = (features.shape[0], -1)
    features = features.reshape(shape)
    Y = np.transpose(Y, (2, 0, 1))
    Y = Y.reshape(-1)
    return Y,features

def split_train_test(Y,features,save_path,filter_label = None):

    #     # 数据归一化
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)

    # 创建一个 DataFrame 便于处理
    data = pd.DataFrame(features)
    data['label'] = Y
    

    # 过滤掉标签为 filter_label 的样本
    if filter_label is not None:
        data = data[data['label'] != filter_label]
        # 更新标签
        data.loc[data['label'] == 4, 'label'] = 3

    train_data_list = []
    test_data_list = []
    for y in np.unique(data['label']):
        group = data[data['label'] == y]
        train, test = train_test_split(group, test_size=0.3, random_state=42)
        train_data_list.append(train)
        test_data_list.append(test)
    train_data = pd.concat(train_data_list).sample(frac=1).reset_index(drop=True)  # 打乱顺序
    test_data = pd.concat(test_data_list).sample(frac=1).reset_index(drop=True)  # 打乱顺序
    train_data.to_csv(save_path+'/train.csv', index=False)
    test_data.to_csv(save_path+'/test.csv', index=False)

def get_features(data_map):
    y = []
    feature_3d = []
    for i, class_name in enumerate(data_map.keys()):
        for filename in os.listdir(data_map[class_name]):
            if os.path.isfile(os.path.join(data_map[class_name], filename)):
                with open(os.path.join(data_map[class_name], filename), 'r') as file:
                    df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                    usecols = range(10)
                    data = df.to_numpy()
                    feature = get_all_feature(data)
                    feature_3d.append(feature)
                    y.append(i)
    Y = np.dstack(y)# (1, 1, 554) (row*col*num)
    features = np.dstack(feature_3d) # (18, 11, 554) (row*col*num)
    features = np.transpose(features, (2, 0, 1))
    shape = (features.shape[0], -1)
    features = features.reshape(shape)
    Y = np.transpose(Y, (2, 0, 1))
    Y = Y.reshape(-1)
    return Y,features

if __name__ == '__main__':
    # # 2分类
    # cg_path = "/home/lyf/code_cg/data_2cls copy/CAG"
    # health_path = "/home/lyf/code_cg/data_2cls copy/JK"
    # # chagename()
    # clean_data(cg_path,health_path)
    # Y,features = get_feature_2(cg_path,health_path)

    # 5分类
    cag_path = "/home/lyf/Data/data_5cls/CAG"
    cag_im_path = "/home/lyf/Data/data_5cls/CAG-IM"
    cag_cancer_path = "/home/lyf/Data/data_5cls/CAG-cancer"
    jk_path = "/home/lyf/Data/data_5cls/JK"
    cnag_path = "/home/lyf/Data/data_5cls/CNAG"
    # clean_data_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path)
    # Y,features = get_feature_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path)
    # split_train_test(Y,features,3)
    # Y,features = get_feature_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path)
    # print(features.shape)
    # # 读取训练集和测试集
    # train_data = pd.read_csv('/home/lyf/Data/class_5/train.csv')

    # # 提取特征和标签
    # X_train = train_data.drop('label', axis=1).values
    # y_train = train_data['label'].values

    
    # # CAG vs CNAG
    # save_path = '/home/lyf/Data/CAG_CNAG'
    # data_map = {'CAG':cag_path, 'CNAG':cnag_path}
    save_path = '/home/lyf/Data/CAG_CAG-IM/'
    data_map = {'CAG':cag_path, 'CAG-IM':cag_im_path}
    Y,features = get_features(data_map)
    split_train_test(Y,features,save_path)

    
    # print(get_feature(df).shape)

