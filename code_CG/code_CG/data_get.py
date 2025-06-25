import os
import numpy as np
import pandas as pd
from features_get import get_all_feature

import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

# def chagename():
#     num = 0
#     for filename in os.listdir(cg_path):
#         if os.path.isfile(os.path.join(cg_path, filename)):
#             new_file_path = str(num).zfill(3)
#             num = num + 1
#             new_file_path = new_file_path + ".dat"
#             old_file_path = os.path.join(cg_path, filename)
#             new_file_path = os.path.join(cg_path, new_file_path)
#             os.rename(old_file_path, new_file_path)
#         else:
#             for filenames in os.listdir(os.path.join(cg_path,filename)):
#                 if os.path.isfile(os.path.join(cg_path,filename,filenames)):
#                     new_file_path = str(num).zfill(3)
#                     num = num + 1
#                     new_file_path = new_file_path + ".dat"
#                     old_file_path = os.path.join(cg_path, filename,filenames)
#                     new_file_path = os.path.join(cg_path, filename, new_file_path)
#                     os.rename(old_file_path, new_file_path)
#     for filename in os.listdir(health_path):
#         if os.path.isfile(os.path.join(health_path, filename)):
#             new_file_path = str(num).zfill(3)
#             num = num + 1
#             new_file_path = new_file_path + ".dat"
#             old_file_path = os.path.join(health_path, filename)
#             new_file_path = os.path.join(health_path, new_file_path)
#             os.rename(old_file_path, new_file_path)
#         else:
#             for filenames in os.listdir(os.path.join(health_path,filename)):
#                 if os.path.isfile(os.path.join(health_path,filename,filenames)):
#                     new_file_path = str(num).zfill(3)
#                     num = num + 1
#                     new_file_path = new_file_path + ".dat"
#                     old_file_path = os.path.join(health_path, filename,filenames)
#                     new_file_path = os.path.join(health_path, filename, new_file_path)
#                     os.rename(old_file_path, new_file_path)

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

def read_data_from_csv(csv):
    X = []
    y = []
    data_paths = pd.read_csv(csv)
    for index in range(len(data_paths)):
        data_path = data_paths.iloc[index,0]
        with open(data_path, 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                # df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=dict[task_name])

                data = df.to_numpy()
                feature = get_all_feature(data)
                X.append(feature)
                y.append(data_paths.iloc[index,1])
    X = np.dstack(X) # (18, 11, 554) (row*col*num)
    X = np.transpose(X, (2, 0, 1))
    shape = (X.shape[0], -1)
    X = X.reshape(shape)
    y = np.dstack(y)# (1, 1, 554) (row*col*num)
    y = np.transpose(y, (2, 0, 1))
    y = y.reshape(-1)
    return X,y

def read_data_from_csv_HP(csv):
    X = []
    y = []
    data_paths = pd.read_csv(csv)
    for index in range(len(data_paths)):
        data_path = data_paths.iloc[index,0]
        with open(data_path, 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                # df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=dict[task_name])

                data = df.to_numpy()
                feature = get_all_feature(data)
                # 将feature从[10,8]变成[180]:
                feature = feature.flatten()
                # 将data_paths.iloc[index,1]加入feature:
                additional_data = np.array([data_paths.iloc[index, 2], data_paths.iloc[index, 3], data_paths.iloc[index, 4]])
                feature = np.append(feature,additional_data)
                X.append(feature)
                y.append(data_paths.iloc[index,1])
    X = np.dstack(X) # (18, 11, 554) (row*col*num)
    X = np.transpose(X, (2, 1,0))
    # shape = (X.shape[0], -1)
    # X = X.reshape(shape)
    X = X.reshape((X.shape[0], -1))
    y = np.dstack(y)# (1, 1, 554) (row*col*num)
    y = np.transpose(y, (2, 0, 1))
    y = y.reshape(-1)
    return X,y

def get_feature(train_csv,test_csv):
    X_train,y_train = read_data_from_csv(train_csv)
    X_test,y_test = read_data_from_csv(test_csv)
    return X_train, X_test, y_train, y_test

def get_feature_HP(train_csv,test_csv):
    X_train,y_train = read_data_from_csv_HP(train_csv)
    X_test,y_test = read_data_from_csv_HP(test_csv)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # 2分类
    # cg_path = "/home/lyf/code_cg/data_2cls copy/CAG"
    # health_path = "/home/lyf/code_cg/data_2cls copy/JK"
    # chagename()
    # clean_data(cg_path,health_path)
    # Y,features = get_feature_2(cg_path,health_path)

    # 5分类
    # cag_path = "/home/lyf/code_cg/data_5cls/CAG"
    # cag_im_path = "/home/lyf/code_cg/data_5cls/CAG-IM"
    # cag_cancer_path = "/home/lyf/code_cg/data_5cls/CAG-cancer"
    # jk_path = "/home/lyf/code_cg/data_5cls/JK"
    # cnag_path = "/home/lyf/code_cg/data_5cls/CNAG"
    # # clean_data_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path)
    # Y,features = get_feature_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path)
    # Y,features = get_feature_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path)
    # print(features.shape)
    # print(get_feature(df).shape)
    test_csv = "/home/lyf/Data/CAG_CNAG/Conv_HP/test.csv"
    train_csv = "/home/lyf/Data/CAG_CNAG/Conv_HP/train.csv"
    get_feature_HP(train_csv,test_csv)
