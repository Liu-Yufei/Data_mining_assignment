import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_save(data_dir, split_path, train_ratio=0.7, random_seed=42):
    '''
        将文件按照train_ratio随机划分为train和test，并保存到csv文件中。
        Args:
            data_dir: str, the path to the dataset directory.
            train_ratio: float, the ratio of the training set.
            random_seed: int, the random seed for splitting the dataset.1    
    '''
    np.random.seed(random_seed)
    train_path = split_path + 'dataset_train_split.csv'
    test_path = split_path + 'dataset_test_split.csv'

    if not os.path.exists(train_path):
        os.mknod(train_path)
    if not os.path.exists(test_path):
        os.mknod(test_path)
    
    with open (train_path,'w') as f1:
        with open (test_path,'w') as f2:
            f1.write('filename,label\n')
            f2.write('filename,label\n')
            for dirpath, dirnames, _ in os.walk(data_dir):
                for dirname in dirnames:
                    for _,_, filenames in os.walk(os.path.join(dirpath, dirname)):
                        filenames = np.array(filenames)
                        train_filenames, test_filenames = train_test_split(filenames, train_size=train_ratio, random_state=random_seed)
                        for filename in train_filenames:
                            f1.write(os.path.join(dirpath, dirname, filename) + ','+dirname+'\n')
                        for filename in test_filenames:
                            f2.write(os.path.join(dirpath, dirname, filename) + ','+dirname+'\n')

if __name__ == '__main__':
    split_path = "/home/lyf/code_cg/"
    data_dir = '/home/lyf/code_cg/data_5cls'
    split_and_save(data_dir,split_path)
    