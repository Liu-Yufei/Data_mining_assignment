import matplotlib.pyplot as plt
import pandas as pd
if __name__ == '__main__':
    path = '/home/lyf/Data/data_5cls/CNAG/2022.05.13黄青山呼气179.dat'
    with open(path, 'r') as file:
        df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=1, usecols=[1])
        data = df.to_dict()
    for i in df.keys():
        df[i].tolist()
        plt.plot(range(len(df[i])),df[i],label = i)
        plt.title('Wave')
        plt.xlabel('time')
        # plt.ylabel('')
        plt.savefig('./wave1')