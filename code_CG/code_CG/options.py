import argparse
from random import seed
import os
# 不动
def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--model', type = str, default = 'DT',help = 'choice from [AB](AdaBoost), [DT](decision tree), [RF](random forest), [SVM](support vector machine), [LR](Logistic regression)')
    parser.add_argument('--s',type = bool, default = True, help = 'show feature importances,1 or 0')
    parser.add_argument('--epoch',type = int, default = 50, help = 'num of epoch')
    parser.add_argument('--seed',type = int, default = 42, help = 'random state')
    parser.add_argument('--cls',type = int, default = 2, help = '2 or 5')

    return parser.parse_args()


# def init_args(args):
#     if not os.path.exists(args.model_path):
#         os.makedirs(args.model_path)
#     if not os.path.exists(args.output_path):
#         os.makedirs(args.output_path)

#     return args
