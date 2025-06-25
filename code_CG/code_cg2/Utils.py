import torch
import numpy as np
import random
import os
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def save_args(save_path, args):
    if not os.path.exists(save_path): os.makedirs(save_path)
    with open(os.path.join(save_path,'args.txt'),'w') as f:
        for arg, val in sorted(vars(args).items()):
            f.writelines('%s:%s\n'%(str(arg),str(val)))

def set_optimizer(args, model):
    # optimizer
    if args.opt == 'Adam':
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # 优化器 前期冻结ResNet只进行最后一层优化 
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    return optimizer

def set_scheduler(args, optimizer):
    # ReduceLR
    if args.ReduceLR == 'Plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.8,
                                                         min_lr=10e-30)  # 测量损失动态下降学习率
    elif args.ReduceLR == 'LambdaLR':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1),
                                                last_epoch=-1)  # 根据epoch下降学习率
    elif args.ReduceLR == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每训练step_size 个epoch 按 gamma倍数下降学习率
    elif args.ReduceLR == 'MutiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7],
                                                   gamma=0.1)  # 每次遇到milestones里面的epoch 按gamma倍数更新学习率
    elif args.ReduceLR == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95, last_epoch=-1)
    else:
        assert 'ReduceLR error'
    return scheduler

def calculate_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    flag = 0
    # print('classification_report:\n',classification_report(labels, preds)) # 输出每个类别的精确度 召回率 F1值
    cm = confusion_matrix(labels, preds) # 输出混淆矩阵
    if cm[0][0]>cm[0][1] and cm[1][0]<cm[1][1]:
        flag = 1
    return acc, pre, recall, f1, cm, flag
    # print('roc_auc_score:',roc_auc_score(labels, preds, multi_class='ovr', average='macro')) # 输出每个类别的roc_auc值
    
    '''
    if len(set(labels))>1:
        print('roc_auc_score:',roc_auc_score(labels, preds.reshape(-1, 1), multi_class='ovr', average='weighted')) # 输出每个类别的roc_auc值
    else:
        print('ROC AUC Score: Not defined (only one class present in y_true)') # 输出roc_auc值不可用的提示
    '''

    '''
    auc = roc_auc_score(labels, preds_two, multi_class='ovr', average='weighted')
    print('roc_auc_score:',auc) # 输出每个类别的roc_auc值
    cm = confusion_matrix(labels, preds) # 输出混淆矩阵
    print('Confusion Matrix:\n', cm)
    return auc
    '''