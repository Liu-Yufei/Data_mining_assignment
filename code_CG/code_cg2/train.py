import config
from torch.utils.data import DataLoader
from DataSet.DataSet import *
from Utils import *
from Loss_funcs import *
from tqdm import tqdm
from Models.MHNet import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def train(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels, _ in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        # outputs = model(inputs)
        outputs, features = model(inputs)
        _, preds = torch.max(outputs, 1)
        # loss = loss_fn(outputs, labels)
        loss = loss_fn(features, outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        running_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    acc, pre, recall, f1, cm, _ = calculate_metrics(all_labels, all_preds)
    return running_loss/len(train_loader), acc, pre, recall, f1, cm

def find_hard_sample(all_labels, all_outputs, all_cases, worst_sample):
    losses = []
    for i in range(len(all_labels)):
        label_tensor = torch.tensor([all_labels[i]], dtype=torch.long)  # 转换为长整型张量
        output_tensor = all_outputs[i].cpu().unsqueeze(0)  # 转换为2D张量
        loss = F.cross_entropy(output_tensor, label_tensor).item()
        losses.append((all_cases[i], loss))
    
    # 按损失排序
    losses.sort(key=lambda x: x[1], reverse=True)

    # 记录损失最大的10个样本
    worst_samples = losses[:10]
    for sample_id, loss in worst_samples:
        if sample_id not in worst_sample.keys():
            worst_sample[sample_id] = 1
        else:
            worst_sample[sample_id] += 1




def test(model, test_loader,device):
    model.eval()
    all_preds = []
    all_labels = []
    all_cases = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels, case in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # _, outputs = model(inputs)
            outputs,_ = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(outputs)
            all_cases.extend(case)
    acc, pre, recall, f1, cm, flag = calculate_metrics(all_labels, all_preds)
    # 输出：
    # if cm[0][0]>cm[0][1] and cm[1][0]<cm[1][1] and acc>0.60: # 在对角线是最大值的时候输出
    # if True:
    if flag == 1:
        # print('accuracy:',acc) # 输出准确率
        # print('precision:',pre) # 输出每个类别的精确度
        # print('recall:',recall) # 输出每个类别的召回率
        # print('f1:',f1) # 输出每个类别的F1值
        print('Confusion Matrix:\n', cm)
    return acc, pre, recall, f1, cm, flag, all_labels, all_outputs, all_cases

if __name__ == '__main__':
    # dict = {'CAGvsCNAG':[1,6,7,8,9]} # 用于确定要选择哪些通道
    dict = {'CAGvsCNAG':[0,1,2,3,4,5,6,7,8,9]} # 用于确定要选择哪些通道

    args = config.args
    seed_everything(args.seed)
    args.save_path = os.path.join(args.Exp_path)
    save_args(args.save_path, args)

    device = torch.device('cuda' if args.gpu_num>0 else 'cpu')

    model = CNN_BiLSTM(input_dim = len(dict[args.task_name]),batch_size=args.batch_size).to(device)
    # model = MultiHeadAttention().to(device)

    loss_fn = Loss_func(args, device)
    # import torch.nn as nn
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = set_optimizer(args, model) # 优化器
    scheduler = set_scheduler(args, optimizer) # 学习率调整器

    # 数据加载
    train_loader = DataLoader(dataset=ImageFolder(args, 'train',dict), batch_size=args.batch_size*args.gpu_num, num_workers=args.num_workers,shuffle=True )
    test_loader = DataLoader(dataset=ImageFolder(args, 'test',dict), batch_size=args.batch_size*args.gpu_num, num_workers=args.num_workers,shuffle=True )

    print('train number:', len(train_loader.dataset))
    print('val number:', len(test_loader.dataset))

    best_acc = 0
    best_pre = 0
    best_recall = 0
    best_f1 = 0
    best_epoch = 0
    loss_all = []
    worst_sample = {}
    for epoch in range(args.epochs):
        # print("=======Epoch:{}=======".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        # if epoch == 200:
        #     print('here')
        train_loss,acc_train, pre, recall, f1, cm = train(model, train_loader, loss_fn, optimizer,device)
        loss_all.append(train_loss)
        # print('accuracy:',acc_train) # 输出准确率
        # print('precision:',pre) # 输出每个类别的精确度
        # print('recall:',recall) # 输出每个类别的召回率
        # print('f1:',f1) # 输出每个类别的F1值
        # print('Confusion Matrix:\n', cm)
        scheduler.step(train_loss)
        acc_test, pre, recall, f1, cm, flag, all_labels, all_outputs, all_cases = test(model, test_loader, device)
        if epoch > 30:
            find_hard_sample(all_labels, all_outputs, all_cases, worst_sample)
        if acc_test>best_acc and flag==1:
            print('best result:', acc_test, pre, recall, f1)
            print(cm)
            best_epoch = epoch
            best_acc = acc_test
            best_pre = pre
            best_f1 = f1
            best_recall = recall
            best_cm = cm
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.4f}, Acc_train:{acc_train:.4f}, Acc_test:{acc_test:.4f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
    
    # for name, param in model.named_parameters():
    #     print(f'Layer: {name}')
    #     print(f'\tSize: {param.size()}')
    #     print(f'\tValues: {param}')
        

    print('best epoch:', best_epoch, best_acc, best_pre, best_recall, best_f1)
    print(best_cm)
    plt.figure()
    plt.plot(range(args.epochs),loss_all)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./loss')

    sorted_keys = sorted(worst_sample, key=worst_sample.get, reverse=True)
    for key in sorted_keys:
        print(key, worst_sample[key])
