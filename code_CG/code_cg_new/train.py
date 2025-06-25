import config
from Models.FCN import *
from Utils import *
from Loss_funcs import *
from DataSet.Dataset import *
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='0'

def train(model, train_loader, loss_fn, optimizer,device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        features, outputs = model(inputs)
        loss = loss_fn(features, outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        running_loss += loss.item()
    return running_loss/len(train_loader)

def test(model, test_loader,device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    acc, pre, recall, f1 = print_metrics(all_labels, all_preds)
    return acc, pre, recall, f1


    


if __name__ == '__main__':
    args = config.args
    seed_everything(args.seed)

    args.save_path = os.path.join(args.Exp_path)
    save_args(args.save_path, args)

    device = torch.device('cuda' if args.gpu_num>0 else 'cpu')

    # model = FCNN(args.class_num).to(device)
    model = DeepNN(args.class_num).to(device)
    loss_fn = Loss_func(args, device)
    optimizer = set_optimizer(args, model) # 优化器
    scheduler = set_scheduler(args, optimizer) # 学习率调整器

    # 数据加载
    train_loader, test_loader = load_data(args)


    print('train number:', len(train_loader.dataset))
    print('val number:', len(test_loader.dataset))

    best_acc = 0
    best_pre = 0
    best_recall = 0
    best_f1 = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss = train(model, train_loader, loss_fn, optimizer,device)
        scheduler.step(train_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss:.4f}")
        acc, pre, recall, f1 = test(model, test_loader,device)
        if acc>best_acc:
            print('best result:', acc, pre, recall, f1)
            best_epoch = epoch
            best_acc = acc
            best_pre = pre
            best_f1 = f1
            best_recall = recall
    
    print('best epoch:', best_epoch, best_acc, best_pre, best_recall, best_f1)

