import argparse

parser = argparse.ArgumentParser(description='classifier')

parser.add_argument('--task_name', type=str, default='CAGvsCNAG')
parser.add_argument('--seed', type=int, default=1024, help='random seed')
parser.add_argument('--data_path', type=str, default='/home/lyf/Data/CAG_CAG-IM')
parser.add_argument('--Exp_path', type=str, default='/home/lyf/Exp')
parser.add_argument('--gpu_num', type=int, default=1, help='number of gpu to train')
parser.add_argument('--class_num', type=int, default=2)


parser.add_argument('--opt', type=str, default='SGD', help='Adam SGD RMSprop')
parser.add_argument('--ReduceLR', type=str, default='LambdaLR', help='Plateau, LambdaLR, StepLR, MutiStepLR, ExponentialLR')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=27)
parser.add_argument('--lr', type=float, default=0.8)
parser.add_argument('--loss_func', type=str, default='CombinedLoss', help = 'FocalLoss, CombinedLoss, CE, SCLoss')

args = parser.parse_args()

