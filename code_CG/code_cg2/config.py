import argparse

parser = argparse.ArgumentParser(description='Deep learning ')

parser.add_argument('--task_name', type=str, default='CAGvsCNAG')
parser.add_argument('--seed', type=int, default=6980, help='random seed')
parser.add_argument('--data_path', type=str, default='/home/lyf/Data/CAG_CAG-IM/Conv_LSTM')
parser.add_argument('--Exp_path', type=str, default='/home/lyf/Exp')
parser.add_argument('--gpu_num', type=int, default=1, help='number of gpu to train')
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--num_workers', default=8, type=int)

# parser.add_argument('--max_len', default=474, type=int)
parser.add_argument('--feature_len', default=512, type=int)



parser.add_argument('--opt', type=str, default='Adam', help='Adam SGD RMSprop')
parser.add_argument('--ReduceLR', type=str, default='Plateau', help='Plateau, LambdaLR, StepLR, MutiStepLR, ExponentialLR')
# Plateau
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default= 0.0001 )
parser.add_argument('--loss_func', type=str, default='CombinedLoss', help = 'FocalLoss, CombinedLoss, CE, SCLoss')
args = parser.parse_args()
