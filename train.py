import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer
from model.ParaTransCNN import ParaTransCNN

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str,
                    default='', help='train root dir for data')
parser.add_argument('--checkpoint_path', type=str,
                    default='', help='weight root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--model_name', type=str,
                    default=" ", help='the name of network')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            "checkpoint_path" : './checkpoints/{}_SGD_{}_{}_{}'.format(args.model_name, args.base_lr,
                                                                               args.max_epochs, args.dataset),
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'AVT': {
            "checkpoint_path" : './checkpoints/{}_SGD_{}_{}_{}'.format(args.model_name, args.base_lr,
                                                                              args.max_epochs, args.dataset),
            'list_dir': './lists/lists_AVT',
            'num_classes': 2,
        },
    }

    args.checkpoint_path = dataset_config[dataset_name]['checkpoint_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.num_classes = dataset_config[dataset_name]['num_classes']

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    net = ParaTransCNN(num_classes=args.num_classes).cuda()
    trainer(args, net)
