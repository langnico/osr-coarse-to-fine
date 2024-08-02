import os
import torch
import random
import numpy as np
import inspect
import argparse

import pandas as pd
from datetime import datetime
import os

from config import project_root_dir

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def init_experiment(args, loss_dir=None, runner_name=None):

    args.cuda = torch.cuda.is_available()

    if args.device == 'None':
        args.device = torch.device("cuda:0" if args.cuda else "cpu")
    else:
        args.device = torch.device(args.device if args.cuda else "cpu")

    print(args.gpus)

    root_dir = args.exp_root

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    def get_unique_id():
        return '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                datetime.now().strftime("%S.%f")[:-3] + ')'

    if loss_dir is None:
        loss_dir = args.loss

    if args.optim == 'sgd':
        optim_dir = ''
    else:
        optim_dir = args.optim

    args.model_path = os.path.join(root_dir, "runs", args.model, args.resnet50_pretrain, args.dataset, loss_dir,
                                   "split_{}".format(args.split_idx), optim_dir, "lr_{}".format(args.lr), "model_{}".format(args.model_id))
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    else:
        print("model_path exists already: ", args.model_path)

    args.log_dir = os.path.join(root_dir, 'log')
    print(f'Experiment saved to: {args.model_path}')

    print(runner_name)
    print(args)

    return args


def accuracy(output, target, topk=(1,)):

    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
