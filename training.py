import copy
import os
import argparse
import datetime
import time
import pandas as pd
import importlib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import wandb
import numpy as np
from torch.utils.data import DataLoader

from models.wrapper_classes import TimmResNetWrapper, EfficientNetWrapper
from models.model_utils import get_model
from utils.arpl_utils import save_networks, mkdir_if_missing
from core import train, test
from utils.utils import init_experiment, seed_torch, str2bool
from utils.schedulers import get_scheduler
from datasets.open_set_datasets import get_class_splits, get_datasets, create_inat_dataset_funcs, get_dataset_funcs
from config import exp_root


parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cub', help="")
parser.add_argument('--image_size', type=int, default=64)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts')
parser.add_argument("--steps", default=[30, 60, 90], nargs='+', type=int,
                    help="List of epoch indices at which the learning rate is dropped. Must be increasing.")
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
#parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='Softmax')
#parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
#parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='timm_resnet50',)
parser.add_argument('--resnet50_pretrain', type=str, default='scratch',
                        help='Which pretraining to use if --model=timm_resnet50.'
                             'Options are: {iamgenet_timm,}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")
parser.add_argument('--model_id', type=int, default=0, help="model_id of the ensemble member.")
parser.add_argument('--norm_features', default=False, type=str2bool, help='L2 normalize features', metavar='BOOL')

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)

# misc
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--persistent_workers', default=False, type=str2bool,
                    help='This allows to maintain the workers Dataset instances alive between epochs. '
                         'Needed for InatFromTar to keep the cached tar members.', metavar='BOOL')
parser.add_argument('--split_train_val', default=False, type=str2bool, help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--checkpt_freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--train_feat_extractor', default=True, type=str2bool,
                        help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')
# loss
parser.add_argument('--alpha_max', type=float, default=0.25, help="maximum weight of the reversed gradient in SoftmaxMultilabelGRL")


def get_optimizer(args, params_list):
    if args.optim is None:
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=args.lr)
    else:
        raise NotImplementedError
    return optimizer


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()


def main_worker(options, args):

    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # -----------------------------
    # DATALOADERS
    # -----------------------------
    dataloaders = options['dataloaders']

    # -----------------------------
    # MODEL
    # -----------------------------
    print("Creating model: {}".format(options['model']))

    if args.model == 'timm_resnet50':
        wrapper_class = TimmResNetWrapper
    elif args.model == 'timm_resnet50_norm_features':
        wrapper_class = TimmResNetWrapper
        args.norm_features = True
    elif "resnet" in args.model:
        wrapper_class = TimmResNetWrapper
    elif "efficientnet" in args.model:
        wrapper_class = EfficientNetWrapper
    else:
        wrapper_class = None
    net = get_model(args, wrapper_class=wrapper_class, norm_features=args.norm_features)

    feat_dim = args.feat_dim

    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    # -----------------------------
    # GET LOSS
    # -----------------------------
    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    # -----------------------------
    # PREPARE EXPERIMENT
    # -----------------------------
    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    params_list = [{'params': net.parameters()},
                    {'params': criterion.parameters()}]
    
    # Get base network and criterion
    optimizer = get_optimizer(args=args, params_list=params_list)

    # -----------------------------
    # GET SCHEDULER
    # ----------------------------
    scheduler = get_scheduler(optimizer, args)
    start_time = time.time()

    # -----------------------------
    # TRAIN
    # -----------------------------
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        train_loss = train(net, criterion, optimizer, dataloaders['train'], epoch=epoch, **options)
        wandb.log({'train_CE': train_loss}, step=epoch)

        # VAL
        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:

            save_checkpoint = epoch % options['checkpt_freq'] == 0 or epoch == options['max_epoch'] - 1
            save_val_test_output = epoch == options['max_epoch'] - 1

            print("==> Val", options['loss'])
            results_val = test(net, criterion, dataloaders['val'], outloader=None, epoch=epoch,
                               return_outputs=save_val_test_output, **options)
            print("Epoch {}: Acc (%): {:.3f}".format(epoch, results_val['ACC']))

            print("==> Test", options['loss'])
            results_test = test(net, criterion, dataloaders['test_known'], outloader=dataloaders['test_unknown'],
                                epoch=epoch, return_outputs=save_val_test_output, **options)

            print("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t AUPR (%): {:.3f}".format(epoch,
                                                                                              results_test['ACC'],
                                                                                              results_test['AUROC'],
                                                                                              results_test['OSCR'],
                                                                                              results_test['AUPR']))
            if save_checkpoint:
                name = file_name.split('.')[0]+'_{}'.format(epoch)

                save_networks(net, args.model_path, name,
                              options['loss'],
                              criterion=criterion)

            if save_val_test_output:
                # save val outputs
                mkdir_if_missing(os.path.join(args.model_path, 'val'))
                output_filepath = '{}/val/{}_{}.npz'.format(args.model_path, name, options['loss'])
                np.savez(file=output_filepath, **results_val.pop('output_dict'))

                # save test outputs
                mkdir_if_missing(os.path.join(args.model_path, 'test'))
                output_filepath = '{}/test/{}_{}.npz'.format(args.model_path, name, options['loss'])
                np.savez(file=output_filepath, **results_test.pop('output_dict'))

            # ----------------
            # LOG
            # ----------------
            # log to wandb
            results_val_log = copy.deepcopy(results_val)
            for key in list(results_val_log.keys()):
                new_key = "{}_{}".format("val", key)
                results_val_log[new_key] = results_val_log.pop(key)
            wandb.log(results_val_log, step=epoch)

            results_test_log = copy.deepcopy(results_test)
            for key in list(results_test_log.keys()):
                new_key = "{}_{}".format("test", key)
                results_test_log[new_key] = results_test_log.pop(key)
            wandb.log(results_test_log, step=epoch)

        # -----------------------------
        # STEP SCHEDULER
        # ----------------------------
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(results_val['ACC'], epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    # merge val (closed-set ACC) and test (OSR) metrics
    results = copy.deepcopy(results_val_log)
    results.update(results_test)

    return results


if __name__ == '__main__':

    args = parser.parse_args()
    args_input = copy.deepcopy(args)

    # ------------------------
    # Update parameters with default hyperparameters if specified
    # ------------------------

    args.exp_root = exp_root
    args.epochs = args.max_epoch
    img_size = args.image_size
    results = dict()

    for i in range(1):

        # ------------------------
        # INIT
        # ------------------------
        if args.feat_dim is None:
            args.feat_dim = 128 if args.model == 'classifier32' else 2048

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset)

        img_size = args.image_size

        args.save_name = '{}_{}_{}'.format(args.model, args.seed, args.dataset)
        runner_name = os.path.dirname(__file__).split("/")[-2:]
        
        # set args.model_path
        if args.loss in ['SoftmaxMultilabelGRL']:
            loss_suffix = f"_a{args.alpha_max:.2f}"
        else:
            loss_suffix = ""
        args = init_experiment(args, loss_dir=f"{args.loss}{loss_suffix}", runner_name=runner_name)

        # ------------------------
        # SEED
        # ------------------------
        seed_torch(args.seed)

        # ------------------------
        # DATASETS
        # ------------------------
        # append inat21 datasets to global datasets dict (overwrite return_multilabel)
        args.return_multilabel = args.loss in ["SoftmaxMultilabel", "SoftmaxMultilabelGRL"]
        create_inat_dataset_funcs(dataset_funcs_dict=get_dataset_funcs, return_multilabel=args.return_multilabel)

        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)

        # ------------------------
        # RANDAUG HYPERPARAM SWEEP
        # ------------------------
        if args.transform == 'rand-augment':
            if args.rand_aug_m is not None:
                if args.rand_aug_n is not None:
                    if hasattr(datasets['train'], 'dataset'):
                        # If datasets are object of type Subset
                        datasets['train'].dataset.transform.transforms[0].m = args.rand_aug_m
                        datasets['train'].dataset.transform.transforms[0].n = args.rand_aug_n
                    else:
                        datasets['train'].transform.transforms[0].m = args.rand_aug_m
                        datasets['train'].transform.transforms[0].n = args.rand_aug_n

        # ------------------------
        # DATALOADER
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers,
                                        persistent_workers=args.persistent_workers)

        # ------------------------
        # SAVE PARAMS
        # ------------------------
        # set number of output classes
        if args.loss in ["SoftmaxMultilabel", "SoftmaxMultilabelGRL"]:
            num_output = datasets['train'].dataset.num_multilabel_output
        elif "inat" in args.dataset:
            num_output = datasets['train'].dataset.num_classes
        else:
            num_output = len(args.train_classes)
        print("num_output: ", num_output)
        
        options = vars(args)
        options.update(
            {
                'item':     i,
                'known':    args.train_classes,
                'unknown':  args.open_set_classes,
                'img_size': img_size,
                'dataloaders': dataloaders,
                'num_classes': num_output
            }
        )

        file_name = options['dataset'] + '.csv'
        print('result path:', os.path.join(args.model_path, file_name))

        # init wandb    
        run = wandb.init(
            entity="osr-multi-label",
            project="osr-closed-set",
            name=f"{options['dataset']}_model{options['model_id']}_lr{options['lr']}_{options['loss']}{loss_suffix}",
            config=options,
            reinit=True,
            mode="online"  # set to: "online", "offline", "disabled" 
        )

        # ------------------------
        # TRAIN
        # ------------------------
        res = main_worker(options, args)

        # ------------------------
        # LOG
        # ------------------------
        res['unknown'] = args.open_set_classes
        res['known'] = args.train_classes
        res['ID'] = args.log_dir.split("/")[-1]
        results[str(args.split_idx)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.model_path, file_name), mode='a', header=False)

        # update wandb config with new parameters
        wandb.config.update(options)

        run.finish()

