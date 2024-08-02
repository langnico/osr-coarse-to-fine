import sys
import os
import torch
import argparse
import numpy as np
import json
import torchmetrics
import importlib
from torch.utils.data import DataLoader

from core import test, embed_features
from utils.test_utils import ModelTemplate
from utils.utils import strip_state_dict, str2bool
from models.model_utils import get_model
from models.wrapper_classes import TimmResNetWrapper, EfficientNetWrapper
from datasets.inat2021 import tax_id_to_supertax_id
from datasets.open_set_datasets import get_class_splits, get_datasets, create_inat_dataset_funcs, get_dataset_funcs

from config import root_model_path, root_criterion_path


def print_state_dict(state_dict):
    for k in list(state_dict.keys()):
        print(k)


def load_state_dict(model, args, path):
    if args.loss == 'ARPLoss':

        state_dict_list = [torch.load(p) for p in path]
        model.load_state_dict(state_dict_list)

    else:

        state_dict = strip_state_dict(torch.load(path[0]))
        model.load_state_dict(state_dict)

    return model


def load_models(path, args, wrapper_class=None):

    model = get_model(args, wrapper_class=wrapper_class, evaluate=True)

    if args.loss == 'ARPLoss':

        state_dict_list = [torch.load(p) for p in path]
        model.load_state_dict(state_dict_list)

    else:

        state_dict = strip_state_dict(torch.load(path[0]))  # strip_key='module.'
        state_dict = strip_state_dict(state_dict, strip_key='resnet.')
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--device', default='None', type=str, help='Which GPU to use')
    parser.add_argument('--osr_mode', default='max_softmax', type=str, help='{entropy, max_softmax}')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dir_experiment', default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--criterion_path', default=None)

    # Model
    parser.add_argument('--model', type=str, default='classifier32')
    parser.add_argument('--loss', type=str, default='Softmax')
    parser.add_argument('--feat_dim', default=128, type=int)
    parser.add_argument('--max_epoch', default=599, type=int)
    parser.add_argument('--cs', default=False, type=str2bool)
    parser.add_argument('--use_default_parameters', default=False, type=str2bool,
                        help='Set to True to use optimized hyper-parameters from paper', metavar='BOOL')
    parser.add_argument('--norm_features', default=False, type=str2bool, help='L2 normalize features', metavar='BOOL')

    # Data params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', type=str, default='tinyimagenet')
    parser.add_argument('--dataset_train', type=str, default='tinyimagenet')
    parser.add_argument('--transform', type=str, default='rand-augment')
    parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
    parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing."
                                                                            "No smoothing if None or 0")
    parser.add_argument('--temp', type=float, default=1.0, help="temp")

    # Eval args
    parser.add_argument('--use_balanced_eval', default=True, type=str2bool)
    parser.add_argument('--use_softmax', default=False, type=str2bool)  # OBSOLETE
    parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')

    parser.add_argument('--sweep_image_size', default=False, type=str2bool)

    # loss
    #parser.add_argument('--alpha_max', type=float, default=0.25, help="maximum weight of the reversed gradient in SoftmaxMultilabelGRL and SoftmaxMultilabelHAL")


    # Train params
    args = parser.parse_args()

    if args.sweep_image_size:
        args.image_size_list = args.image_size + np.arange(-10, 11, 1) * 4
    else:
        args.image_size_list = [args.image_size]

    args.feat_dim = 2048

    options = vars(args)
    options['use_gpu'] = torch.cuda.is_available()
    print(options)
    device = torch.device('cuda:0')

    if args.cs:
        dataset_cs = args.dataset_train + 'cs'
    else:
        dataset_cs = args.dataset_train


    if args.criterion_path is None:
        args.v = args.model_path.replace('.pth', '_criterion.pth')
    print("model_path: ", args.model_path)
    print("criterion_path: ", args.criterion_path)


    # ------------------------
    # HOOKS to extract intermediate features
    # ------------------------
    # Names of layers of which the activation maps shall be saved as outputs
    hook_names = None # e.g. ['layer4.2.conv3', 'layer4.2.bn3']
 
    all_test_metrics = []

    for args.image_size in args.image_size_list:
        if args.sweep_image_size:
            args.save_dir = os.path.join(args.model_path.split('checkpoints/')[0], 'test', args.dataset, "imagesize_{}".format(args.image_size))
        else:
            args.save_dir = os.path.join(args.model_path.split('checkpoints/')[0], 'test', args.dataset)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        print(args.save_dir)

        # ------------------------
        # DATASETS
        # ------------------------
        # append inat21 datasets to global datasets dict (overwrite return_multilabel)
        args.return_multilabel = args.loss in ["SoftmaxMultilabel", "SoftmaxMultilabelGRL"]
        create_inat_dataset_funcs(dataset_funcs_dict=get_dataset_funcs, return_multilabel=args.return_multilabel)

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset)

        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                image_size=args.image_size, balance_open_set_eval=args.use_balanced_eval,
                                split_train_val=False, open_set_classes=args.open_set_classes)

        # ------------------------
        # DATALOADERS
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            # no need to shuffle the train dataset
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=False, sampler=None, num_workers=args.num_workers)

        # -----------------------------
        # MODEL
        # -----------------------------
        # set number of output classes
        if args.loss in ["SoftmaxMultilabel", "SoftmaxMultilabelGRL"]:
            num_output = datasets['train'].dataset.num_multilabel_output
        elif "inat" in args.dataset:
            num_output = datasets['train'].dataset.num_classes
        else:
            num_output = len(args.train_classes)
        print("num_output: ", num_output)
        options['num_classes'] = num_output

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
        model = get_model(args, wrapper_class=wrapper_class, norm_features=args.norm_features)

        model.eval()
        model = model.to(device)

        # -----------------------------
        # GET LOSS
        # -----------------------------
        Loss = importlib.import_module('loss.' + options['loss'])
        criterion = getattr(Loss, options['loss'])(**options)
        criterion = criterion.to(device)

        # -----------------------------
        # LOAD MODEL WEIGHTS
        # -----------------------------
        model = load_state_dict(model, args, path=[args.model_path, args.criterion_path])

        # ------------------------
        # EMBED TRAINING DATA 
        # ------------------------

        # Compute embeddings for training data for KNN
        filepath_model_outputs_train = os.path.join(os.path.dirname(args.save_dir), 'model_outputs_train.npz')
        if not os.path.exists(filepath_model_outputs_train):
            print("embedding training data...")
            results_train = embed_features(model, criterion, dataloaders['train'], **options)
            print("results_train['output_dict']['feat_k'].shape", results_train['output_dict']['feat_k'].shape)
            print("results_train['output_dict']['labels'].shape", results_train['output_dict']['labels'].shape)

            # save test outputs
            np.savez(file=filepath_model_outputs_train, **results_train.pop('output_dict'))
            print("saved training data embeddings to: ", filepath_model_outputs_train)
            del(results_train)
        else:
            print("training data is already embedded.")
        
        # ------------------------
        # EVALUATE
        # ------------------------

        # Evaluate TEST SEEN Acc on test_known_all pooled to all super-classes
        results_test_known = test(model, criterion, dataloaders['test_known_all'], outloader=None,
                                  epoch=None, return_outputs=True, log_wandb=False, hook_names=hook_names, **options)

        # get enumerated class prediction
        results_test_known['output_dict']['preds_class_enum_k'] = np.argmax(results_test_known['output_dict']['preds_k'], axis=1)
        if args.loss in ["SoftmaxMultilabel", "SoftmaxMultilabelGRL"]:
            
            # select logits of training ids
            results_test_known['output_dict']['preds_k'] = results_test_known['output_dict']['preds_k'][:, args.train_classes]
            # recompute softmax probability using only training ids
            results_test_known['output_dict']['preds_k_probs'] = torch.nn.Softmax(dim=-1)(torch.tensor(results_test_known['output_dict']['preds_k'])).numpy()
        
            # enumerated class equals original class id
            results_test_known['output_dict']['preds_class_id_k'] = results_test_known['output_dict']['preds_class_enum_k']
            results_test_known['output_dict']['labels_id'] = results_test_known['output_dict']['labels']
        else:
            # get original tax id prediction (used to pool to coarser taxon ids)
            results_test_known['output_dict']['preds_class_id_k'] = np.array([datasets['test_known_all'].dataset.target_dict_enumerated_to_tax_id[i] for i in results_test_known['output_dict']['preds_class_enum_k']])
            results_test_known['output_dict']['labels_id'] = np.array([datasets['test_known_all'].dataset.target_dict_enumerated_to_tax_id[i] for i in results_test_known['output_dict']['labels']])

        print("results_test_known['output_dict']['preds_k'].shape", results_test_known['output_dict']['preds_k'].shape)
        print("results_test_known['output_dict']['labels'].shape", results_test_known['output_dict']['labels'].shape)
        print(list(results_test_known.keys()))

        # compute accuracy of pooled super-classes (i.e. pool both targets and predictions to coarser taxon)
        if all(x in args.dataset_train for x in ['inat', '1hop']):
            train_taxon = args.dataset_train.split('-')[3]
            print("train_taxon: ", train_taxon)
            test_taxon_list = ['id', 'genus', 'family', 'order']
            test_taxon_list = test_taxon_list[test_taxon_list.index(train_taxon):]
            for test_taxon in test_taxon_list:
                print("Pooling to test_taxon: ", test_taxon)
                # get predicted_id from logits and pool to test_taxon

                pred_ids = tax_id_to_supertax_id(tax_ids=results_test_known['output_dict']['preds_class_id_k'],
                                                 tax_level_source=train_taxon, tax_level_target=test_taxon,
                                                 taxonomy=datasets['test_known_all'].dataset.taxonomy)

                target_ids = tax_id_to_supertax_id(tax_ids=results_test_known['output_dict']['labels_id'],
                                                   tax_level_source=train_taxon, tax_level_target=test_taxon,
                                                   taxonomy=datasets['test_known_all'].dataset.taxonomy)

                print("len(pred_ids)", len(pred_ids))
                print("len(target_ids)", len(target_ids))

                num_classes = len(set(target_ids))
                print("num_classes: ", num_classes)

                accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
                # balanced accuracy (average acc per category)
                accuracy_macro = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes,
                                                                                top_k=1, average='macro',
                                                                                multidim_average='global',
                                                                                validate_args=True)
                # enumerate ids from 0, C-1
                target_dict_id_to_enumerated = {id: i for i, id in enumerate(sorted(set(target_ids)))}
                target_ids_enum = [target_dict_id_to_enumerated[id] for id in target_ids]
                pred_ids_enum = [target_dict_id_to_enumerated[id] for id in pred_ids]
                print("num_classes enumerated: ", len(set(target_ids_enum)))
                # to tensor
                pred_ids_enum = torch.tensor(pred_ids_enum)
                target_ids_enum = torch.tensor(target_ids_enum)

                results_test_known['ACC_{}'.format(test_taxon)] = float(accuracy(pred_ids_enum, target_ids_enum)) * 100.
                results_test_known['ACCB_{}'.format(test_taxon)] = float(accuracy_macro(pred_ids_enum, target_ids_enum)) * 100.

        # Evaluate OSR performance on balanced test_known
        results_test = test(model, criterion, dataloaders['test_known'], outloader=dataloaders['test_unknown'],
                            epoch=None, return_outputs=True, log_wandb=False, hook_names=hook_names, **options)

        # save labels as enumerated and original ids
        if args.loss in ["SoftmaxMultilabel", "SoftmaxMultilabelGRL"]:
            # select logits of training ids
            results_test['output_dict']['preds_k'] = results_test['output_dict']['preds_k'][:, args.train_classes]
            results_test['output_dict']['preds_u'] = results_test['output_dict']['preds_u'][:, args.train_classes]
            # recompute softmax probability using only training ids
            results_test['output_dict']['preds_k_probs'] = torch.nn.Softmax(dim=-1)(torch.tensor(results_test['output_dict']['preds_k'])).numpy()
            results_test['output_dict']['preds_u_probs'] = torch.nn.Softmax(dim=-1)(torch.tensor(results_test['output_dict']['preds_u'])).numpy()
        else:
            # enumerated
            results_test["output_dict"]["labels_enum"] = results_test["output_dict"]["labels"]
            # original ids
            results_test["output_dict"]["labels"] = np.array([datasets['test_known_all'].dataset.target_dict_enumerated_to_tax_id[i] for i in results_test["output_dict"]["labels"]])


        # Evaluate Multilabel (Multi-Tax-Pred) on test_known_all
        if args.loss in ["SoftmaxMultilabel"]:
            print("evaluating SoftmaxMultilabel predictions for each taxon...")
            if args.dataset_train == "inat21-id-1hop":
                results_test["Multi-Tax-Pred"] = {}
                test_taxon_list = ['id', 'genus', 'family', 'order']
                for test_taxon in test_taxon_list:
                    print("Evaluating SoftmaxMultilabel for test_taxon: ", test_taxon)
                    # return logits and labels masked for test_taxon
                    results_test_known_multilabel = test(model, criterion, dataloaders['test_known_all'],
                                                         outloader=None,
                                                         epoch=None, return_outputs=True, log_wandb=False,
                                                         hook_names=hook_names,
                                                         mask_taxon=test_taxon, **options)
                    # convert logits to predicted class id
                    pred_ids = np.argmax(results_test_known_multilabel['output_dict']['preds_k'], axis=1)
                    # one-hot labels are already argmaxed in test()
                    target_ids = results_test_known_multilabel['output_dict']['labels']

                    print(len(pred_ids))
                    print(len(target_ids))

                    num_classes = len(set(target_ids))
                    print("num_classes: ", num_classes)

                    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
                    results_test["Multi-Tax-Pred"]['ACC_{}'.format(test_taxon)] = float(accuracy(torch.tensor(pred_ids),
                                                                                       torch.tensor(target_ids))) * 100.

        print("type(results_test_known_multilabel['output_dict']['preds_k_probs'])", type(results_test_known['output_dict']['preds_k_probs']))

        # save the ACC and ACC_taxon on all known test data
        for key in results_test_known:
            if 'ACC' in key:
                results_test[key] = results_test_known[key]

        print("TEST: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t AUPR (%): {:.3f}".format(results_test['ACC'],
                                                                                                       results_test['AUROC'],
                                                                                                       results_test['OSCR'],
                                                                                                       results_test['AUPR']))

        # print ACC pooled to coarser taxons
        print("HARD POOLED TAXON ACC")
        print(*["{:12s}: {:.1f}\n".format(key, results_test[key]) for key in results_test if 'ACC_' in key])

        if "Tax-Pool" in results_test:
            # print ACC pooled to coarser taxons
            print("SOFT POOLED TAXON ACC")
            print(*["{:12s}: {:.1f}\n".format(key, results_test["Tax-Pool"][key]) for key in results_test["Tax-Pool"] if 'ACC_' in key])

        print("POOLED TAXON ACC balanced")
        print(*["{:12s}: {:.1f}\n".format(key, results_test[key]) for key in results_test if 'ACCB_' in key])

        # Save features, logits, probs, mean squared activations of hook layers
        # save test outputs
        output_filepath = '{}/model_outputs.npz'.format(args.save_dir)
        print("results_test['output_dict'].keys():", list(results_test['output_dict'].keys()))
        np.savez(file=output_filepath, **results_test.pop('output_dict'))

        # Save results metrics dict
        output_filepath = '{}/metrics.json'.format(args.save_dir)
        print(results_test)
        with open(output_filepath, 'w') as f:
            json.dump(results_test, f, indent=2)



