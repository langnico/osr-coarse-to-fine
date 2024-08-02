import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from core import evaluation

from sklearn.metrics import average_precision_score

import wandb
import core.calibration_utils as cal_utils
from datasets.open_set_datasets import parse_batch_dict_or_tuple
from loss.SoftmaxMultilabel import SoftmaxMultilabel
from loss.SoftmaxMultilabelGRL import SoftmaxMultilabelGRL


def get_activation(name, activation_dict):
    def hook(model, input, output):
        # return the average squared activation map
        activation_dict[name].append(torch.square(output.detach()).mean([2, 3]).data.cpu().numpy())

    return hook


def register_hooks(model, hook_names):
    # init activations with empty lists to collect forward hook data
    activation_dict = {k: [] for k in hook_names}
    hook_handles = []
    # register hooks
    for name in hook_names:
        h = model.get_submodule(name).register_forward_hook(get_activation(name, activation_dict))
        hook_handles.append(h)
    return activation_dict, hook_handles


def unregister_hooks(hook_handles):
    [h.remove() for h in hook_handles]


def embed_features(net, criterion, dataloader, mask_taxon="id", **options):
    net.eval()
    results = {}
    torch.cuda.empty_cache()

    _feat_k = []
    _labels = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            data, labels, idx = parse_batch_dict_or_tuple(batch_data)

            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                z, logits = net(data, True)

                if isinstance(criterion, (SoftmaxMultilabel, SoftmaxMultilabelGRL)):
                    logits, loss, taxlevel_losses = criterion(z, logits, labels, batch_data, return_taxlevel_losses=True)
                    # mask logits and one-hot labels to use only species ids output for the evaluation
                    mask_id = batch_data['multilabel_masks'][mask_taxon].cuda()
                    logits = mask_id * logits
                    labels = mask_id * batch_data["multilabel"].cuda()
                    # get argmax from one-hot encoded labels
                    labels = labels.data.max(1)[1]
                else:
                    logits, loss = criterion(z, logits, labels)

                _labels.append(labels.data.cpu().numpy())
                _feat_k.append(z.data.cpu().numpy())
   
    _labels = np.concatenate(_labels, 0)  # shape (n,)
    print("_labels.shape:", _labels.shape)

    # collect features (i.e. activation maps before the last linear layer yields the logits)
    _feat_k = np.concatenate(_feat_k, 0)  # features z with shape (n, d)
    print("_feat_k.shape:", _feat_k.shape)

    output_dict = {'labels': _labels, 'feat_k': _feat_k}
    results['output_dict'] = output_dict
    return results


def test(net, criterion, testloader, outloader=None, epoch=None, return_outputs=False, log_wandb=True,
         hook_names=None, mask_taxon="id", return_probs=True, **options):

    net.eval()
    correct, total = 0, 0
    results = {}
    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels, _labels_u = [], [], [], []
    _feat_k, _feat_u = [], []
    if hook_names is not None:
        print('registering hooks...')
        activation_known, hook_handles_k = register_hooks(model=net.resnet, hook_names=hook_names)

    with torch.no_grad():
        for batch_data in tqdm(testloader):
            data, labels, idx = parse_batch_dict_or_tuple(batch_data)

            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                z, logits = net(data, True)

                if isinstance(criterion, (SoftmaxMultilabel, SoftmaxMultilabelGRL)):
                    logits, loss, taxlevel_losses = criterion(z, logits, labels, batch_data, return_taxlevel_losses=True)
                    # mask logits and one-hot labels to use only one granularity e.g. only species ids (default) output for the evaluation
                    mask_id = batch_data['multilabel_masks'][mask_taxon].cuda()
                    logits = mask_id * logits
                    labels = mask_id * batch_data["multilabel"].cuda()
                    # get argmax from one-hot encoded labels
                    labels = labels.data.max(1)[1]

                    for k, v in taxlevel_losses.items():
                        results["CE_{}".format(k)] = float(v)
                        results["CE_AVG"] = float(loss)
                else:
                    logits, loss = criterion(z, logits, labels)

                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                if options['use_softmax_in_eval']:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())
                _feat_k.append(z.data.cpu().numpy())

        if hook_names is not None:
            # unregister hooks
            unregister_hooks(hook_handles_k)

        # if open-set dataloader is given run prediction
        if outloader is not None:
            if hook_names is not None:
                print('registering hooks...')
                activation_unknown, hook_handles_u = register_hooks(model=net.resnet, hook_names=hook_names)

            for batch_idx, batch_data in enumerate(tqdm(outloader)):
                data, labels, idx = parse_batch_dict_or_tuple(batch_data)

                if options['use_gpu']:
                    data, labels = data.cuda(), labels.cuda()

                with torch.set_grad_enabled(False):
                    z, logits = net(data, True)
                    if isinstance(criterion, (SoftmaxMultilabel, SoftmaxMultilabelGRL)):
                        logits, _, _ = criterion(z, logits, labels, batch_data, return_taxlevel_losses=True)
                        # Use only species ids output for evaluation
                        mask_id = batch_data['multilabel_masks']["id"].cuda()
                        logits = mask_id * logits
                        labels = mask_id * batch_data["multilabel"].cuda()
                        # get argmax from one-hot encoded labels
                        labels = labels.data.max(1)[1]
                    else:
                        logits, _ = criterion(z, logits)

                    if options['use_softmax_in_eval']:
                        logits = torch.nn.Softmax(dim=-1)(logits)

                    _pred_u.append(logits.data.cpu().numpy())
                    _feat_u.append(z.data.cpu().numpy())
                    _labels_u.append(labels.data.cpu().numpy())
            if hook_names is not None:
                # unregister hooks
                unregister_hooks(hook_handles_u)

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))
    results['ACC'] = acc

    _pred_k = np.concatenate(_pred_k, 0)  # logits of shape (n, c)
    print("_pred_k.shape:", _pred_k.shape)
    if return_probs:
        _pred_k_probs = torch.nn.Softmax(dim=1)(torch.Tensor(_pred_k)).data.cpu().numpy()  # softmax probs of shape (n, c)
    else:
        _pred_k_probs = None
    _labels = np.concatenate(_labels, 0)  # shape (n,)
    print("_labels.shape:", _labels.shape)

    # collect features (i.e. activation maps before the last linear layer yields the logits)
    _feat_k = np.concatenate(_feat_k, 0)  # features z with shape (n, d)

    # collect activations
    if hook_names is not None:
        for name in hook_names:
            activation_known[name] = np.concatenate(activation_known[name], 0)  # activation with shape (n, d)
            if outloader is not None:
                activation_unknown[name] = np.concatenate(activation_unknown[name], 0)  # activation with shape (n, d)

    # compute cross-entropy
    results['CE'] = float(torch.nn.functional.cross_entropy(input=torch.Tensor(_pred_k),
                                                      target=torch.Tensor(_labels).type(torch.LongTensor)).data.cpu().numpy())

    # Calibration error on closed-set
    if _pred_k_probs is not None:
        num_bins = 10
        results.update(cal_utils.get_calibration_metrics(probs=_pred_k_probs, labels=_labels, num_bins=num_bins))

    if outloader is not None:
        _pred_u = np.concatenate(_pred_u, 0)  # logits of shape (n, c)
        _pred_u_probs = torch.nn.Softmax(dim=1)(torch.Tensor(_pred_u)).data.cpu().numpy()  # softmax probs of shape (n, c)
        _feat_u = np.concatenate(_feat_u, 0)  # features z with shape (n, d)
        _labels_u = np.concatenate(_labels_u, 0)  # shape (n,)

        # Out-of-Distribution detection evaluation
        score_k, score_u = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
        print(score_k.shape, score_u.shape)
        results.update(evaluation.metric_ood(score_k, score_u, _labels, _pred_k_probs)['Bas'])

        if log_wandb:
            # Log calibration plot to wandb
            confidence_bins, accuracy_bins, count_bins = cal_utils.get_calibration_plot_arrays(labels=_labels, probs=_pred_k_probs, num_bins=num_bins)
            fig_cal = cal_utils.plot_calibration(confidence_bins, accuracy_bins)
            wandb.log({"cal_plot": fig_cal}, step=epoch)

            # Log distribution of max_probs to wandb
            data = [[s] for s in np.max(_pred_k_probs, axis=1)]
            table = wandb.Table(data=data, columns=["max_probs"])
            wandb.log({'confidence_histogram': wandb.plot.histogram(table, "max_probs",
                                                            title="Confidence (max prob) histogram")}, step=epoch)

    if return_outputs:
        if outloader is not None:
            output_dict = {'labels': _labels,
                           'labels_u': _labels_u,
                           'preds_k': _pred_k, 'preds_u': _pred_u,
                           'preds_k_probs': _pred_k_probs, 'preds_u_probs': _pred_u_probs,
                           'feat_k': _feat_k, 'feat_u': _feat_u}
            if hook_names is not None:
                output_dict['activations_k'] = activation_known
                output_dict['activations_u'] = activation_unknown
        else:
            output_dict = {'labels': _labels, 'preds_k': _pred_k, 'preds_k_probs': _pred_k_probs, 'feat_k': _feat_k}

        results['output_dict'] = output_dict
    return results

