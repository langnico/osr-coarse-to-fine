# Gradient reversal layer code from: https://github.com/tadeephuy/GradientReversal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .masked_functionals import masked_log_softmax
from .LabelSmoothing import smooth_cross_entropy_loss
from .gradient_reversal import revgrad


class ClassifierTwoLayer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128):
        super(ClassifierTwoLayer, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module('fc1', nn.Linear(in_features, hidden_features))
        self.block.add_module('bn1', nn.BatchNorm1d(hidden_features))
        self.block.add_module('relu1', nn.ReLU(True))
        self.block.add_module('fc2', nn.Linear(hidden_features, out_features))

    def forward(self, x):
        x = self.block(x)
        return x


class ClassifierLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ClassifierLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


class SoftmaxMultilabelGRL(nn.Module):
    """
    Hierarchy-adversarial loss that discourages hierarchical structure. I.e., a features space in which coarser granularities are not linearly separable,
    but the finest granularity is.
    """
    def __init__(self, finest_tax_level="id", eps=1e-7, **options):
        super(SoftmaxMultilabelGRL, self).__init__()
        self.temp = options['temp']
        self.finest_tax_level = finest_tax_level
        self.eps = eps  # eps could be 1/C (i.e. informed by the number of categories)

        # set maximum weight of the GRL
        if 'alpha_max' in options:
            self.alpha_max = options['alpha_max']
        else:
            self.alpha_max = 0.25

        # init coarse classifier at first forward pass
        # Note: labels from all tax levels are concatenated, but only the coarse part of this classifier used and trained.
        self.classifier_coarse = ClassifierLinear(in_features=options["feat_dim"], out_features=options["num_classes"])
        self.classifier_coarse = self.classifier_coarse.cuda()
    
    def forward(self, x, y, labels=None, data_dict=None, return_taxlevel_losses=False, progress=1.):
        """
        :param x: embeddings
        :param y: logits
        :param labels: concatenated one-hot encoded labels for all tax levels
        :param data_dict: dictionary containing masks for each tax level to mask out the labels from other tax levels
        :param return_taxlevel_losses: return tax level losses
        :param progress: training progress ranging from 0 to 1
        """

        embeddings = x
        logits = y

        # increase the weight of alpha linearly with training progress
        alpha = torch.tensor(progress * self.alpha_max)

        loss = 0
        taxlevel_losses = {}

        # loop through tax_levels
        for tax_level, mask in data_dict['multilabel_masks'].items():
            mask = mask.cuda()

            if tax_level != self.finest_tax_level:
                # coarser levels
                # gradient reversal layer
                embeddings = revgrad(embeddings, alpha)
                logits_tax_level = self.classifier_coarse(embeddings)               
            else:
                logits_tax_level = logits

            # mask the labels
            labels_masked = mask * data_dict['multilabel'].cuda()

            loss_tax_level = torch.mean(torch.sum(-labels_masked * masked_log_softmax(logits_tax_level/self.temp, mask=mask, dim=-1), dim=-1))

            taxlevel_losses[tax_level] = loss_tax_level
            loss += loss_tax_level

        # return the average of tax level losses to preserve the order of magnitude compared to a single tax level loss
        loss /= len(data_dict['multilabel_masks'])

        if return_taxlevel_losses:
            return logits, loss, taxlevel_losses
        else:
            return logits, loss, alpha

