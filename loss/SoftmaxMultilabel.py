import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_functionals import masked_log_softmax
from .LabelSmoothing import smooth_cross_entropy_loss


class SoftmaxMultilabel(nn.Module):
    """
    Hierarchy-supportive loss that encourages hierarchical structure. Here the cross-entropy loss is optimized for each tax level.
    """
    def __init__(self, **options):
        super(SoftmaxMultilabel, self).__init__()
        self.temp = options['temp']

    def forward(self, x, y, labels=None, data_dict=None, return_taxlevel_losses=False):
        """
        :param x: embeddings (not used for this loss, but for others in this framework)
        :param y: logits
        :param labels: concatenated one-hot encoded labels for all tax levels
        :param data_dict: dictionary containing masks for each tax level to mask out the labels from other tax levels
        :param return_taxlevel_losses: return losses for each tax level
        """

        # 'y' is logits of the classifier
        logits = y
        loss = 0
        taxlevel_losses = {}

        # loop through tax_levels
        for tax_level, mask in data_dict['multilabel_masks'].items():
            mask = mask.cuda()
            # mask the labels and logits
            labels_masked = mask * data_dict['multilabel'].cuda()  # one-hot encoded labels
            loss_tax_level = torch.mean(torch.sum(-labels_masked * masked_log_softmax(logits/self.temp, mask=mask, dim=-1), dim=-1))

            taxlevel_losses[tax_level] = loss_tax_level
            loss += loss_tax_level

        # return the average of tax level losses to preserve the order of magnitude compared to a single tax level loss
        loss /= len(data_dict['multilabel_masks'])

        if return_taxlevel_losses:
            return logits, loss, taxlevel_losses
        else:
            return logits, loss

