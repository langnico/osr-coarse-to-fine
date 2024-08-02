import torch
import torch.nn as nn
import torch.nn.functional as F

from .LabelSmoothing import smooth_cross_entropy_loss

class Softmax(nn.Module):
    def __init__(self, **options):
        super(Softmax, self).__init__()
        self.temp = options['temp']
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        """
        :param x: embeddings (not used for this loss, but for others in this framework)
        :param y: logits
        :param labels: labels
        """

        # 'y' is logits of the classifier
        logits = y

        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss = F.cross_entropy(logits / self.temp, labels)
        else:
            loss = smooth_cross_entropy_loss(logits / self.temp, labels=labels, smoothing=self.label_smoothing, dim=-1)

        return logits, loss
