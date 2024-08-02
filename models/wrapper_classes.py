import torch
from torch import nn


class TimmResNetWrapper(nn.Module):
    """ Wrapper for ResNet models from timm package to extract embeddings."""

    def __init__(self, resnet, norm_features=False):

        super().__init__()
        self.resnet = resnet
        self.norm_features = norm_features

    def forward(self, x, return_features=True):

        x = self.resnet.forward_features(x)
        embedding = self.resnet.global_pool(x)
        if self.resnet.drop_rate:
            embedding = torch.nn.functional.dropout(embedding, p=float(self.drop_rate), training=self.training)
        if self.norm_features:
            embedding = nn.functional.normalize(embedding, p=2.0, dim=1)  # L2 normalized features
        preds = self.resnet.fc(embedding)

        if return_features:
            return embedding, preds
        else:
            return preds
        

class EfficientNetWrapper(nn.Module):
    """ 
    Wrapper for EfficientNet models from timm package to extract embeddings.
    Last layer is called classifier instead of fc (resnet).
    """

    def __init__(self, net, norm_features=False):

        super().__init__()
        self.net = net
        self.norm_features = norm_features

    def forward(self, x, return_features=True):

        x = self.net.forward_features(x)
        embedding = self.net.global_pool(x)
        if self.net.drop_rate:
            embedding = torch.nn.functional.dropout(embedding, p=float(self.drop_rate), training=self.training)
        if self.norm_features:
            embedding = nn.functional.normalize(embedding, p=2.0, dim=1)  # L2 normalized features
        preds = self.net.classifier(embedding)

        if return_features:
            return embedding, preds
        else:
            return preds

