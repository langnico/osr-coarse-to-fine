import os
import sys
import json

from .inat2021osr import create_inat_dataset_funcs
from .augmentations import get_transform
from config import inat_2021_root, inat21_osr_splits


# construct dict with inat datasets creation functions
"""
For each dataset, define function which returns a dictionary with keys:
    dict_keys(['train', 'val', 'test_known', 'test_unknown', 'test_known_all'])
"""
get_dataset_funcs = create_inat_dataset_funcs()


def parse_batch_dict_or_tuple(batch_data):
    if isinstance(batch_data, tuple):
        data, labels, idx = batch_data
    else:
        data, labels, idx = batch_data['data'], batch_data['labels'], batch_data['idx']
    return data, labels, idx


def get_datasets(name, transform='default', image_size=224, train_classes=(0, 1, 8, 9),
                 open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):

    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return: datasets, a dict of pytorch datasets with keys: train, val, test_known, test_unknown, test_known_all
    """

    print('Loading datasets...')

    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(transform_type=transform, image_size=image_size, args=args)

    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](train_transform, test_transform,
                                  train_classes=train_classes,
                                  open_set_classes=open_set_classes,
                                  balance_open_set_eval=balance_open_set_eval,
                                  split_train_val=split_train_val,
                                  seed=seed)
    else:
        raise NotImplementedError

    return datasets

def get_class_splits(dataset):

    # iNat21
    if "inat21" in dataset:
        # Note: always load the x-hop open-set split with species ids. Pooling to super-class is done after splitting.
        # to load the dataset split, we replace the tax_level with "id"
        tax_level = dataset.split("-")[3]  
        osr_path = os.path.join(inat21_osr_splits, dataset.replace(tax_level, 'id')+".json") 
        with open(osr_path, "r") as f:
            split_dict = json.load(f)
        train_classes = split_dict["train_categories"]
        open_set_classes = split_dict["test_categories"]

    else:
        raise NotImplementedError

    return train_classes, open_set_classes

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
