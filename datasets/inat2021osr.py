import os
import copy
import numpy as np
from config import inat_2021_root
import torchvision
from torch.utils.data import Subset
import sklearn.model_selection

from .inat2021 import INAT, InatFromTar, id_to_tax_rank


inat21_supercat_dict = {
    "aves": {"tax_level": "class", "key": "Aves"},
    "insecta": {"tax_level": "class", "key": "Insecta"}
}


def create_inat_dataset_funcs(dataset_funcs_dict=None, return_multilabel=False):
    if dataset_funcs_dict is None:
        dataset_funcs_dict = {}
    # append dict with inat21 datasets: "inat21-id-1hop", ..., "inat21-order-1hop", "inat21-id-1hop", "inat21-id-4hop",
    for supercat in inat21_supercat_dict.keys():
        for taxon in ['id', 'genus', 'family', 'order']:
            for hop in [1, 2, 3, 4, 5, 6, 7]:
                name = f"inat21-osr-{supercat}-{taxon}-{hop}hop"
                dataset_funcs_dict[name] = InatOSRDatasetGetter(train_filename="train.json", test_filename="val.json",
                                                                target_rank=taxon,
                                                                return_multilabel=return_multilabel).get_datasets
    return dataset_funcs_dict


def get_equal_len_datasets(dataset1, dataset2):

    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2,)), replace=False)
        if isinstance(dataset1, Subset):
            dataset1.indices = [dataset1.indices[index] for index in rand_idxs]
        else:
            Subset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1,)), replace=False)
        if isinstance(dataset2, Subset):
            dataset2.indices = [dataset2.indices[index] for index in rand_idxs]
        else:
            Subset(dataset2, rand_idxs)

    return dataset1, dataset2


def stratified_random_split(dataset, val_frac=0.1, seed=0, stratify_array=None):
    # hyper_indices index the sample indices
    train_hyper_indices, val_hyper_indices = sklearn.model_selection.train_test_split(np.arange(len(dataset)),
                                                                                      test_size=val_frac,
                                                                                      shuffle=True,
                                                                                      stratify=stratify_array,
                                                                                      random_state=seed)
    ds_train = copy.deepcopy(dataset)
    ds_val = copy.deepcopy(dataset)
    # select the train and val sample indices
    ds_train.indices = [dataset.indices[i] for i in train_hyper_indices]
    ds_val.indices = [dataset.indices[i] for i in val_hyper_indices]
    assert set(ds_train.indices).isdisjoint(set(ds_val.indices))
    return ds_train, ds_val


def subsample_classes(dataset, include_classes=(0, 1, 8, 9), deepcopy=True, enumerate_labels=True):
    """Returns a Subset of the dataset keeping only specific class ids.
    This function sets a target transform that maps the class ids (i.e. target taxon ids) to enumerated indices [0,C].
    Per default, it creates a deepcopy of the dataset to not modify the original dataset."""

    # get the sample indices for which the target is within the included classes.
    subset_sample_indices = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    if deepcopy:
        ds_sub = Subset(copy.deepcopy(dataset), subset_sample_indices)
    else:
        ds_sub = Subset(dataset, subset_sample_indices)

    # transform species id to target taxon ids
    tax_ids = sorted(list(set([ds_sub.dataset.target_transform_id_to_tax_rank(species_id) for species_id in include_classes])))
    # set number of tax ids (pooled to target rank)
    ds_sub.dataset.num_classes = len(tax_ids)

    # set the dictionary to transform the target labels to the new class ids of the subset of classes
    ds_sub.dataset.target_dict_tax_id_to_enumerated = {}
    ds_sub.dataset.target_dict_enumerated_to_tax_id = {}
    for i, tax_id in enumerate(tax_ids):
        # get target_rank tax id (super-class)
        ds_sub.dataset.target_dict_tax_id_to_enumerated[tax_id] = i
        ds_sub.dataset.target_dict_enumerated_to_tax_id[i] = tax_id  # can be used to revert model outputs to tax ids

    if enumerate_labels:
        # append the target transform to enumerate target labels
        ds_sub.dataset.target_transforms.transforms.append(lambda x: ds_sub.dataset.target_dict_tax_id_to_enumerated[x])

    return ds_sub


def get_inat_datasets(root_dir, train_file, test_file, patch_size, target_rank,
                      train_transform, test_transform, train_classes=range(10),
                      open_set_classes=range(10, 20), balance_open_set_eval=False, split_train_val=True, seed=0,
                      return_multilabel=False):
    """
    Note: split_train_val has no effect, always split train into train and val
    """

    np.random.seed(seed)

    # Load datasets train, test
    ds_dict = {}
    ds_dict['train'] = InatOSRWrapper(train_transform, root_dir, train_file, is_train=True, return_dict=True,
                                      patch_size=patch_size, target_rank=target_rank, return_multilabel=return_multilabel)

    ds_dict['test'] = InatOSRWrapper(test_transform, root_dir, test_file, is_train=False, return_dict=True,
                                     patch_size=patch_size, target_rank=target_rank, return_multilabel=return_multilabel)

    # Subsample train with known classes
    ds_dict['train'] = subsample_classes(ds_dict['train'], include_classes=train_classes, deepcopy=True)

    # Split train into train and val set
    if split_train_val:
        # first get the actual targets of the train Subset
        stratify_array = [ds_dict['train'].dataset.targets[i] for i in ds_dict['train'].indices]
        ds_dict['train'], ds_dict['val'] = stratified_random_split(ds_dict['train'], val_frac=0.1,
                                                                   stratify_array=stratify_array)
    else:
        ds_dict['val'] = InatOSRWrapper(test_transform, root_dir, test_file, is_train=False, return_dict=True,
                                        patch_size=patch_size, target_rank=target_rank)
        ds_dict['val'] = subsample_classes(ds_dict['val'], include_classes=train_classes)

    # Get test set for known and unknown classes (Note: important to deepcopy the datasets test for each Subset)
    ds_dict['test_known'] = subsample_classes(ds_dict['test'], include_classes=train_classes, deepcopy=True)
    ds_dict['test_unknown'] = subsample_classes(ds_dict['test'], include_classes=open_set_classes, deepcopy=True,
                                                enumerate_labels=False)   # we keep original label tax_ids
    del ds_dict['test']

    print('Before balancing test datasets')
    for k in ['train', 'val', 'test_known', 'test_unknown']:
        print('{}:\t{}'.format(k, len(ds_dict[k])))

    if balance_open_set_eval:
        ds_dict['test_known_all'] = copy.deepcopy(ds_dict['test_known'])
        print('balancing test known and unknown...')
        ds_dict['test_known'], ds_dict['test_unknown'] = get_equal_len_datasets(ds_dict['test_known'],
                                                                                ds_dict['test_unknown'])
    print('After balancing test_known and test_unknown datasets')
    for k in ['train', 'val', 'test_known', 'test_unknown']:
        print('{}:\t{}'.format(k, len(ds_dict[k])))

    return ds_dict


# Note this can be used to define e.g. inat21_mini_genus into data.open_set_datasets.get_dataset_funcs as:
class InatOSRDatasetGetter:
    def __init__(self, root_dir=inat_2021_root,
                 train_filename="train_mini.json",
                 test_filename="val.json",
                 patch_size=224, target_rank='id',
                 return_multilabel=False):

        self.root_dir = root_dir
        self.train_file = os.path.join(self.root_dir, train_filename)
        self.test_file = os.path.join(self.root_dir, test_filename)
        self.patch_size = patch_size
        self.target_rank = target_rank
        self.return_multilabel = return_multilabel

    def get_datasets(self, train_transform, test_transform, train_classes=range(10), open_set_classes=range(10, 20),
                     balance_open_set_eval=True, split_train_val=False, seed=0):

        return get_inat_datasets(self.root_dir, self.train_file, self.test_file, self.patch_size, self.target_rank,
                                 train_transform, test_transform, train_classes, open_set_classes,
                                 balance_open_set_eval, split_train_val, seed, self.return_multilabel)


# OBSOLETE --> use InatOSRDatasetGetter().get_datasets
def get_inat_2021_mini_datasets(train_transform, test_transform, train_classes=range(10),
    open_set_classes=range(10, 20), balance_open_set_eval=False, split_train_val=True, seed=0):

    root_dir = inat_2021_root
    train_file = os.path.join(inat_2021_root, "train_mini.json")
    test_file = os.path.join(inat_2021_root, "val.json")
    patch_size = 224
    target_rank = 'id'

    return get_inat_datasets(root_dir, train_file, test_file, patch_size, target_rank,
                             train_transform, test_transform, train_classes, open_set_classes,
                             balance_open_set_eval, split_train_val, seed)


# OBSOLETE --> use InatOSRDatasetGetter().get_datasets
def get_inat_2021_datasets(train_transform, test_transform, train_classes=range(10),
                           open_set_classes=range(10, 20), balance_open_set_eval=False, split_train_val=True, seed=0):
    root_dir = inat_2021_root
    train_file = os.path.join(inat_2021_root, "train.json")
    test_file = os.path.join(inat_2021_root, "val.json")
    patch_size = 224
    target_rank = 'id'

    return get_inat_datasets(root_dir, train_file, test_file, patch_size, target_rank,
                             train_transform, test_transform, train_classes, open_set_classes,
                             balance_open_set_eval, split_train_val, seed)


# make a wrapper class to reformat the output of get_item as
# Use either InatOSRWrapper(InatFromTar) or InatOSRWrapper(INAT)
class InatOSRWrapper(INAT):
    """
    Note: that the values in category{} are names (strings)

    category{
      "id" : int,
      "name" : str,
      "common_name" : str,
      "supercategory" : str,
      "kingdom" : str,
      "phylum" : str,
      "class" : str,
      "order" : str,
      "family" : str,
      "genus" : str,
      "specific_epithet" : str,
      "image_dir_name" : str,
    }
    """
    def __init__(self, transform, root, ann_file, target_rank='id', return_multilabel=False, **kwargs):

        super(InatOSRWrapper, self).__init__(root, ann_file, transform_input=transform,
                                             return_multilabel=return_multilabel, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

        # set targets using classes from parent class
        self.targets = self.classes
        self.num_classes = None

        # set transforms, Note: target_transforms might be set later when creating subsets holding out complete classes
        self.transform = transform
        self.target_rank = target_rank
        # get target ids of taxon target_rank (i.e. pool species id to a coarser super-class in the taxonomy)
        self.target_transform_id_to_tax_rank = lambda x: id_to_tax_rank(id=x, rank=self.target_rank, taxonomy=self.taxonomy)
        self.target_transforms = torchvision.transforms.Compose([self.target_transform_id_to_tax_rank])

    def __getitem__(self, item):

        out_dict = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        target = self.targets[item]
        # check that the target id equals the label returned by the parent class getitem()
        assert out_dict['labels'] == target

        # Note: input transforms are already applied in super().__getitem__
        target = self.target_transforms(target)

        # rename the keys, without preserving the order
        out_dict['data'] = out_dict.pop('images')
        out_dict['labels'] = target
        out_dict['idx'] = uq_idx
        return out_dict

