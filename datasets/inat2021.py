import os
import torch
import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import numpy as np
import tarfile
import io

from torchvision import transforms


class PadIfNeeded(object):
    """ Pad tensor image if needed to match the target size. Expects tensor image (C x H x W) """

    def __init__(self, target_size):
        """
        Args:
            target_size: int, assumes square target size (width==height)
        """
        self.target_size = torch.Size([target_size, target_size])

    def __call__(self, x):
        if x.shape[1:3] == self.target_size:
            return x
        else:
            height, width = x.shape[1:3]
            # compute padding
            left = (self.target_size[1] - width) // 2
            right = self.target_size[1] - left - width
            top = (self.target_size[0] - height) // 2
            bottom = self.target_size[0] - top - height
            # pad to target size
            x = transforms.functional.pad(x, padding=[left, top, right, bottom], fill=0, padding_mode='constant')
            return x


def id_to_tax_rank(id, rank, taxonomy):
    return taxonomy[rank][id]


def ids_to_tax_rank(ids, rank, taxonomy):
    return [taxonomy[rank][id] for id in ids]


def tax_ids_to_species_ids(tax_ids, tax_level, taxonomy):
    all_species_ids = np.array(list(taxonomy[tax_level].keys()))
    all_tax_ids = np.array(list(taxonomy[tax_level].values()))
    out_species_ids = []
    for tax_id in tax_ids:
        out_species_ids.extend(list(all_species_ids[all_tax_ids == tax_id]))
    return out_species_ids


def tax_id_to_supertax_id(tax_ids, tax_level_source, tax_level_target, taxonomy):
    # first, get the first species id for every tax id
    first_species_ids = []
    for tax_id in tax_ids:
        for k, v in taxonomy[tax_level_source].items():
            if v == tax_id:
                first_species_ids.append(k)
                break
    assert len(first_species_ids) == len(tax_ids)
    # second, convert species ids to the target taxon id
    supertax_ids = ids_to_tax_rank(first_species_ids, tax_level_target, taxonomy)
    return supertax_ids


def hop_distance(id_a, id_b, taxonomy):
    for hop, taxon in enumerate(taxonomy.keys()):
        tax_id_a = id_to_tax_rank(id_a, taxon, taxonomy)
        tax_id_b = id_to_tax_rank(id_b, taxon, taxonomy)
        if tax_id_a == tax_id_b:
            return hop
    return len(taxonomy.keys())


def compute_hop_distances(labels_a, labels_b, taxonomy):
    dists = np.zeros((len(labels_a), len(labels_b)))
    for i, l_i in enumerate(labels_a):
        for j, l_j in enumerate(labels_b):
            dists[i, j] = hop_distance(l_i, l_j, taxonomy=taxonomy)
    return dists


def compute_min_hop_distances(labels_a, labels_b, taxonomy):
    dists = compute_hop_distances(labels_a, labels_b, taxonomy)
    return np.min(dists, axis=1)


def make_sub_taxonomy_from_species_ids(taxonomy, species_ids_sub):
    taxonomy_sub = {}
    for tax_level in taxonomy:
        taxonomy_sub[tax_level] = {}
        for k, v in taxonomy[tax_level].items():
            if k in species_ids_sub:
                taxonomy_sub[tax_level][k] = v

    return taxonomy_sub

def get_num_samples_per_tax_id(ann_data, tax_level, taxonomy):
    tax_ids = np.unique(list(taxonomy[tax_level].values()))
    num_samples_per_tax_id = dict.fromkeys(tax_ids, 0)

    for annotation in ann_data['annotations']:
        species_id = annotation['category_id']
        if species_id in taxonomy[tax_level].keys():
            tax_id = taxonomy[tax_level][species_id]
            num_samples_per_tax_id[tax_id] += 1

    return num_samples_per_tax_id


def load_tax_id_name_dicts(ann_data, tax_levels):
    rank_categories_to_names = {}
    rank_names_to_categories = {}
    for tt in tax_levels:
        tax_data = [aa[tt] for aa in ann_data['categories']]
        tax_names = np.unique(tax_data, return_inverse=False)
        rank_categories_to_names[tt] = dict(zip(range(len(tax_names)), list(tax_names)))
        rank_names_to_categories[tt] = dict(zip(list(tax_names), range(len(tax_names))))
    return rank_categories_to_names, rank_names_to_categories


def soft_tax_pool(probs, tax_level, taxonomy, return_lookup_enumerated_to_tax_id=False):
    """
    Pool (i.e. aggregate) the species probabilities to coarser tax probabilities.
    :param probs: batch of probabilities (num_samples, num_species)
    :param tax_level: string key in taxonomy dictionary
    :param taxonomy: dict as created by load_taxonomy()
    :return:
    """
    tax_ids_unique = torch.Tensor(np.unique(list(taxonomy[tax_level].values())))[..., None]  # (num_pooled_ids, 1)
    # create masks M.shape = (num_tax_ids, num_probs)
    tax_ids = torch.Tensor(list(taxonomy[tax_level].values())).repeat(len(tax_ids_unique), 1)  # (num_pooled_ids, num_probs)
    masks = torch.eq(tax_ids, tax_ids_unique).type(torch.float32).cuda()  # (num_pooled_ids, num_probs)
    probs_pooled_T = torch.mm(masks, probs.T)
    probs_pooled = probs_pooled_T.T  # (num_samples, num_pooled_ids)
    if return_lookup_enumerated_to_tax_id:
        dict_enumerated_to_tax_id = {}
        dict_tax_id_to_enumerated = {}
        for i, id in enumerate(tax_ids_unique):
            dict_enumerated_to_tax_id[i] = id
            dict_tax_id_to_enumerated[i] = id
        return probs_pooled, dict_enumerated_to_tax_id, dict_tax_id_to_enumerated
    else:
        return probs_pooled

def load_taxonomy(ann_data, tax_levels, classes):
    """
    First output: taxonomy dictionary
    With number of species C (categories with finest granularity). "id" represents the species level.
    The dictionary taxonomy allows to map the species 'id' to another taxonomic rank index (i.e. integer).
    E.g. the 'genus' index of a species 'id' is obtained with taxonomy['genus']['id']

    Second output: classes_taxonomic dictionary
    classes_taxonomic[species_id] = list of tax_ids


    taxonomy = {
        'id': {
            0: 0,
            1: 1,
            ...,
            C-1: C-1
        },
        'genus': {
            0: int,
            1: int,
            ...,
            C-1: int
        },
        'fammiliy': {},
        'order': {},
        'class': {},
        'phylum': {},
        'kingdom': {}
    }
    """

    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


def get_num_categories_per_taxonomic_rank(taxonomy):
    num_classes_taxonomy = {}
    for key in taxonomy:
        unique_tax_ids = np.unique(list(taxonomy[key].values()))
        num_classes_taxonomy[key] = len(unique_tax_ids)
    return num_classes_taxonomy


class INAT(data.Dataset):
    def __init__(self, root, ann_file, is_train=True, return_dict=False, patch_size=224, crop_padding=32,
                 transform_input=None, return_multilabel=False):

        self.ann_file = ann_file
        self.return_dict = return_dict
        self.return_multilabel = return_multilabel

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        # inat_2021        10000, 4884,    1103,     273,     51,      13,       3
        # inat_2018         8142, 4412,    1120,     273,     57,      25,       6
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']

        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)
        self.num_taxonomic_rank = get_num_categories_per_taxonomic_rank(self.taxonomy)

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        if return_multilabel:
            self.num_multilabel_output = sum(v for v in self.num_taxonomic_rank.values())
            self.masks = self.make_masks()

        self.transform_input = transform_input

        if self.transform_input == "inat2018_transforms" or self.transform_input is None:
            # augmentation params
            self.im_size = [patch_size, patch_size]  # can change this to train on higher res
            self.crop_padding = crop_padding
            self.interpolation = transforms.InterpolationMode.BILINEAR
            self.mu_data = [0.485, 0.456, 0.406]
            self.std_data = [0.229, 0.224, 0.225]
            self.brightness = 0.4
            self.contrast = 0.4
            self.saturation = 0.4
            self.hue = 0.25

            # augmentations
            self.resize_test_first = transforms.Resize(size=int(self.im_size[0] / 0.875), interpolation=self.interpolation)
            self.resize_test = transforms.Resize(size=self.im_size, interpolation=self.interpolation)
            self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
            self.pad_if_needed = PadIfNeeded(target_size=self.im_size[0])
            self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0], interpolation=self.interpolation)
            self.flip_aug = transforms.RandomHorizontalFlip()
            self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
            self.tensor_aug = transforms.ToTensor()
            self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

        if self.transform_input is None:
            self.transform_input = transforms.Compose([
                self.tensor_aug,
                self.resize_test_first,
                self.center_crop
            ])

        if self.transform_input == "inat2018_transforms":
            # set input transforms
            if self.is_train:
                self.transform_input = transforms.Compose([
                    self.tensor_aug,
                    self.scale_aug,
                    self.flip_aug,
                    self.color_aug,
                    self.norm_aug
                    ])
            else:
                self.transform_input = transforms.Compose([
                    self.tensor_aug,
                    self.resize_test_first,
                    self.center_crop,
                    self.norm_aug
                ])


    def __getitem__(self, index):
        path = self.get_path(index)
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        # transform image:
        img = self.transform_input(img)

        if self.return_dict:
            out_dict = {'images': img, 'labels': species_id, 'image_ids': im_id, 'tax_ids': tax_ids}

            if self.return_multilabel:
                out_dict['multilabel'] = self.get_multilabel(tax_ids)
                out_dict['multilabel_masks'] = self.masks
            return out_dict
        else:
            return img, im_id, species_id, tax_ids

    def get_path(self, index):
        return os.path.join(self.root, self.imgs[index])

    def make_masks(self):
        total = 0
        mask_ranges = {}
        for tl in self.tax_levels:
            start = total
            total += self.num_taxonomic_rank[tl]
            mask_ranges[tl] = (start, total)
        masks = {}
        for tl in mask_ranges:
            mask = torch.zeros(total)
            start, end = mask_ranges[tl]
            mask[start: end] = 1
            masks[tl] = mask
        return masks

    def get_multilabel(self, tax_ids):
        labels = torch.zeros(self.num_multilabel_output)
        total = 0
        for idx, tl in enumerate(self.tax_levels):
            # assumes same order as tax_levels
            sub_label = tax_ids[idx]
            labels[total + sub_label] = 1
            total += self.num_taxonomic_rank[tl]
        return labels

    def __len__(self):
        return len(self.imgs)


def tar_loader(tar, path):
    f = tar.extractfile(path)
    img = f.read()
    img = Image.open(io.BytesIO(img))
    return img


class InatFromTar(INAT):
    def __init__(self, *args, **kwargs):
        super(InatFromTar, self).__init__(*args, **kwargs)

        self.tar_filename = self.ann_file.replace('.json', '.tar')
        # overwrite loader
        self.loader = self._tar_loader

    def __getitem__(self, item):
        # open the h5 file in the first iteration --> each worker has its own connection
        if not hasattr(self, 'tar'):
            self._open_tar()

        return super().__getitem__(item)

    def get_path(self, index):
        """ overwrites the get_path() that returns the absolute path to return the cached member.
         Members correspond to the relative paths: E.g. train/class_dir/image_name.jpg """
        name = self.imgs[index]
        return self.tar_members_by_name[name]

    def _tar_loader(self, path):
        """ self.loader only takes path as an argument in the getitem() """
        return tar_loader(tar=self.tar, path=path)

    def _open_tar(self):
        #worker = torch.utils.data.get_worker_info()
        #print("opening tar in worker: ", worker.id)
        self.tar = tarfile.open(os.path.join(self.root, self.tar_filename), 'r')
        # cache all tar members (keep only file members, remove directory members)
        self.tar_members = [m for m in self.tar.getmembers() if m.isfile()]
        self.tar_members = sorted(self.tar_members, key=lambda m: m.name)
        self.tar_members_by_name = {m.name: m for m in self.tar_members}

    def __del__(self):
        if hasattr(self, 'tar'):
            self.tar.close()
