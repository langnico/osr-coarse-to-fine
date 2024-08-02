# From Coarse to Fine-Grained Open-Set Recognition 
This repository contains the code used to create the results presented in the paper: [From Coarse to Fine-Grained Open-Set Recognition](https://openaccess.thecvf.com/content/CVPR2024/papers/Lang_From_Coarse_to_Fine-Grained_Open-Set_Recognition_CVPR_2024_paper.pdf)

TLDR: We investigate the role of label granularity, semantic similarity, and hierarchical representations in open-set recognition (OSR) with an OSR-benchmark based on iNat2021.

More information on the [project page](https://langnico.github.io/fine-grained-osr/). 

## Installation
We use [miniforge3](https://github.com/conda-forge/miniforge) to install a conda environment. 

**For the datasets only**: 

    mamba env create -f env_datasets.yml
    conda activate osr-datasets

**For the full repository**:

    mamba env create -f env_all.yml
    conda activate osr-coarse-to-fine

## Setup 
Update the configuration of code, data, and ouput paths to your system in: 
`config.sh` and `config.py`.

### Weights and Biases
Setup a wandb account to get your wandb api key.
The training bash scripts set the wandb api key from a file in the home directory: `source ~/.config_wandb` which contains `export WANDB_API_KEY=YOUR-API-KEY-HERE`.


## iNat2021-OSR dataset

### Downloading the iNat2021 dataset
We use the iNat2021 dataset ([Van Horn et al., 2021](https://arxiv.org/abs/2103.16483)) that can be downloaded from [here](https://github.com/visipedia/inat_comp/tree/master/2021).
We provide a bash script to download the data and check the md5sum as follows: 

```bash
inat_dir=/path/to/inat21
mkdir $inat_dir
bash bash/data_download/inat21_download.sh $inat_dir
bash bash/data_download/inat21_check_md5sum.sh $inat_dir
```

Note: If your filesystem does not like many files, the train and val folder can be converted into a .tar (without compression) and directly loaded from the tar file.

## Loading iNat2021-OSR pytorch datasets
We introduce iNat2021-OSR, a benchmark with curated open-set splits for the iNat2021 dataset ([Van Horn et al., 2021](https://arxiv.org/abs/2103.16483)) for two taxa: birds (AVES) and insects (INSECTA). This enables the study of OSR along seven discrete “hops” that encode the semantic distance from coarse-grained (7-hop) to fine-grained (1-hop). 

The closed-set (`"train_categories"`) and open-set (`"test_categories"`) species ids are provided as json files by datasetname in the folder `datasets/inat21_osr_splits`.

```
├── datasets
│   ├── inat21_osr_splits
│   │   ├── inat21-osr-aves-id-1hop.json
│   │   ├── inat21-osr-aves-id-2hop.json
│   │   ├── inat21-osr-aves-id-3hop.json
│   │   ├── inat21-osr-aves-id-4hop.json
│   │   ├── inat21-osr-aves-id-5hop.json
│   │   ├── inat21-osr-aves-id-6hop.json
│   │   ├── inat21-osr-aves-id-7hop.json
│   │   ├── inat21-osr-insecta-id-1hop.json
│   │   ├── inat21-osr-insecta-id-2hop.json
│   │   ├── inat21-osr-insecta-id-3hop.json
│   │   ├── inat21-osr-insecta-id-4hop.json
│   │   ├── inat21-osr-insecta-id-5hop.json
│   │   ├── inat21-osr-insecta-id-6hop.json
│   │   └── inat21-osr-insecta-id-7hop.json

```

See example notebook for how to load the pytorch datasets [notebooks/example_load_dataset_inat21osr.ipynb](notebooks/example_load_dataset_inat21osr.ipynb).

```python
from datasets.open_set_datasets import get_class_splits, get_datasets

dataset_name = 'inat21-osr-aves-id-1hop'
# load the data split ids
train_classes, open_set_classes = get_class_splits(dataset_name)
# load pytorch datasets as a dict with dict_keys(['train', 'val', 'test_known', 'test_unknown'])
dataset_dict = get_datasets(dataset_name, transform='visualize', 
                            train_classes=train_classes, open_set_classes=open_set_classes, 
                            balance_open_set_eval=True, split_train_val=True, image_size=224)
```

#### Creating new open-set splits for iNat21
1. In `datasets/inat2021osr.py` add a new supercategory to this dictionary: 
    ```python
    inat21_supercat_dict = {
        "aves": {"tax_level": "class", "key": "Aves"},
        "insecta": {"tax_level": "class", "key": "Insecta"}
    }
    ```
2. Run the script to create a new split: `python datasets/inat2021_create_osr_splits.py`


## Training 
Set parameters in `bash/osr_train_inat21_array.sh` and run the shell script.
Loss functions:
    
- For standard cross-entropy: `LOSS="Softmax"`
- For hierarchy-supporting: `LOSS="SoftmaxMultilabel"`
- For hierarchy-adversarial: `LOSS="SoftmaxMultilabelGRL"` 

Run slurm job: 
    
    sbatch < bash/osr_train_inat21_array.sh


## Testing
To evaluate the trained models on all 7 open-set splits, set the parameters in `bash/osr_test_inat21_array.sh`.

Run slurm job: 

    sbatch < bash/osr_test_inat21_array.sh


## Collecting ensemble results and evaluating OSR scores
To collect ensemble results and evaluate ensemble OSR-scores, set parameters in `bash/osr_test_ensemble_inat21_array.sh`.

Run slurm job: 

    sbatch < bash/osr_test_ensemble_inat21_array.sh`



## Citation
1. Please cite our paper if you use this code or any of the provided data.

Lang, N., Snæbjarnarson, V., Cole, E., Mac Aodha, O., Igel, C., & Belongie, S. (2024). From Coarse to Fine-Grained Open-Set Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 17804-17814).

```
@InProceedings{Lang_2024_CVPR,
    author    = {Lang, Nico and Sn{\ae}bjarnarson, V\'esteinn and Cole, Elijah and Mac Aodha, Oisin and Igel, Christian and Belongie, Serge},
    title     = {From Coarse to Fine-Grained Open-Set Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17804-17814}
}
```

2. Please also cite the original paper introducing the iNat2021 dataset:

Van Horn, G., Cole, E., Beery, S., Wilber, K., Belongie, S., & Mac Aodha, O. (2021). [Benchmarking representation learning for natural world image collections](https://arxiv.org/abs/2103.16483). In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 12884-12893).

```
@InProceedings{Van_Horn_2021_CVPR,
    author    = {Van Horn, Grant and Cole, Elijah and Beery, Sara and Wilber, Kimberly and Belongie, Serge and Mac Aodha, Oisin},
    title     = {Benchmarking Representation Learning for Natural World Image Collections},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12884-12893}
}
```



## Credits
This repository is based on code form:
- https://github.com/sgvaze/osr_closed_set_all_you_need
- https://github.com/iCGY96/ARPL 


