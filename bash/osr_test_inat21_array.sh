#!/bin/bash
#SBATCH --job-name=inat21_test
#SBATCH --gres=gpu:1   
#SBATCH --tasks-per-node=1
#SBATCH --array=0   # 0-34
#SBATCH --cpus-per-task=4
#SBATCH --time=0-4:00:00
#SBATCH --mem=70G   # total memory per job

# NOTE: For every TAXON_TRAIN: submit 5*7=35 jobs in total (5 models, 7 test datasets)

echo "NODELIST: " $SLURM_NODELIST
echo "NODENAME: " $SLURMD_NODENAME
sstat -j $SLURM_JOB_ID -o Nodelist
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
nvidia-smi

# set the split index using the job array index 
index=${SLURM_ARRAY_TASK_ID}
echo "index: ${index}"

num_models=5  #5

MODEL_ID=$((${index}%${num_models}))
dataset_index=$((${index}/${num_models}))
echo "MODEL_ID: ${MODEL_ID}"
echo "dataset_index: ${dataset_index}"

# set python, save_dir
source config.sh
source ~/.config_wandb
echo $PYTHON

LOSS='Softmax'  # Softmax SoftmaxMultilabel SoftmaxMultilabelGRL 

# UNCOMMENT for alpha sweep with LOSS='SoftmaxMultilabelGRL'
# num_sweep=5
# alpha_index=$((${index}%${num_sweep}))
# dataset_index=$((${index}/${num_sweep}))
# echo "alpha_index: ${alpha_index}"
# echo "dataset_index: ${dataset_index}"
# ALPHA_MAX_ARRAY=(0.30 0.35 0.40 0.45 0.50)
# ALPHA_MAX=${ALPHA_MAX_ARRAY[${alpha_index}]}
# echo "ALPHA_MAX: ${ALPHA_MAX}"
# LOSS_SUFFIX="_a${ALPHA_MAX}"

# TODO: set TAXON_TRAIN
SUPERCAT="aves"  # aves animalia
TAXON_TRAIN="id"  # "id" "genus" "family" "order"

HOP=$((${dataset_index}+1))
DATASET_TRAIN="inat21-osr-${SUPERCAT}-${TAXON_TRAIN}-1hop"
DATASET_TEST="inat21-osr-${SUPERCAT}-${TAXON_TRAIN}-${HOP}hop"
echo "DATASET_TRAIN: ${DATASET_TRAIN}"
echo "DATASET_TEST:  ${DATASET_TEST}"

MODEL="timm_resnet50"
EPOCH=99  
IMAGE_SIZE=224
LR=0.1
SPLIT_IDX=0  # has no effect for inat21-osr
MODEL_PATH="${SAVE_DIR}/runs/${MODEL}/scratch/${DATASET_TRAIN}/${LOSS}${LOSS_SUFFIX}/split_${SPLIT_IDX}/lr_${LR}/model_${MODEL_ID}/checkpoints/${DATASET_TRAIN}_${EPOCH}_${LOSS}.pth"

LOG_TEST_DIR=${SAVE_DIR}/test
mkdir -p ${LOG_TEST_DIR}

EXP_NUM=$(ls ${LOG_TEST_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -um testing --dataset_train=${DATASET_TRAIN} --dataset=${DATASET_TEST} --loss=${LOSS} --use_default_parameters='False' \
--num_workers=4 --gpus 0 \
--model=${MODEL} --model_path=${MODEL_PATH} --split_idx=${SPLIT_IDX} \
--image_size=${IMAGE_SIZE} 

