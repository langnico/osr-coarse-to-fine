#!/bin/bash
#SBATCH --job-name=inat21_test
#SBATCH --gres=gpu:1  
#SBATCH --tasks-per-node=1
#SBATCH --array=0   
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --mem=128G  # total memory per job

# NOTE: --array=0-2 For every taxon run 3 jobs (3 losses)

# set the split index using the job array index (#SBATCH --array)
index=${SLURM_ARRAY_TASK_ID}
echo "index: ${index}"

# set num_models
num_models=5

# set python, save_dir
source config.sh 
source ~/.config_wandb
echo $PYTHON

# to load the data from work without copying to the node
TMPDIR="/cluster/work/igp_psr/nlang"

LOSS_ARRAY=("Softmax" "SoftmaxMultilabel" "SoftmaxMultilabelGRL")  # "SoftmaxMultilabelGRL_a0.35")
#LOSS_ARRAY=("SoftmaxMultilabelGRL_a0.30" "SoftmaxMultilabelGRL_a0.35" "SoftmaxMultilabelGRL_a0.40" "SoftmaxMultilabelGRL_a0.45" "SoftmaxMultilabelGRL_a0.50")
LOSS=${LOSS_ARRAY[${index}]}
#LOSS="Softmax"

# TODO: set TAXON_TRAIN
SUPERCAT="aves"  # aves insecta 
#TAXON_TRAIN_ARRAY=("id" "genus" "family" "order")
#TAXON_TRAIN=${TAXON_TRAIN_ARRAY[${index}]}
TAXON_TRAIN="id"

DATASET_TRAIN="inat21-osr-${SUPERCAT}-${TAXON_TRAIN}-1hop"
echo "DATASET_TRAIN: ${DATASET_TRAIN}"

MODEL="timm_resnet50"
EPOCH=99
IMAGE_SIZE=224
LR=0.1
SPLIT_IDX=0
MODEL_PATH="${SAVE_DIR}/runs/${MODEL}/scratch/${DATASET_TRAIN}/${LOSS}/split_${SPLIT_IDX}/lr_${LR}"

LOG_TEST_DIR=${SAVE_DIR}/test
mkdir -p ${LOG_TEST_DIR}

EXP_NUM=$(ls ${LOG_TEST_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -um test_ensemble \
    --dataset_train ${DATASET_TRAIN} \
    --model_path ${MODEL_PATH} --num_models ${num_models} \
    --test_hops 1


