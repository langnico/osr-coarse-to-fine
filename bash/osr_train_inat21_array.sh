#!/bin/bash
#SBATCH --job-name=inat21
#SBATCH --gres=gpu:1 
#SBATCH --tasks-per-node=1 
#SBATCH --array=0 
#SBATCH --cpus-per-task=4
#SBATCH --time=0-60:00:00
#SBATCH --mem=64G  # total memory per job

# NOTE: 
# To study supervision granularity: 5*4=20 jobs in total (5 models, 4 train datasets): --array=0-19%5
# Hierarchical losses: 5 jobs for multilabel losses (only train for id train dataset): --array=0-4%5

# set the split index using the job array index (#SBATCH --array)
index=${SLURM_ARRAY_TASK_ID}
echo "index: ${index}"

num_models=5

MODEL_ID=$((${index}%${num_models}))
dataset_index=$((${index}/${num_models}))
echo "MODEL_ID: ${MODEL_ID}"
echo "dataset_index: ${dataset_index}"

# set python, save_dir
source config.sh
source ~/.config_wandb
echo $PYTHON

echo "NODELIST: " $SLURM_NODELIST
echo "NODENAME: " $SLURMD_NODENAME
sstat -j $SLURM_JOB_ID -o Nodelist
nvidia-smi
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"


SUPERCAT='aves'  # aves insecta 
DATASET_ARRAY=( "inat21-osr-${SUPERCAT}-id-1hop" \
                "inat21-osr-${SUPERCAT}-genus-1hop" \
                "inat21-osr-${SUPERCAT}-family-1hop" \
                "inat21-osr-${SUPERCAT}-order-1hop" )

DATASET=${DATASET_ARRAY[${dataset_index}]}
echo "DATASET: ${DATASET}"

LOSS='Softmax'  # Softmax SoftmaxMultilabel SoftmaxMultilabelGRL 
ALPHA_MAX=0.25
MODEL="timm_resnet50"  
RESNET50_PRETRAIN="scratch"  
FEAT_DIM=2048

IMAGE_SIZE=224

BATCH_SIZE=64 
SPLIT_TRAIN_VAL="True"
LR=0.1

TRANSFORM='rand-augment'
RAND_AUG_N=1
RAND_AUG_M=6
LABEL_SMOOTHING=0

SEED=${MODEL_ID}

OPTIM="sgd"
MAX_EPOCH=100   
SCHEDULER="multi_step"  
STEPS=( 30 60 90 )  # for multi_step

WEIGHT_DECAY=1e-4

NUM_WORKERS=4
PERSISTENT_WORKERS="True"

######################################

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -um training --dataset=$DATASET --loss=${LOSS} \
--num_workers=${NUM_WORKERS} --persistent_workers=${PERSISTENT_WORKERS} --gpus 0 \
--model=${MODEL} --resnet50_pretrain=${RESNET50_PRETRAIN} --feat_dim=${FEAT_DIM} --model_id=${MODEL_ID} \
--image_size=${IMAGE_SIZE} --lr=${LR} --batch_size=${BATCH_SIZE} --split_train_val=${SPLIT_TRAIN_VAL} \
--transform=${TRANSFORM} --rand_aug_n=${RAND_AUG_N} --rand_aug_m=${RAND_AUG_M} --label_smoothing=${LABEL_SMOOTHING} \
--seed=${SEED} --optim=${OPTIM} --max_epoch=${MAX_EPOCH} --scheduler=${SCHEDULER} \
--weight_decay=${WEIGHT_DECAY} \
--steps ${STEPS[@]} \
--alpha_max=${ALPHA_MAX} 



