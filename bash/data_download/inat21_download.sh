#!/bin/bash

# Check if the data_dir argument is provided
if [ -z "$1" ]; then
  # If no argument is provided, set data_dir to the current working directory
  :
else
  # If argument is provided, set data_dir to the provided argument
  data_dir=$1
  cd $data_dir
fi


# train mini
echo "downloading TRAIN MINI"
curl https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz --output train_mini.tar.gz
curl https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz --output train_mini.json.tar.gz


# validation
echo "downloading VAL"
curl https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz --output val.tar.gz
curl https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz --output val.json.tar.gz


# test
echo "downloading TEST"
curl https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.json.tar.gz --output public_test.json.tar.gz
curl https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz --output public_test.tar.gz


# train
echo "downloading TRAIN"
curl https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz --output train.json.tar.gz
curl https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz --output train.tar.gz 

echo "done"

