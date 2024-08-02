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

check_md5sum () {
	target_checksum=$1
	checksum=$2
	echo "${target_checksum} (target)"
	echo "${checksum} (file)"
	if [ "${target_checksum}" == "${checksum}" ]; then
	    echo "Checksum is correct"
	else
	    echo "Checksum is NOT correct"
	fi
}

# train mini
echo "TRAIN MINI"
target_checksum="db6ed8330e634445efc8fec83ae81442"
checksum=($(md5sum train_mini.tar.gz))
check_md5sum ${target_checksum} ${checksum}

target_checksum="395a35be3651d86dc3b0d365b8ea5f92"
checksum=($(md5sum train_mini.json.tar.gz))
check_md5sum ${target_checksum} ${checksum}


# val
echo "VAL"
target_checksum="f6f6e0e242e3d4c9569ba56400938afc"
checksum=($(md5sum val.tar.gz))
check_md5sum ${target_checksum} ${checksum}

target_checksum="4d761e0f6a86cc63e8f7afc91f6a8f0b"
checksum=($(md5sum val.json.tar.gz))
check_md5sum ${target_checksum} ${checksum}

# test
echo "TEST"
target_checksum="7124b949fe79bfa7f7019a15ef3dbd06"
checksum=($(md5sum public_test.tar.gz))
check_md5sum ${target_checksum} ${checksum}

target_checksum="7a9413db55c6fa452824469cc7dd9d3d"
checksum=($(md5sum public_test.json.tar.gz))
check_md5sum ${target_checksum} ${checksum}

# train
echo "TRAIN"
target_checksum="e0526d53c7f7b2e3167b2b43bb2690ed"
checksum=($(md5sum train.tar.gz))
check_md5sum ${target_checksum} ${checksum}

target_checksum="38a7bb733f7a09214d44293460ec0021"
checksum=($(md5sum train.json.tar.gz))
check_md5sum ${target_checksum} ${checksum}

echo "done"

