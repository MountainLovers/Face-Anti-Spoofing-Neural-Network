#!/bin/bash
# Usage: ./copyfile.sh [type]
# type: dev/train/test
# 从原始数据集中根据txt拷贝数据构造小数据集
process_type=$1

listfile_prefix="/workspace/Face-Anti-Spoofing-Neural-Network/SMALLOULU/"
originpath_prefix="/mnt/hdd.user/datasets/FAS/Oulu-NPU/"

if [ $process_type == "dev" ]
then
    dstpath="${listfile_prefix}Dev_files/"
    listfile="${dstpath}dev.txt"
    originpath="${originpath_prefix}Dev_files/"
elif [ $process_type == "train" ]
then
    dstpath="${listfile_prefix}Train_files/"
    listfile="${dstpath}train.txt"
    originpath="${originpath_prefix}Train_files/"
elif [ $process_type == "test" ]
then
    dstpath="${listfile_prefix}Test_files/"
    listfile="${dstpath}test.txt"
    originpath="${originpath_prefix}Test_files/"
else
    echo "Process_type ERROR!"
    exit
fi

for line in `cat $listfile`
do
    srcpath="${originpath}${line}"
    cp ${srcpath}".avi" $dstpath
    cp ${srcpath}".txt" $dstpath
    echo $srcpath
done