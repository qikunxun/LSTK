#!/bin/bash

python=python3
dataset=DocRED
exp_dir=./exps_$dataset
exp_name=$dataset
batch_size=32
max_epoch=50
data_dir=../../datasets/$dataset/
relation_file=$data_dir/relations.dict
gpu_id=0
cat $relation_file | awk -F "\t" '{print $2}' | while read line
do
$python -u main.py \
        --data_dir $data_dir \
        --exps_dir ./$exp_dir/ \
        --exp_name $exp_name \
        --target_relation $line \
        --batch_size $batch_size \
        --max_epoch $max_epoch \
        --with_constrain \
        --use_gpu \
        --step 4 \
        --gpu_id $gpu_id \
        --delta 0.5 \
        --seed 33
done
$python -u evaluate.py $data_dir $exp_dir $exp_name $batch_size 0 $gpu_id
