#!/bin/bash

python=python3
dataset=$1
exp_dir=./exps_$dataset
exp_name=$dataset
data_dir=../../datasets/$dataset/
relation_file=$data_dir/relations.dict

beam_size=30
cat $relation_file | awk -F "\t" '{print $2}' | while read line
do
$python -u rule_extraction_chain.py $exp_dir $exp_name $line ori $beam_size
break
$python -u rule_extraction_chain.py $exp_dir $exp_name $line inv $beam_size
done
