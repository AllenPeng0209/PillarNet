#!/bin/bash
 
export PYTHONPATH='/root/second.pytorch'



function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}
rnd=$(rand 1 50)


CUDA_VISIBLE_DEVICES=0 python ../../pytorch/engine/train.py train --config_path=../../configs/FCOS/all/xyres_20.config --model_dir=../../test/all_xy20  --resume


