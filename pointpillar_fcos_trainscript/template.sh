#!/bin/bash
 
export PYTHONPATH='/root/second.pytorch'



function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}
rnd=$(rand 1 50)


CUDA_VISIBLE_DEVICES=2 python ../pytorch/engine/train.py train --config_path=../configs/FCOS/ped_cycle/xyres_20.config --model_dir=../test/ped_cyclist_v3_20 --resume 


