#!/bin/bash
 
export PYTHONPATH='/root/second.pytorch'



function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}
rnd=$(rand 1 50)


CUDA_VISIBLE_DEVICES=7 python ./pytorch/engine/train.py train --config_path=./configs/FCOS/xyres_20.config --model_dir=./test/baby_9 --resume

