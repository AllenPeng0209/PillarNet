#!/bin/bash
 
export PYTHONPATH='/root/second.pytorch'

python ../../pytorch/engine/train.py train --config_path=../../configs/FCOS/car/xyres_12.config --model_dir=/cephfs/person/yanlunpeng/experiment_result/pointpillar_fcos/car/ --resume

