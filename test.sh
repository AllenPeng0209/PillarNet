


export PATHONPATH="/root/second.pytorch"
CUDA_VISIBLE_DEVICES=6 gdb -ex r --args python ./pytorch/train.py train --config_path=./configs/FCOS/xyres_20.config  --model_dir=./test/debug2

