#!/bin/bash

module load singularity

singularity exec --nv ../singularity/candle-gpu.img python ../src/run_unet.py ../data/imgs_train.npy ../data/binaries_train.npy --nlayers=5  --conv_size=3  --activation=relu  --loss_func=dice  --num_filters=32  --last_act=sigmoid 
