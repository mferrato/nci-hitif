#!/bin/bash

module load singularity openmpi/3.0.0/gcc-7.3.0

mpiexec -n $SLURM_NTASKS -x NCCL_P2P_DISABLE=1 singularity exec --nv ../singularity/horovod.simg python ../src/run_unet_hvd.py ../data/imgs_train.npy ../data/binaries_train.npy --nlayers=5  --conv_size=3  --activation=relu  --loss_func=dice  --num_filters=32  --last_act=sigmoid --batch_size=5 
