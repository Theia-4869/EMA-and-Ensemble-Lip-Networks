#!/bin/bash
#SBATCH --partition=orion --qos=normal
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:titanxp:1

#SBATCH --job-name="Lipnet++_mnist"
#SBATCH --output=/orion/u/hewang/qizhe/myLipnet/log/mnist++.txt

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

srun python main.py --model 'MLPFeature(depth=4,width=5)' --dataset MNIST --predictor-hidden-size 256 --loss 'cross_entropy' --p-start 8 --p-end 1000 --epochs 0,50,50,350,400 --kappa 0.5 --eps-test 0.3 --eps-train 0.35 -b 512 --lr 0.02 --wd 5e-3 --gpu 0 --visualize -p 200