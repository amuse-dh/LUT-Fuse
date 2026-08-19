#!/bin/bash
#SBATCH -J etri 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 1-0
#SBATCH -o /ceph_data/fovert/etri_dh/LUT-Fuse/logs/slurm-%A.out

mkdir -p /ceph_data/fovert/etri_dh/LUT-Fuse/logs
cd /ceph_data/fovert/etri_dh/LUT-Fuse

source /ceph_data/fovert/anaconda3/etc/profile.d/conda.sh
conda activate lutfuse

export TMPDIR=/ceph_data/fovert/tmp
export TORCH_HOME=/ceph_data/fovert/.cache/torch
export XDG_CACHE_HOME=/ceph_data/fovert/.cache
mkdir -p "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME"

python test_lut.py \
  --visible_dir /ceph_data/fovert/etri_dh/ETRI_Night_Fusion/kiro_night/RGB \
  --infrared_dir /ceph_data/fovert/etri_dh/ETRI_Night_Fusion/kiro_night/Thermal \
  --save_dir /ceph_data/fovert/etri_dh/LUT-Fuse/results_kiro