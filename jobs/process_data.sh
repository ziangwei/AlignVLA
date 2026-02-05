#!/bin/bash
# =============================================================================
# AlignVLA CPU数据处理作业
# =============================================================================
# 使用方法：
#   1. 解注释你要运行的步骤
#   2. sbatch jobs/process_data.sh
#
# 步骤顺序：
#   Step 1: 验证环境
#   Step 2: 解析DriveLM数据
#   Step 3: 提取nuScenes轨迹
#   Step 4: 合并数据集
# =============================================================================

#SBATCH --job-name=alignvla_data
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.out

# -----------------------------------------------------------------------------
# 环境设置
# -----------------------------------------------------------------------------
set -e  # 出错时退出

# 项目根目录
PROJECT_ROOT="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/AlignVLA"
cd $PROJECT_ROOT

# 激活conda环境
source ~/.bashrc
conda activate alignvla

# 打印环境信息
echo "=================================================="
echo "AlignVLA 数据处理作业"
echo "=================================================="
echo "时间: $(date)"
echo "节点: $(hostname)"
echo "工作目录: $(pwd)"
echo "Python: $(which python)"
echo "=================================================="

# -----------------------------------------------------------------------------
# Step 1: 验证环境（首次运行时解注释）
# -----------------------------------------------------------------------------
echo "Step 1: 验证环境..."
python setup_check.py

# -----------------------------------------------------------------------------
# Step 2: 解析DriveLM数据
# -----------------------------------------------------------------------------
echo "Step 2: 解析DriveLM数据..."
python scripts/data_processing/parse_drivelm.py \
    --input data/drivelm/raw/v1_1_train_nus.json \
    --output data/drivelm/processed/drivelm_parsed.jsonl \
    --nuscenes_root data/nuscenes

# -----------------------------------------------------------------------------
# Step 3: 提取nuScenes轨迹并生成额外样本
# -----------------------------------------------------------------------------
echo "Step 3: 提取nuScenes轨迹..."
python scripts/data_processing/extract_trajectory.py \
    --nuscenes_root data/nuscenes \
    --output data/nuscenes/trajectories.jsonl

# echo "Step 3b: 生成额外简化样本..."
# python scripts/data_processing/generate_simple_samples.py \
#     --nuscenes_root data/nuscenes \
#     --trajectories data/nuscenes/trajectories.jsonl \
#     --drivelm_frames data/drivelm/processed/drivelm_parsed.jsonl \
#     --output data/nuscenes/simple_samples.jsonl \
#     --num_samples 6000

# -----------------------------------------------------------------------------
# Step 4: 合并数据集
# -----------------------------------------------------------------------------
# echo "Step 4: 合并数据集..."
# python scripts/data_processing/merge_dataset.py \
#     --drivelm data/drivelm/processed/drivelm_parsed.jsonl \
#     --nuscenes_simple data/nuscenes/simple_samples.jsonl \
#     --output data/merged/train.jsonl \
#     --val_ratio 0.05

echo "=================================================="
echo "作业完成: $(date)"
echo "=================================================="