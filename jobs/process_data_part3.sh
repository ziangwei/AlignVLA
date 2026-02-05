#!/bin/bash
# =============================================================================
# AlignVLA 数据处理作业 - Part 3 (Step 3b + Step 4)
# =============================================================================
# 前置条件：
#   - Step 1-3 已完成
#   - data/drivelm/processed/drivelm_parsed.jsonl 存在
#   - data/nuscenes/trajectories.jsonl 存在
# =============================================================================

#SBATCH --job-name=alignvla_merge
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.out

# -----------------------------------------------------------------------------
# 环境设置
# -----------------------------------------------------------------------------
set -e

PROJECT_ROOT="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/AlignVLA"
cd $PROJECT_ROOT

source ~/.bashrc
conda activate alignvla

echo "=================================================="
echo "AlignVLA 数据合并作业 (Part 3)"
echo "=================================================="
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "=================================================="

# 检查前置文件
echo "检查前置文件..."
if [ ! -f "data/drivelm/processed/drivelm_parsed.jsonl" ]; then
    echo "Error: drivelm_parsed.jsonl not found!"
    exit 1
fi

if [ ! -f "data/nuscenes/trajectories.jsonl" ]; then
    echo "Error: trajectories.jsonl not found!"
    exit 1
fi

echo "前置文件检查通过"

# -----------------------------------------------------------------------------
# Step 3b: 生成额外简化样本
# -----------------------------------------------------------------------------
echo ""
echo "Step 3b: 生成额外简化样本..."
python scripts/data_processing/generate_simple_samples.py \
    --trajectories data/nuscenes/trajectories.jsonl \
    --drivelm_frames data/drivelm/processed/drivelm_parsed.jsonl \
    --output data/nuscenes/simple_samples.jsonl \
    --num_samples 6000

# -----------------------------------------------------------------------------
# Step 4: 合并数据集
# -----------------------------------------------------------------------------
echo ""
echo "Step 4: 合并数据集..."
python scripts/data_processing/merge_dataset.py \
    --drivelm data/drivelm/processed/drivelm_parsed.jsonl \
    --nuscenes_simple data/nuscenes/simple_samples.jsonl \
    --trajectories data/nuscenes/trajectories.jsonl \
    --output data/merged \
    --val_ratio 0.05

# -----------------------------------------------------------------------------
# 验证输出
# -----------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "验证输出文件"
echo "=================================================="

echo "简化样本:"
wc -l data/nuscenes/simple_samples.jsonl

echo ""
echo "训练集:"
wc -l data/merged/train.jsonl

echo ""
echo "验证集:"
wc -l data/merged/val.jsonl

echo ""
echo "=================================================="
echo "数据处理全部完成!"
echo "=================================================="
echo "完成时间: $(date)"
echo ""
echo "下一步: 运行训练脚本"
echo "  sbatch jobs/train_sft.sh"
echo "=================================================="