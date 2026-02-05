#!/usr/bin/env python3
"""
数据合并脚本
============
合并DriveLM和简化样本，生成最终训练数据

功能：
1. 为DriveLM样本匹配轨迹信息
2. 合并DriveLM（专家级）和简化样本
3. 生成统一的训练格式（包含prompt和response）
4. 划分训练集和验证集
5. 应用轨迹tokenization

最终格式：
{
    "id": "...",
    "images": ["path1", "path2", ...],  # 6个相机图像
    "instruction": "分析当前驾驶场景...",
    "input": "当前车速: X.X m/s\n导航指令: Go straight",
    "output": "<scene_analysis>...</scene_analysis>\n<risk_assessment>...</risk_assessment>\n<decision>...</decision>\n<trajectory><X10><Y20> ...</trajectory>",
    "source": "drivelm" | "nuscenes_simple"
}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random
import jsonlines
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils.trajectory_tokenizer import TrajectoryTokenizer


# 相机顺序
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT"
]

# 系统指令
SYSTEM_INSTRUCTION = """You are an autonomous driving planning system. Analyze the current driving scene from 6 camera views and plan the trajectory for the next 3 seconds.

Output format:
<scene_analysis>Describe the current scene, including road conditions, surrounding vehicles, pedestrians, and traffic signals.</scene_analysis>
<risk_assessment>Identify potential risks and objects that require attention.</risk_assessment>
<decision>State the driving decision and reasoning.</decision>
<trajectory>Output 6 waypoints in vehicle coordinates (x: forward, y: left) using trajectory tokens.</trajectory>"""

# 用户输入模板
USER_INPUT_TEMPLATE = """Current speed: {ego_speed:.1f} m/s
Navigation instruction: {behavior}

Analyze the scene and plan the trajectory."""


def load_trajectories_index(trajectories_path: Path) -> Dict[str, Dict]:
    """
    加载轨迹并建立sample_token索引
    """
    index = {}
    with jsonlines.open(trajectories_path) as reader:
        for item in reader:
            index[item["sample_token"]] = item
    return index


def format_output(
    scene_analysis: str,
    risk_assessment: str,
    decision: str,
    trajectory_3s: List[Dict],
    tokenizer: TrajectoryTokenizer,
) -> str:
    """
    格式化模型输出
    """
    # Tokenize轨迹
    traj_tuples = [(p["x"], p["y"]) for p in trajectory_3s]
    trajectory_tokens = tokenizer.encode_trajectory(traj_tuples)
    
    output = f"""<scene_analysis>{scene_analysis}</scene_analysis>
<risk_assessment>{risk_assessment}</risk_assessment>
<decision>{decision}</decision>
<trajectory>{trajectory_tokens}</trajectory>"""
    
    return output


def process_drivelm_sample(
    sample: Dict,
    trajectory_info: Optional[Dict],
    tokenizer: TrajectoryTokenizer,
) -> Optional[Dict]:
    """
    处理DriveLM样本
    """
    if not trajectory_info:
        return None
    
    # 检查轨迹长度
    if len(trajectory_info.get("trajectory_3s", [])) < 3:
        return None
    
    # 构建图像列表（按固定顺序）
    images = [sample["images"][cam] for cam in CAMERAS]
    
    # 用户输入
    user_input = USER_INPUT_TEMPLATE.format(
        ego_speed=trajectory_info["ego_speed"],
        behavior=trajectory_info["behavior"],
    )
    
    # 模型输出
    output = format_output(
        scene_analysis=sample["scene_analysis"],
        risk_assessment=sample["risk_assessment"],
        decision=sample["decision"],
        trajectory_3s=trajectory_info["trajectory_3s"],
        tokenizer=tokenizer,
    )
    
    return {
        "id": sample["sample_id"],
        "sample_token": sample.get("frame_token", sample.get("sample_token")),
        "scene_token": sample["scene_token"],
        "images": images,
        "instruction": SYSTEM_INSTRUCTION,
        "input": user_input,
        "output": output,
        "behavior": trajectory_info["behavior"],
        "ego_speed": trajectory_info["ego_speed"],
        "trajectory_3s": trajectory_info["trajectory_3s"],
        "source": "drivelm",
    }


def process_simple_sample(
    sample: Dict,
    tokenizer: TrajectoryTokenizer,
) -> Optional[Dict]:
    """
    处理简化样本
    """
    # 检查轨迹长度
    if len(sample.get("trajectory_3s", [])) < 3:
        return None
    
    # 构建图像列表
    images = [sample["images"][cam] for cam in CAMERAS]
    
    # 使用behavior_label作为导航指令（如果存在），否则回退到behavior
    behavior_label = sample.get("behavior_label", sample.get("behavior", "Go straight"))
    
    # 用户输入
    user_input = USER_INPUT_TEMPLATE.format(
        ego_speed=sample["ego_speed"],
        behavior=behavior_label,
    )
    
    # 模型输出（使用behavior描述，不是标签）
    output = format_output(
        scene_analysis=sample["scene_analysis"],
        risk_assessment=sample["risk_assessment"],
        decision=sample["decision"],
        trajectory_3s=sample["trajectory_3s"],
        tokenizer=tokenizer,
    )
    
    return {
        "id": sample["sample_id"],
        "sample_token": sample["sample_token"],
        "scene_token": sample["scene_token"],
        "images": images,
        "instruction": SYSTEM_INSTRUCTION,
        "input": user_input,
        "output": output,
        "behavior": behavior_label,  # 用标签，方便统计
        "ego_speed": sample["ego_speed"],
        "trajectory_3s": sample["trajectory_3s"],
        "source": "nuscenes_simple",
    }


def merge_datasets(
    drivelm_path: Path,
    simple_path: Path,
    trajectories_path: Path,
    output_dir: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Dict[str, int]:
    """
    合并数据集
    """
    random.seed(seed)
    
    # 初始化tokenizer
    tokenizer = TrajectoryTokenizer()
    
    # 加载轨迹索引
    print("Loading trajectories index...")
    traj_index = load_trajectories_index(trajectories_path)
    print(f"  Loaded {len(traj_index)} trajectories")
    
    # 处理DriveLM样本
    print("Processing DriveLM samples...")
    drivelm_samples = []
    drivelm_skipped = 0
    
    with jsonlines.open(drivelm_path) as reader:
        for sample in tqdm(reader, desc="DriveLM"):
            # 用frame_token匹配轨迹
            frame_token = sample["frame_token"]
            trajectory_info = traj_index.get(frame_token)
            
            processed = process_drivelm_sample(sample, trajectory_info, tokenizer)
            if processed:
                drivelm_samples.append(processed)
            else:
                drivelm_skipped += 1
    
    print(f"  Processed: {len(drivelm_samples)}, Skipped: {drivelm_skipped}")
    
    # 处理简化样本
    print("Processing simple samples...")
    simple_samples = []
    simple_skipped = 0
    
    with jsonlines.open(simple_path) as reader:
        for sample in tqdm(reader, desc="Simple"):
            processed = process_simple_sample(sample, tokenizer)
            if processed:
                simple_samples.append(processed)
            else:
                simple_skipped += 1
    
    print(f"  Processed: {len(simple_samples)}, Skipped: {simple_skipped}")
    
    # 合并
    all_samples = drivelm_samples + simple_samples
    random.shuffle(all_samples)
    
    print(f"Total samples: {len(all_samples)}")
    
    # 划分训练集和验证集
    val_size = int(len(all_samples) * val_ratio)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    with jsonlines.open(train_path, mode="w") as writer:
        writer.write_all(train_samples)
    
    with jsonlines.open(val_path, mode="w") as writer:
        writer.write_all(val_samples)
    
    # 统计
    stats = {
        "drivelm_processed": len(drivelm_samples),
        "drivelm_skipped": drivelm_skipped,
        "simple_processed": len(simple_samples),
        "simple_skipped": simple_skipped,
        "total": len(all_samples),
        "train": len(train_samples),
        "val": len(val_samples),
    }
    
    # 保存统计信息
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="合并数据集")
    parser.add_argument(
        "--drivelm",
        type=str,
        default="data/drivelm/processed/drivelm_parsed.jsonl",
        help="DriveLM解析结果路径"
    )
    parser.add_argument(
        "--nuscenes_simple",
        type=str,
        default="data/nuscenes/simple_samples.jsonl",
        help="简化样本路径"
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        default="data/nuscenes/trajectories.jsonl",
        help="轨迹文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/merged",
        help="输出目录"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="验证集比例"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    drivelm_path = Path(args.drivelm)
    simple_path = Path(args.nuscenes_simple)
    trajectories_path = Path(args.trajectories)
    output_dir = Path(args.output)
    
    for path, name in [(drivelm_path, "DriveLM"), 
                       (simple_path, "Simple samples"),
                       (trajectories_path, "Trajectories")]:
        if not path.exists():
            print(f"Error: {name} not found: {path}")
            return 1
    
    stats = merge_datasets(
        drivelm_path,
        simple_path,
        trajectories_path,
        output_dir,
        args.val_ratio,
        args.seed,
    )
    
    print("\n" + "=" * 50)
    print("数据合并完成")
    print("=" * 50)
    print(f"DriveLM样本: {stats['drivelm_processed']} (跳过 {stats['drivelm_skipped']})")
    print(f"简化样本: {stats['simple_processed']} (跳过 {stats['simple_skipped']})")
    print(f"总样本数: {stats['total']}")
    print(f"训练集: {stats['train']}")
    print(f"验证集: {stats['val']}")
    print(f"输出目录: {output_dir}")
    
    # 显示样例
    print("\n" + "=" * 50)
    print("训练样例:")
    print("=" * 50)
    with jsonlines.open(output_dir / "train.jsonl") as reader:
        for i, item in enumerate(reader):
            if i >= 1:
                break
            # 显示关键字段
            print(f"ID: {item['id']}")
            print(f"Source: {item['source']}")
            print(f"Behavior: {item['behavior']}")
            print(f"Images: {item['images'][0]}...")  # 只显示第一个
            print(f"\n--- Input ---")
            print(item['input'])
            print(f"\n--- Output ---")
            print(item['output'][:500] + "..." if len(item['output']) > 500 else item['output'])
    
    # 行为分布
    print("\n" + "=" * 50)
    print("训练集行为分布:")
    print("=" * 50)
    behavior_counts = defaultdict(int)
    source_counts = defaultdict(int)
    
    with jsonlines.open(output_dir / "train.jsonl") as reader:
        for item in reader:
            behavior_counts[item["behavior"]] += 1
            source_counts[item["source"]] += 1
    
    for behavior, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
        pct = count / stats['train'] * 100
        print(f"  {behavior}: {count} ({pct:.1f}%)")
    
    print("\n数据来源分布:")
    for source, count in source_counts.items():
        pct = count / stats['train'] * 100
        print(f"  {source}: {count} ({pct:.1f}%)")
    
    return 0


if __name__ == "__main__":
    exit(main())