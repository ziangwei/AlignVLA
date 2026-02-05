#!/usr/bin/env python3
"""
简化样本生成脚本
================
为非DriveLM覆盖的nuScenes帧生成简化的场景描述

策略：
- DriveLM覆盖了4072帧，有丰富的QA描述
- 剩余约27000帧只有轨迹，没有文本描述
- 为其中6000帧生成模板化的简化描述

简化描述格式：
- scene_analysis: 基于行为和速度的模板描述
- risk_assessment: 简单的风险声明
- decision: 基于behavior的决策
- behavior: 直接使用推断的behavior

采样策略：
- 按behavior类型分层采样，确保各类行为都有代表
- 排除已被DriveLM覆盖的帧
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import random
import jsonlines
from tqdm import tqdm


# 场景描述模板
SCENE_TEMPLATES = {
    "Go straight": [
        "The ego vehicle is driving on a clear road. Traffic conditions are normal.",
        "The road ahead is clear. The ego vehicle maintains its lane.",
        "Normal driving conditions. The ego vehicle proceeds forward.",
    ],
    "Slow down": [
        "The ego vehicle is decelerating. Traffic or obstacles ahead require reduced speed.",
        "The ego vehicle is slowing down due to traffic conditions.",
        "Speed reduction in progress. The ego vehicle is approaching a slower section.",
    ],
    "Turn left": [
        "The ego vehicle is approaching a left turn. Preparing to change direction.",
        "Left turn maneuver in progress. The ego vehicle is navigating the intersection.",
        "The ego vehicle is executing a left turn at the junction.",
    ],
    "Turn right": [
        "The ego vehicle is approaching a right turn. Preparing to change direction.",
        "Right turn maneuver in progress. The ego vehicle is navigating the intersection.",
        "The ego vehicle is executing a right turn at the junction.",
    ],
    "Change lane left": [
        "The ego vehicle is changing to the left lane. Checking blind spots.",
        "Lane change to the left in progress. The ego vehicle is moving laterally.",
        "The ego vehicle is merging into the left lane.",
    ],
    "Change lane right": [
        "The ego vehicle is changing to the right lane. Checking blind spots.",
        "Lane change to the right in progress. The ego vehicle is moving laterally.",
        "The ego vehicle is merging into the right lane.",
    ],
}

# 风险评估模板
RISK_TEMPLATES = {
    "Go straight": "No immediate risks. Maintain current speed and lane position.",
    "Slow down": "Potential obstacle or traffic ahead. Reduce speed for safety.",
    "Turn left": "Check for oncoming traffic and pedestrians before completing the turn.",
    "Turn right": "Watch for pedestrians and cyclists on the right side.",
    "Change lane left": "Ensure left lane is clear. Check mirrors and blind spots.",
    "Change lane right": "Ensure right lane is clear. Check mirrors and blind spots.",
}

# 决策模板
DECISION_TEMPLATES = {
    "Go straight": "Target action: Go straight. Maintain current speed and heading.",
    "Slow down": "Target action: Decelerate. Reduce speed gradually.",
    "Turn left": "Target action: Turn left. Execute turn when safe.",
    "Turn right": "Target action: Turn right. Execute turn when safe.",
    "Change lane left": "Target action: Change lane left. Merge when gap is available.",
    "Change lane right": "Target action: Change lane right. Merge when gap is available.",
}

# 行为描述模板
BEHAVIOR_TEMPLATES = {
    "Go straight": [
        "The ego vehicle is going straight.",
        "The ego vehicle maintains forward motion.",
        "The ego vehicle is driving forward.",
    ],
    "Slow down": [
        "The ego vehicle is slowing down.",
        "The ego vehicle is decelerating.",
        "The ego vehicle is reducing speed.",
    ],
    "Turn left": [
        "The ego vehicle is turning left.",
        "The ego vehicle is making a left turn.",
    ],
    "Turn right": [
        "The ego vehicle is turning right.",
        "The ego vehicle is making a right turn.",
    ],
    "Change lane left": [
        "The ego vehicle is changing to the left lane.",
        "The ego vehicle is moving to the left lane.",
    ],
    "Change lane right": [
        "The ego vehicle is changing to the right lane.",
        "The ego vehicle is moving to the right lane.",
    ],
}


def add_speed_context(text: str, ego_speed: float) -> str:
    """根据速度添加上下文"""
    if ego_speed < 1.0:
        return text + " The vehicle is nearly stationary."
    elif ego_speed < 5.0:
        return text + " The vehicle is moving slowly."
    elif ego_speed < 10.0:
        return text + " The vehicle is at moderate speed."
    else:
        return text + " The vehicle is moving at higher speed."


def generate_simple_description(behavior: str, ego_speed: float) -> Dict[str, str]:
    """
    生成简化的场景描述
    """
    # 随机选择模板
    scene = random.choice(SCENE_TEMPLATES.get(behavior, SCENE_TEMPLATES["Go straight"]))
    scene = add_speed_context(scene, ego_speed)
    
    risk = RISK_TEMPLATES.get(behavior, RISK_TEMPLATES["Go straight"])
    decision = DECISION_TEMPLATES.get(behavior, DECISION_TEMPLATES["Go straight"])
    behavior_desc = random.choice(BEHAVIOR_TEMPLATES.get(behavior, BEHAVIOR_TEMPLATES["Go straight"]))
    
    # 添加速度信息到behavior
    if ego_speed > 0:
        behavior_desc += f" Current speed: {ego_speed:.1f} m/s."
    
    return {
        "scene_analysis": scene,
        "risk_assessment": risk,
        "decision": decision,
        "behavior": behavior_desc,
    }


def load_drivelm_frame_tokens(drivelm_path: Path) -> Set[str]:
    """
    加载DriveLM已覆盖的frame tokens
    """
    covered_tokens = set()
    
    with jsonlines.open(drivelm_path) as reader:
        for item in reader:
            # DriveLM的frame_token对应nuScenes的sample_token
            covered_tokens.add(item["frame_token"])
    
    return covered_tokens


def generate_simple_samples(
    trajectories_path: Path,
    drivelm_path: Path,
    output_path: Path,
    num_samples: int = 6000,
    seed: int = 42,
) -> Dict[str, int]:
    """
    生成简化样本
    """
    random.seed(seed)
    
    # 加载DriveLM覆盖的帧
    print("Loading DriveLM covered frames...")
    covered_tokens = load_drivelm_frame_tokens(drivelm_path)
    print(f"  DriveLM covers {len(covered_tokens)} frames")
    
    # 加载所有轨迹
    print("Loading trajectories...")
    all_trajectories = []
    with jsonlines.open(trajectories_path) as reader:
        for item in reader:
            # 排除DriveLM已覆盖的帧
            if item["sample_token"] not in covered_tokens:
                all_trajectories.append(item)
    
    print(f"  Available for simple samples: {len(all_trajectories)}")
    
    # 按behavior分组
    behavior_groups = defaultdict(list)
    for traj in all_trajectories:
        behavior_groups[traj["behavior"]].append(traj)
    
    print("  Behavior distribution (available):")
    for behavior, items in sorted(behavior_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {behavior}: {len(items)}")
    
    # 分层采样
    # 目标：6000样本，按比例分配，但确保少数类至少有一定数量
    total_available = len(all_trajectories)
    samples_per_behavior = {}
    
    # 计算每个behavior的目标数量
    for behavior, items in behavior_groups.items():
        # 按比例分配，但最少100个（如果有的话）
        proportion = len(items) / total_available
        target = max(100, int(num_samples * proportion))
        samples_per_behavior[behavior] = min(target, len(items))
    
    # 调整总数到目标
    current_total = sum(samples_per_behavior.values())
    if current_total > num_samples:
        # 按比例缩减
        scale = num_samples / current_total
        for behavior in samples_per_behavior:
            samples_per_behavior[behavior] = int(samples_per_behavior[behavior] * scale)
    
    print(f"  Sampling plan:")
    for behavior, count in sorted(samples_per_behavior.items(), key=lambda x: -x[1]):
        print(f"    {behavior}: {count}")
    
    # 执行采样
    selected_samples = []
    for behavior, count in samples_per_behavior.items():
        sampled = random.sample(behavior_groups[behavior], count)
        selected_samples.extend(sampled)
    
    random.shuffle(selected_samples)
    
    # 生成描述
    print(f"Generating descriptions for {len(selected_samples)} samples...")
    results = []
    
    for traj in tqdm(selected_samples, desc="Generating"):
        description = generate_simple_description(traj["behavior"], traj["ego_speed"])
        
        result = {
            "sample_id": f"simple__{traj['sample_token']}",
            "sample_token": traj["sample_token"],
            "scene_token": traj["scene_token"],
            "images": traj["images"],
            "scene_analysis": description["scene_analysis"],
            "risk_assessment": description["risk_assessment"],
            "decision": description["decision"],
            "behavior": description["behavior"],        # 描述性文本（用于模型输出）
            "behavior_label": traj["behavior"],         # 原始标签（用于导航指令和统计）
            "ego_speed": traj["ego_speed"],
            "trajectory_3s": traj["trajectory_3s"],
            "trajectory_6s": traj["trajectory_6s"],
            "source": "nuscenes_simple",
        }
        results.append(result)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(results)
    
    # 统计
    stats = {
        "total_available": len(all_trajectories),
        "generated": len(results),
        "behavior_distribution": {b: sum(1 for r in results if traj["behavior"] == b) 
                                   for b in samples_per_behavior},
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="生成简化样本")
    parser.add_argument(
        "--trajectories",
        type=str,
        default="data/nuscenes/trajectories.jsonl",
        help="轨迹文件路径"
    )
    parser.add_argument(
        "--drivelm_frames",
        type=str,
        default="data/drivelm/processed/drivelm_parsed.jsonl",
        help="DriveLM解析结果路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nuscenes/simple_samples.jsonl",
        help="输出文件路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6000,
        help="生成样本数量"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    trajectories_path = Path(args.trajectories)
    drivelm_path = Path(args.drivelm_frames)
    output_path = Path(args.output)
    
    if not trajectories_path.exists():
        print(f"Error: Trajectories file not found: {trajectories_path}")
        return 1
    
    if not drivelm_path.exists():
        print(f"Error: DriveLM file not found: {drivelm_path}")
        return 1
    
    stats = generate_simple_samples(
        trajectories_path,
        drivelm_path,
        output_path,
        args.num_samples,
        args.seed,
    )
    
    print("\n" + "=" * 50)
    print("简化样本生成完成")
    print("=" * 50)
    print(f"可用样本: {stats['total_available']}")
    print(f"生成样本: {stats['generated']}")
    print(f"输出文件: {output_path}")
    
    # 显示样例
    print("\n" + "=" * 50)
    print("样例输出:")
    print("=" * 50)
    with jsonlines.open(output_path) as reader:
        for i, item in enumerate(reader):
            if i >= 1:
                break
            # 只显示关键字段
            display = {
                "sample_id": item["sample_id"],
                "behavior": item["behavior"],
                "scene_analysis": item["scene_analysis"],
                "decision": item["decision"],
                "trajectory_3s": item["trajectory_3s"][:2],  # 只显示前2个点
            }
            print(json.dumps(display, indent=2, ensure_ascii=False))
    
    return 0


if __name__ == "__main__":
    exit(main())