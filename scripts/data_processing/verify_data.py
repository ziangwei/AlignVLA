#!/usr/bin/env python3
"""
数据验证脚本
============
检查合并后的数据是否正确
"""

import json
import jsonlines
from pathlib import Path
from collections import Counter
import re


def main():
    merged_dir = Path("data/merged")
    train_path = merged_dir / "train.jsonl"
    
    if not train_path.exists():
        print(f"Error: {train_path} not found")
        return 1
    
    print("=" * 60)
    print("数据验证")
    print("=" * 60)
    
    # 统计
    total = 0
    has_trajectory_token = 0
    behavior_counts = Counter()
    source_counts = Counter()
    trajectory_lengths = []
    
    # 检查样例
    samples_to_show = []
    
    with jsonlines.open(train_path) as reader:
        for item in reader:
            total += 1
            
            # 检查trajectory token
            output = item.get("output", "")
            traj_match = re.search(r"<trajectory>(.*?)</trajectory>", output, re.DOTALL)
            
            if traj_match:
                traj_content = traj_match.group(1).strip()
                # 检查是否包含X和Y token
                if "<X" in traj_content and "<Y" in traj_content:
                    has_trajectory_token += 1
                    # 统计waypoint数量
                    waypoints = re.findall(r"<X\d+><Y\d+>", traj_content)
                    trajectory_lengths.append(len(waypoints))
            
            # 行为统计
            behavior_counts[item.get("behavior", "unknown")] += 1
            source_counts[item.get("source", "unknown")] += 1
            
            # 收集样例（每种来源各一个）
            if len(samples_to_show) < 2:
                if item["source"] not in [s["source"] for s in samples_to_show]:
                    samples_to_show.append(item)
    
    # 打印结果
    print(f"\n总样本数: {total}")
    print(f"包含轨迹token: {has_trajectory_token} ({has_trajectory_token/total*100:.1f}%)")
    
    if trajectory_lengths:
        avg_len = sum(trajectory_lengths) / len(trajectory_lengths)
        print(f"平均轨迹长度: {avg_len:.1f} waypoints")
    
    print(f"\n行为分布:")
    for behavior, count in behavior_counts.most_common():
        print(f"  {behavior}: {count} ({count/total*100:.1f}%)")
    
    print(f"\n数据来源分布:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} ({count/total*100:.1f}%)")
    
    # 显示样例
    print("\n" + "=" * 60)
    print("样例展示")
    print("=" * 60)
    
    for sample in samples_to_show:
        print(f"\n--- {sample['source']} 样例 ---")
        print(f"ID: {sample['id']}")
        print(f"Behavior: {sample['behavior']}")
        print(f"\n[Input]")
        print(sample['input'])
        print(f"\n[Output]")
        print(sample['output'])
        print("-" * 40)
    
    # 检查DriveLM和nuScenes的映射
    print("\n" + "=" * 60)
    print("DriveLM映射检查")
    print("=" * 60)
    
    # 加载DriveLM原始数据的frame_tokens
    drivelm_path = Path("data/drivelm/processed/drivelm_parsed.jsonl")
    traj_path = Path("data/nuscenes/trajectories.jsonl")
    
    if drivelm_path.exists() and traj_path.exists():
        # 获取所有轨迹的sample_token
        traj_tokens = set()
        with jsonlines.open(traj_path) as reader:
            for item in reader:
                traj_tokens.add(item["sample_token"])
        
        # 检查DriveLM的frame_token匹配情况
        matched = 0
        unmatched = []
        with jsonlines.open(drivelm_path) as reader:
            for item in reader:
                frame_token = item["frame_token"]
                if frame_token in traj_tokens:
                    matched += 1
                else:
                    if len(unmatched) < 5:
                        unmatched.append(frame_token)
        
        print(f"DriveLM frame_tokens: 4072")
        print(f"成功匹配轨迹: {matched}")
        print(f"未匹配: {4072 - matched}")
        
        if unmatched:
            print(f"未匹配样例: {unmatched[:3]}")
    
    return 0


if __name__ == "__main__":
    exit(main())