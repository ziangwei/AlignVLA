#!/usr/bin/env python3
"""
nuScenes轨迹提取脚本
===================
从nuScenes数据集提取每个sample的未来轨迹和自车信息

输入: nuScenes数据集
输出: data/nuscenes/trajectories.jsonl

输出格式:
{
    "sample_token": "...",
    "scene_token": "...",
    "timestamp": 1234567890,
    "ego_speed": 5.2,  # m/s
    "images": {
        "CAM_FRONT": "samples/CAM_FRONT/xxx.jpg",
        ...
    },
    "trajectory_3s": [  # 未来3秒，6个点，用于模型输出
        {"x": 5.0, "y": 0.1},
        ...
    ],
    "trajectory_6s": [  # 未来6秒，12个点，用于行为推断
        {"x": 5.0, "y": 0.1},
        ...
    ],
    "behavior": "Go straight"  # 推断的行为
}
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import jsonlines
import sys

# 添加项目根目录到path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nuscenes import NuScenes
from pyquaternion import Quaternion

from scripts.utils.trajectory_tokenizer import infer_behavior


# 相机列表
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT"
]


def get_ego_speed(nusc: NuScenes, sample: Dict) -> float:
    """
    计算自车速度
    
    Args:
        nusc: NuScenes实例
        sample: sample记录
        
    Returns:
        速度 (m/s)
    """
    # 获取当前和前一个sample的ego pose
    current_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    current_pose = nusc.get("ego_pose", current_lidar["ego_pose_token"])
    
    # 尝试获取前一个sample
    if sample["prev"]:
        prev_sample = nusc.get("sample", sample["prev"])
        prev_lidar = nusc.get("sample_data", prev_sample["data"]["LIDAR_TOP"])
        prev_pose = nusc.get("ego_pose", prev_lidar["ego_pose_token"])
        
        # 计算位移
        dx = current_pose["translation"][0] - prev_pose["translation"][0]
        dy = current_pose["translation"][1] - prev_pose["translation"][1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # 计算时间差（微秒转秒）
        dt = (current_lidar["timestamp"] - prev_lidar["timestamp"]) / 1e6
        
        if dt > 0:
            return distance / dt
    
    return 0.0


def get_future_trajectory(
    nusc: NuScenes,
    sample: Dict,
    num_waypoints: int,
    time_interval: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    提取未来轨迹（车辆局部坐标系）
    
    Args:
        nusc: NuScenes实例
        sample: 当前sample记录
        num_waypoints: 需要的waypoint数量
        time_interval: 期望的时间间隔（秒）
        
    Returns:
        (trajectory_3s, trajectory_6s): 3秒轨迹和6秒轨迹
    """
    # 获取当前位姿
    current_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    current_pose = nusc.get("ego_pose", current_lidar["ego_pose_token"])
    current_pos = np.array(current_pose["translation"][:2])
    current_rot = Quaternion(current_pose["rotation"])
    current_timestamp = current_lidar["timestamp"]
    
    trajectory = []
    next_token = sample["next"]
    
    # 收集未来12个点（6秒）
    while next_token and len(trajectory) < 12:
        next_sample = nusc.get("sample", next_token)
        next_lidar = nusc.get("sample_data", next_sample["data"]["LIDAR_TOP"])
        next_pose = nusc.get("ego_pose", next_lidar["ego_pose_token"])
        
        # 全局坐标
        next_pos = np.array(next_pose["translation"][:2])
        
        # 转换到当前车辆坐标系
        relative_pos = next_pos - current_pos
        
        # 旋转到车辆坐标系 (x前进, y左)
        # Quaternion的inverse旋转
        rotated = current_rot.inverse.rotate(np.array([relative_pos[0], relative_pos[1], 0]))
        
        trajectory.append({
            "x": round(float(rotated[0]), 2),
            "y": round(float(rotated[1]), 2),
            "timestamp": next_lidar["timestamp"],
        })
        
        next_token = next_sample["next"]
    
    # 分割为3秒和6秒轨迹
    trajectory_3s = trajectory[:6] if len(trajectory) >= 6 else trajectory
    trajectory_6s = trajectory[:12] if len(trajectory) >= 12 else trajectory
    
    # 移除timestamp字段（只在计算时使用）
    trajectory_3s = [{"x": p["x"], "y": p["y"]} for p in trajectory_3s]
    trajectory_6s = [{"x": p["x"], "y": p["y"]} for p in trajectory_6s]
    
    return trajectory_3s, trajectory_6s


def get_image_paths(nusc: NuScenes, sample: Dict) -> Optional[Dict[str, str]]:
    """
    获取所有相机的图像路径
    
    Returns:
        {cam_name: relative_path} 或 None（如果缺少相机）
    """
    images = {}
    
    for cam in CAMERAS:
        if cam not in sample["data"]:
            return None
        
        cam_data = nusc.get("sample_data", sample["data"][cam])
        # 返回相对路径
        images[cam] = cam_data["filename"]
    
    return images


def extract_trajectories(
    nuscenes_root: Path,
    output_path: Path,
    version: str = "v1.0-trainval",
) -> Dict[str, int]:
    """
    提取所有sample的轨迹
    """
    print(f"Loading nuScenes {version} from {nuscenes_root}...")
    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=True)
    
    stats = {
        "total_samples": 0,
        "processed_samples": 0,
        "skipped_short_trajectory": 0,
        "skipped_missing_camera": 0,
    }
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for sample in tqdm(nusc.sample, desc="Extracting trajectories"):
        stats["total_samples"] += 1
        
        # 获取图像路径
        images = get_image_paths(nusc, sample)
        if not images:
            stats["skipped_missing_camera"] += 1
            continue
        
        # 获取轨迹
        trajectory_3s, trajectory_6s = get_future_trajectory(nusc, sample, num_waypoints=12)
        
        # 需要至少3个点才能有意义
        if len(trajectory_3s) < 3:
            stats["skipped_short_trajectory"] += 1
            continue
        
        # 获取速度
        ego_speed = get_ego_speed(nusc, sample)
        
        # 推断行为（使用6秒轨迹）
        traj_for_behavior = [(p["x"], p["y"]) for p in trajectory_6s]
        behavior = infer_behavior(traj_for_behavior, ego_speed)
        
        # 构建结果
        result = {
            "sample_token": sample["token"],
            "scene_token": sample["scene_token"],
            "timestamp": sample["timestamp"],
            "ego_speed": round(ego_speed, 2),
            "images": images,
            "trajectory_3s": trajectory_3s,
            "trajectory_6s": trajectory_6s,
            "behavior": behavior,
        }
        
        results.append(result)
        stats["processed_samples"] += 1
    
    # 保存结果
    print(f"Saving {len(results)} trajectories to {output_path}...")
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(results)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="提取nuScenes轨迹")
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default="data/nuscenes",
        help="nuScenes数据根目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nuscenes/trajectories.jsonl",
        help="输出JSONL文件路径"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes版本"
    )
    
    args = parser.parse_args()
    
    nuscenes_root = Path(args.nuscenes_root)
    output_path = Path(args.output)
    
    if not nuscenes_root.exists():
        print(f"Error: nuScenes root not found: {nuscenes_root}")
        return 1
    
    stats = extract_trajectories(nuscenes_root, output_path, args.version)
    
    print("\n" + "=" * 50)
    print("轨迹提取完成")
    print("=" * 50)
    print(f"总样本数: {stats['total_samples']}")
    print(f"成功处理: {stats['processed_samples']}")
    print(f"轨迹过短: {stats['skipped_short_trajectory']}")
    print(f"缺少相机: {stats['skipped_missing_camera']}")
    print(f"输出文件: {output_path}")
    
    # 行为分布统计
    print("\n行为分布:")
    behavior_counts = {}
    with jsonlines.open(output_path) as reader:
        for item in reader:
            behavior = item["behavior"]
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
    
    for behavior, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
        pct = count / stats['processed_samples'] * 100
        print(f"  {behavior}: {count} ({pct:.1f}%)")
    
    # 显示样例
    print("\n" + "=" * 50)
    print("样例输出:")
    print("=" * 50)
    with jsonlines.open(output_path) as reader:
        for i, item in enumerate(reader):
            if i >= 1:
                break
            print(json.dumps(item, indent=2, ensure_ascii=False))
    
    return 0


if __name__ == "__main__":
    exit(main())