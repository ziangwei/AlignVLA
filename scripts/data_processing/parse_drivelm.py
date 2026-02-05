#!/usr/bin/env python3
"""
DriveLM数据解析脚本
==================
将DriveLM的原始JSON格式转换为训练所需的结构化格式

输入: data/drivelm/raw/v1_1_train_nus.json
输出: data/drivelm/processed/drivelm_parsed.jsonl

输出格式:
{
    "sample_id": "scene_token__frame_token",
    "scene_token": "...",
    "frame_token": "...",
    "images": {
        "CAM_FRONT": "path/to/image.jpg",
        ...
    },
    "scene_analysis": "综合场景描述",
    "risk_assessment": "风险评估和推理",
    "decision": "决策和理由",
    "behavior": "自车行为描述",
    "source": "drivelm"
}

注意：轨迹需要从nuScenes提取，此脚本只处理DriveLM的文本部分
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re
from tqdm import tqdm
import jsonlines


# 相机列表（固定顺序）
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT", 
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT"
]


def simplify_object_id(obj_id: str) -> str:
    """
    简化对象ID格式
    <c1,CAM_BACK,1088.3,497.5> → [c1@BACK]
    """
    match = re.match(r"<(c\d+),CAM_(\w+),[\d.]+,[\d.]+>", obj_id)
    if match:
        obj_num = match.group(1)
        cam_position = match.group(2)
        return f"[{obj_num}@{cam_position}]"
    return obj_id


def replace_object_ids_in_text(text: str) -> str:
    """
    替换文本中的所有对象ID为简化格式
    """
    pattern = r"<c\d+,CAM_\w+,[\d.]+,[\d.]+>"
    
    def replacer(match):
        return simplify_object_id(match.group(0))
    
    return re.sub(pattern, replacer, text)


def extract_key_objects_description(key_object_infos: Dict) -> str:
    """
    从key_object_infos提取关键对象描述
    """
    if not key_object_infos:
        return ""
    
    descriptions = []
    for obj_id, info in key_object_infos.items():
        simple_id = simplify_object_id(obj_id)
        category = info.get("Category", "Object")
        status = info.get("Status", "")
        visual_desc = info.get("Visual_description", "")
        
        # 提取相机位置
        match = re.match(r"<c\d+,CAM_(\w+),", obj_id)
        if match:
            cam_pos = match.group(1).replace("_", " ").lower()
        else:
            cam_pos = "unknown"
        
        desc = f"{visual_desc} ({category})"
        if status:
            desc += f" is {status.lower()}"
        desc += f" at {cam_pos}"
        
        descriptions.append(desc)
    
    return "; ".join(descriptions)


def extract_scene_analysis(qa_data: Dict, key_object_infos: Dict) -> str:
    """
    从QA数据提取场景分析
    
    主要使用perception类QA中的关键问题：
    - "Please describe the current scene"
    - "What are the important objects..."
    - 位置相关问题
    """
    perception_qa = qa_data.get("perception", [])
    
    parts = []
    
    # 1. 查找场景描述
    for qa in perception_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "describe the current scene" in q.lower():
            parts.append(replace_object_ids_in_text(a))
            break
    
    # 2. 添加关键对象信息
    key_obj_desc = extract_key_objects_description(key_object_infos)
    if key_obj_desc:
        parts.append(f"Key objects: {key_obj_desc}")
    
    # 3. 添加交通信号信息
    for qa in perception_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "traffic element" in q.lower() and "front view" in q.lower():
            if "yes" in a.lower() or "traffic light" in a.lower():
                parts.append(replace_object_ids_in_text(a))
                break
    
    # 4. 添加位置关系（如果场景描述太短）
    if len(parts) < 2:
        for qa in perception_qa:
            q = qa.get("Q", "")
            a = qa.get("A", "")
            
            if "what are objects to the" in q.lower():
                direction = q.split("to the")[-1].strip().rstrip("?")
                parts.append(f"{direction}: {replace_object_ids_in_text(a)}")
    
    return " ".join(parts) if parts else "Scene information not available."


def extract_risk_assessment(qa_data: Dict) -> str:
    """
    从QA数据提取风险评估
    
    主要使用prediction和planning类QA中的关键问题：
    - 对象未来状态预测
    - 碰撞风险评估
    - 需要注意的对象优先级
    """
    prediction_qa = qa_data.get("prediction", [])
    planning_qa = qa_data.get("planning", [])
    
    parts = []
    
    # 1. 对象未来状态
    for qa in prediction_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "what object should the ego vehicle notice" in q.lower():
            parts.append(replace_object_ids_in_text(a))
            break
    
    # 2. 碰撞风险
    collision_risks = []
    for qa in planning_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "probability of colliding" in q.lower():
            if "high" in a.lower():
                collision_risks.append(f"High collision risk: {replace_object_ids_in_text(q)}")
    
    if collision_risks:
        parts.extend(collision_risks)
    
    # 3. 对象优先级
    for qa in planning_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "priority of the objects" in q.lower():
            parts.append(f"Attention priority: {replace_object_ids_in_text(a)}")
            break
    
    # 4. 危险动作
    for qa in planning_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "dangerous actions" in q.lower():
            parts.append(f"Dangerous actions: {a}")
            break
    
    return " ".join(parts) if parts else "No significant risks identified."


def extract_decision(qa_data: Dict) -> str:
    """
    从QA数据提取决策
    
    主要使用planning类QA：
    - 目标动作
    - 安全动作
    - 影响判断的因素
    """
    planning_qa = qa_data.get("planning", [])
    
    parts = []
    
    # 1. 目标动作
    for qa in planning_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "target action of the ego vehicle" in q.lower():
            parts.append(f"Target action: {a}")
            break
    
    # 2. 安全动作
    for qa in planning_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "safe actions" in q.lower():
            parts.append(f"Safe actions: {a}")
            break
    
    # 3. 交通信号相关决策
    for qa in planning_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "traffic signal" in q.lower():
            parts.append(f"Traffic signal: {a}")
            break
    
    # 4. 影响因素
    for qa in planning_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "affect driving judgment" in q.lower():
            parts.append(f"Judgment factors: {a}")
            break
    
    return " ".join(parts) if parts else "Maintain current driving behavior."


def extract_behavior(qa_data: Dict) -> str:
    """
    从QA数据提取自车行为描述
    """
    behavior_qa = qa_data.get("behavior", [])
    
    for qa in behavior_qa:
        q = qa.get("Q", "")
        a = qa.get("A", "")
        
        if "predict the behavior" in q.lower():
            return a
    
    return "The ego vehicle maintains current behavior."


def process_frame(
    scene_token: str,
    frame_token: str,
    frame_data: Dict,
    nuscenes_root: Optional[Path] = None,
) -> Optional[Dict]:
    """
    处理单个帧的数据
    """
    # 提取图像路径
    image_paths = frame_data.get("image_paths", {})
    
    # 验证6个相机都有图像
    images = {}
    for cam in CAMERAS:
        if cam not in image_paths:
            return None  # 缺少相机，跳过
        
        # 处理路径（DriveLM中是相对路径 ../nuscenes/...）
        img_path = image_paths[cam]
        
        # 转换为相对于nuscenes_root的路径
        if img_path.startswith("../nuscenes/"):
            img_path = img_path.replace("../nuscenes/", "")
        
        # 如果提供了nuscenes_root，验证文件存在
        if nuscenes_root:
            full_path = nuscenes_root / img_path
            if not full_path.exists():
                return None  # 图像不存在，跳过
        
        images[cam] = img_path
    
    # 提取QA数据
    qa_data = frame_data.get("QA", {})
    key_object_infos = frame_data.get("key_object_infos", {})
    
    # 构建结构化输出
    result = {
        "sample_id": f"{scene_token}__{frame_token}",
        "scene_token": scene_token,
        "frame_token": frame_token,
        "images": images,
        "scene_analysis": extract_scene_analysis(qa_data, key_object_infos),
        "risk_assessment": extract_risk_assessment(qa_data),
        "decision": extract_decision(qa_data),
        "behavior": extract_behavior(qa_data),
        "source": "drivelm",
    }
    
    return result


def parse_drivelm(
    input_path: Path,
    output_path: Path,
    nuscenes_root: Optional[Path] = None,
) -> Dict[str, int]:
    """
    解析DriveLM JSON文件
    
    Returns:
        统计信息字典
    """
    print(f"Loading DriveLM data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    stats = {
        "total_scenes": 0,
        "total_frames": 0,
        "processed_frames": 0,
        "skipped_frames": 0,
    }
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理每个场景
    results = []
    
    for scene_token, scene_data in tqdm(data.items(), desc="Processing scenes"):
        stats["total_scenes"] += 1
        
        # 跳过scene_description字段（不是frame）
        if not isinstance(scene_data, dict):
            continue
        
        key_frames = scene_data.get("key_frames", {})
        
        for frame_token, frame_data in key_frames.items():
            stats["total_frames"] += 1
            
            result = process_frame(
                scene_token=scene_token,
                frame_token=frame_token,
                frame_data=frame_data,
                nuscenes_root=nuscenes_root,
            )
            
            if result:
                results.append(result)
                stats["processed_frames"] += 1
            else:
                stats["skipped_frames"] += 1
    
    # 保存结果
    print(f"Saving {len(results)} processed frames to {output_path}...")
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(results)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="解析DriveLM数据")
    parser.add_argument(
        "--input",
        type=str,
        default="data/drivelm/raw/v1_1_train_nus.json",
        help="输入DriveLM JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/drivelm/processed/drivelm_parsed.jsonl",
        help="输出JSONL文件路径"
    )
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default="data/nuscenes",
        help="nuScenes数据根目录（用于验证图像存在）"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    nuscenes_root = Path(args.nuscenes_root) if args.nuscenes_root else None
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    stats = parse_drivelm(input_path, output_path, nuscenes_root)
    
    print("\n" + "=" * 50)
    print("DriveLM解析完成")
    print("=" * 50)
    print(f"总场景数: {stats['total_scenes']}")
    print(f"总帧数: {stats['total_frames']}")
    print(f"成功处理: {stats['processed_frames']}")
    print(f"跳过帧数: {stats['skipped_frames']}")
    print(f"输出文件: {output_path}")
    
    # 显示样例
    print("\n" + "=" * 50)
    print("样例输出:")
    print("=" * 50)
    with jsonlines.open(output_path) as reader:
        for i, item in enumerate(reader):
            if i >= 1:
                break
            print(json.dumps(item, indent=2, ensure_ascii=False)[:2000])
            print("...")
    
    return 0


if __name__ == "__main__":
    exit(main())