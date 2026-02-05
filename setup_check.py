#!/usr/bin/env python3
"""
环境设置和验证脚本
==================
运行此脚本验证：
1. 目录结构是否正确
2. 依赖是否安装
3. 数据文件是否存在
4. 轨迹Tokenizer是否正常工作
"""

import os
import sys
from pathlib import Path


def check_directory_structure(project_root: Path) -> bool:
    """检查并创建目录结构"""
    print("=" * 60)
    print("检查目录结构")
    print("=" * 60)
    
    required_dirs = [
        "configs",
        "data/drivelm/raw",
        "data/drivelm/processed",
        "data/nuscenes",
        "data/merged",
        "scripts/data_processing",
        "scripts/training",
        "scripts/evaluation",
        "scripts/utils",
        "jobs",
        "logs",
        "models/cache",
        "models/sft",
        "models/dpo",
        "outputs",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            print(f"  [创建] {dir_path}")
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"  [存在] {dir_path}")
    
    print()
    return all_ok


def check_dependencies() -> bool:
    """检查Python依赖"""
    print("=" * 60)
    print("检查Python依赖")
    print("=" * 60)
    
    dependencies = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "accelerate": "Accelerate",
        "deepspeed": "DeepSpeed",
        "peft": "PEFT (LoRA)",
        "trl": "TRL (DPO训练)",
        "datasets": "HuggingFace Datasets",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "omegaconf": "OmegaConf",
        "nuscenes": "nuScenes DevKit",
        "pyquaternion": "PyQuaternion",
        "matplotlib": "Matplotlib",
        "cv2": "OpenCV",
        "jsonlines": "JSONLines",
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  [✓] {name}")
        except ImportError:
            print(f"  [✗] {name} - 未安装")
            missing.append(name)
    
    # 检查flash-attn（可选）
    try:
        import flash_attn
        print(f"  [✓] Flash Attention 2")
    except ImportError:
        print(f"  [!] Flash Attention 2 - 未安装（可选，但推荐）")
    
    print()
    
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True


def check_data_files(project_root: Path) -> bool:
    """检查数据文件"""
    print("=" * 60)
    print("检查数据文件")
    print("=" * 60)
    
    # DriveLM数据
    drivelm_path = project_root / "data/drivelm/raw/v1_1_train_nus.json"
    if drivelm_path.exists():
        size_mb = drivelm_path.stat().st_size / (1024 * 1024)
        print(f"  [✓] DriveLM训练数据: {size_mb:.1f} MB")
    else:
        print(f"  [✗] DriveLM训练数据: 未找到")
        print(f"      期望路径: {drivelm_path}")
    
    # nuScenes数据
    nuscenes_root = project_root / "data/nuscenes"
    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
               "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    samples_dir = nuscenes_root / "samples"
    if samples_dir.exists():
        for cam in cameras:
            cam_dir = samples_dir / cam
            if cam_dir.exists():
                num_files = len(list(cam_dir.glob("*.jpg")))
                print(f"  [✓] {cam}: {num_files} 张图像")
            else:
                print(f"  [✗] {cam}: 目录不存在")
    else:
        print(f"  [✗] nuScenes samples目录不存在: {samples_dir}")
    
    # nuScenes annotations
    annotation_dir = nuscenes_root / "v1.0-trainval"
    if annotation_dir.exists():
        required_files = ["sample.json", "sample_data.json", "ego_pose.json", "scene.json"]
        for f in required_files:
            if (annotation_dir / f).exists():
                print(f"  [✓] {f}")
            else:
                print(f"  [✗] {f}: 未找到")
    else:
        print(f"  [!] nuScenes annotations目录不存在: {annotation_dir}")
    
    print()
    return True


def test_trajectory_tokenizer() -> bool:
    """测试轨迹Tokenizer"""
    print("=" * 60)
    print("测试轨迹Tokenizer")
    print("=" * 60)
    
    try:
        # 添加scripts目录到path
        scripts_dir = Path(__file__).parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        from utils.trajectory_tokenizer import TrajectoryTokenizer, infer_behavior
        
        tokenizer = TrajectoryTokenizer()
        
        # 测试编解码
        test_traj = [(5.0, 0.0), (10.0, 0.5), (15.0, 1.0), (20.0, 1.5), (25.0, 2.0), (30.0, 2.5)]
        encoded = tokenizer.encode_trajectory(test_traj)
        decoded = tokenizer.decode_trajectory(encoded)
        ade = tokenizer.compute_ade(decoded, test_traj)
        
        print(f"  Token数量: {len(tokenizer.all_tokens)} (X: {tokenizer.num_x_tokens}, Y: {tokenizer.num_y_tokens})")
        print(f"  测试轨迹编码: {encoded[:50]}...")
        print(f"  编解码ADE: {ade:.4f}m")
        
        # 测试行为推断
        behaviors = {
            "直行": [(5, 0), (10, 0), (15, 0), (20, 0), (25, 0), (30, 0)],
            "左转": [(5, 0), (10, 2), (12, 5), (12, 10), (10, 15), (8, 20)],
            "右换道": [(5, 0), (10, -0.5), (15, -1.5), (20, -2), (25, -2), (30, -2)],
        }
        
        print(f"  行为推断测试:")
        for name, traj in behaviors.items():
            behavior = infer_behavior(traj, 10.0)
            print(f"    {name} → {behavior}")
        
        print(f"  [✓] 轨迹Tokenizer工作正常")
        print()
        return True
        
    except Exception as e:
        print(f"  [✗] 轨迹Tokenizer测试失败: {e}")
        print()
        return False


def print_summary(project_root: Path):
    """打印项目摘要"""
    print("=" * 60)
    print("项目摘要")
    print("=" * 60)
    print(f"  项目根目录: {project_root}")
    print()
    print("  下一步操作:")
    print("  1. 确保nuScenes 6相机数据下载完成")
    print("  2. 运行 Part 2: DriveLM数据解析")
    print("     sbatch jobs/process_data.sh  # 解注释对应部分")
    print()


def main():
    # 获取项目根目录
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    
    print()
    print("=" * 60)
    print("AlignVLA 环境验证")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print()
    
    # 运行检查
    check_directory_structure(project_root)
    deps_ok = check_dependencies()
    check_data_files(project_root)
    
    if deps_ok:
        test_trajectory_tokenizer()
    
    print_summary(project_root)


if __name__ == "__main__":
    main()