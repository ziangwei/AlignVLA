"""
轨迹Tokenizer - 核心工具
========================
负责轨迹的离散化和token化

离散化方案：
- X方向：0-30米，精度0.5米 → 61个token (X0-X60)
- Y方向：-6到+6米，精度0.5米 → 25个token (Y0-Y24)
  - Y0 = -6米, Y12 = 0米, Y24 = +6米

Token格式：
- 单个waypoint: "<X30><Y12>" 表示 (15m, 0m)
- 完整轨迹: "<X10><Y12> <X20><Y12> <X30><Y12> <X40><Y12> <X50><Y12> <X60><Y12>"
"""

from typing import List, Tuple, Optional
import numpy as np


class TrajectoryTokenizer:
    """轨迹离散化和token化工具"""
    
    def __init__(
        self,
        x_range: Tuple[float, float] = (0, 50),
        y_range: Tuple[float, float] = (-10, 10),
        resolution: float = 0.5,
    ):
        """
        初始化轨迹tokenizer
        
        Args:
            x_range: X轴范围 (min, max)，单位米。默认0-50m覆盖高速场景
            y_range: Y轴范围 (min, max)，单位米。默认±10m覆盖多车道
            resolution: 离散化精度，单位米
        """
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.resolution = resolution
        
        # 计算token数量
        self.num_x_tokens = int((self.x_max - self.x_min) / resolution) + 1  # 61
        self.num_y_tokens = int((self.y_max - self.y_min) / resolution) + 1  # 25
        
        # 生成token列表
        self.x_tokens = [f"<X{i}>" for i in range(self.num_x_tokens)]
        self.y_tokens = [f"<Y{i}>" for i in range(self.num_y_tokens)]
        self.all_tokens = self.x_tokens + self.y_tokens
        
        # Y轴的零点索引（Y12 = 0米）
        self.y_zero_index = int((0 - self.y_min) / resolution)
        
    def get_special_tokens(self) -> List[str]:
        """返回需要添加到词表的特殊token列表"""
        return self.all_tokens
    
    def continuous_to_discrete(self, x: float, y: float) -> Tuple[int, int]:
        """
        将连续坐标转换为离散索引
        
        Args:
            x: X坐标（米）
            y: Y坐标（米）
            
        Returns:
            (x_index, y_index) 离散索引
        """
        # 裁剪到范围内
        x = np.clip(x, self.x_min, self.x_max)
        y = np.clip(y, self.y_min, self.y_max)
        
        # 离散化
        x_index = int(round((x - self.x_min) / self.resolution))
        y_index = int(round((y - self.y_min) / self.resolution))
        
        # 确保在范围内
        x_index = np.clip(x_index, 0, self.num_x_tokens - 1)
        y_index = np.clip(y_index, 0, self.num_y_tokens - 1)
        
        return int(x_index), int(y_index)
    
    def discrete_to_continuous(self, x_index: int, y_index: int) -> Tuple[float, float]:
        """
        将离散索引转换为连续坐标
        
        Args:
            x_index: X离散索引
            y_index: Y离散索引
            
        Returns:
            (x, y) 连续坐标（米）
        """
        x = self.x_min + x_index * self.resolution
        y = self.y_min + y_index * self.resolution
        return x, y
    
    def encode_waypoint(self, x: float, y: float) -> str:
        """
        将单个waypoint编码为token字符串
        
        Args:
            x: X坐标（米）
            y: Y坐标（米）
            
        Returns:
            token字符串，如 "<X30><Y12>"
        """
        x_idx, y_idx = self.continuous_to_discrete(x, y)
        return f"<X{x_idx}><Y{y_idx}>"
    
    def decode_waypoint(self, token_str: str) -> Tuple[float, float]:
        """
        将token字符串解码为坐标
        
        Args:
            token_str: token字符串，如 "<X30><Y12>"
            
        Returns:
            (x, y) 连续坐标（米）
        """
        import re
        match = re.match(r"<X(\d+)><Y(\d+)>", token_str)
        if not match:
            raise ValueError(f"Invalid waypoint token: {token_str}")
        
        x_idx = int(match.group(1))
        y_idx = int(match.group(2))
        return self.discrete_to_continuous(x_idx, y_idx)
    
    def encode_trajectory(self, trajectory: List[Tuple[float, float]]) -> str:
        """
        将完整轨迹编码为token字符串
        
        Args:
            trajectory: waypoint列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            token字符串，如 "<X10><Y12> <X20><Y12> <X30><Y12>"
        """
        waypoint_tokens = [self.encode_waypoint(x, y) for x, y in trajectory]
        return " ".join(waypoint_tokens)
    
    def decode_trajectory(self, token_str: str) -> List[Tuple[float, float]]:
        """
        将token字符串解码为轨迹
        
        Args:
            token_str: 完整轨迹token字符串
            
        Returns:
            waypoint列表 [(x1, y1), (x2, y2), ...]
        """
        import re
        pattern = r"<X\d+><Y\d+>"
        waypoint_tokens = re.findall(pattern, token_str)
        return [self.decode_waypoint(wp) for wp in waypoint_tokens]
    
    def compute_ade(
        self, 
        pred_trajectory: List[Tuple[float, float]], 
        gt_trajectory: List[Tuple[float, float]]
    ) -> float:
        """
        计算Average Displacement Error
        
        Args:
            pred_trajectory: 预测轨迹
            gt_trajectory: 真值轨迹
            
        Returns:
            ADE（米）
        """
        if len(pred_trajectory) != len(gt_trajectory):
            # 取最小长度
            min_len = min(len(pred_trajectory), len(gt_trajectory))
            pred_trajectory = pred_trajectory[:min_len]
            gt_trajectory = gt_trajectory[:min_len]
        
        if len(pred_trajectory) == 0:
            return float('inf')
        
        errors = []
        for (px, py), (gx, gy) in zip(pred_trajectory, gt_trajectory):
            error = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            errors.append(error)
        
        return float(np.mean(errors))
    
    def compute_fde(
        self, 
        pred_trajectory: List[Tuple[float, float]], 
        gt_trajectory: List[Tuple[float, float]]
    ) -> float:
        """
        计算Final Displacement Error
        
        Args:
            pred_trajectory: 预测轨迹
            gt_trajectory: 真值轨迹
            
        Returns:
            FDE（米）
        """
        if len(pred_trajectory) == 0 or len(gt_trajectory) == 0:
            return float('inf')
        
        px, py = pred_trajectory[-1]
        gx, gy = gt_trajectory[-1]
        return float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
    
    def get_discretization_error(self) -> float:
        """
        返回离散化的最大误差（对角线距离的一半）
        """
        return np.sqrt(2) * self.resolution / 2


def infer_behavior(
    trajectory: List[Tuple[float, float]],
    ego_speed: float,
    heading_threshold: float = 20.0,
    lane_change_threshold: float = 1.5,
) -> str:
    """
    从轨迹推断驾驶行为/指令
    
    Args:
        trajectory: waypoint列表 [(x, y), ...]
        ego_speed: 当前车速（m/s）
        heading_threshold: 航向角变化阈值（度）
        lane_change_threshold: 换道Y偏移阈值（米）
        
    Returns:
        行为字符串: "Go straight", "Turn left", "Turn right", 
                   "Change lane left", "Change lane right", "Slow down"
    """
    if len(trajectory) < 2:
        return "Go straight"
    
    # 计算航向角变化
    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    
    # 从起点到终点的方向
    dx = end_x - start_x
    dy = end_y - start_y
    
    if dx < 0.1:  # 几乎没有前进
        if ego_speed < 0.5:
            return "Slow down"
        return "Go straight"
    
    # 计算航向角（相对于正前方）
    heading_angle = np.degrees(np.arctan2(dy, dx))
    
    # 最终Y偏移
    final_y = end_y
    
    # 判断行为
    if abs(heading_angle) > heading_threshold:
        if heading_angle > 0:
            return "Turn left"
        else:
            return "Turn right"
    elif abs(final_y) > lane_change_threshold:
        if final_y > 0:
            return "Change lane left"
        else:
            return "Change lane right"
    else:
        # 检查是否减速（轨迹点间距逐渐减小）
        if len(trajectory) >= 3:
            first_dist = np.sqrt(
                (trajectory[1][0] - trajectory[0][0]) ** 2 + 
                (trajectory[1][1] - trajectory[0][1]) ** 2
            )
            last_dist = np.sqrt(
                (trajectory[-1][0] - trajectory[-2][0]) ** 2 + 
                (trajectory[-1][1] - trajectory[-2][1]) ** 2
            )
            if last_dist < first_dist * 0.5:  # 最后的间距小于一半
                return "Slow down"
        
        return "Go straight"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    # 创建tokenizer
    tokenizer = TrajectoryTokenizer()
    
    print("=== 轨迹Tokenizer测试 ===")
    print(f"X tokens: {tokenizer.num_x_tokens} (0-50m, 0.5m精度)")
    print(f"Y tokens: {tokenizer.num_y_tokens} (-10到+10m, 0.5m精度)")
    print(f"总token数: {len(tokenizer.all_tokens)}")
    print(f"最大离散化误差: {tokenizer.get_discretization_error():.3f}m")
    print()
    
    # 测试编码解码（扩展范围的轨迹）
    test_trajectory = [
        (8.0, 0.0),
        (16.0, 0.5),
        (24.0, 1.0),
        (32.0, 1.5),
        (40.0, 2.0),
        (48.0, 2.5),
    ]
    
    print("测试轨迹:", test_trajectory)
    encoded = tokenizer.encode_trajectory(test_trajectory)
    print("编码结果:", encoded)
    decoded = tokenizer.decode_trajectory(encoded)
    print("解码结果:", decoded)
    print()
    
    # 测试ADE
    ade = tokenizer.compute_ade(decoded, test_trajectory)
    print(f"编解码ADE: {ade:.3f}m (应该 < {tokenizer.get_discretization_error():.3f}m)")
    print()
    
    # 测试行为推断
    print("=== 行为推断测试 ===")
    
    # 直行（50m范围内）
    straight_traj = [(8, 0), (16, 0), (24, 0), (32, 0), (40, 0), (48, 0)]
    print(f"直行轨迹: {infer_behavior(straight_traj, 10.0)}")
    
    # 左转（大角度）
    left_turn = [(8, 0), (15, 3), (18, 8), (18, 15), (15, 22), (10, 28)]
    print(f"左转轨迹: {infer_behavior(left_turn, 10.0)}")
    
    # 右换道（Y偏移但航向角小）
    right_lane = [(8, 0), (16, -0.5), (24, -1.5), (32, -2.5), (40, -3), (48, -3.5)]
    print(f"右换道轨迹: {infer_behavior(right_lane, 10.0)}")
    
    print()
    print("=== 特殊token列表（前10个）===")
    for token in tokenizer.get_special_tokens()[:10]:
        print(f"  {token}")
    print("  ...")