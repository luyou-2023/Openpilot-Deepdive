import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
import math
from tqdm import tqdm


# ---------- 1. 模型定义 ----------
class PlaningNetwork(nn.Module):
    def __init__(self, M=1, num_pts=10):
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.traj_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, M * num_pts * 3),
        )

    def forward(self, x):
        features = self.backbone(x)
        traj = self.traj_head(features).view(x.size(0), self.M, self.num_pts, 3)
        return None, traj


# ---------- 2. 数据生成 ----------
def generate_synthetic_data(num_samples=100, num_pts=10):
    images = torch.randn(num_samples, 6, 224, 224)
    trajectories = torch.zeros(num_samples, num_pts, 3)
    for i in range(num_samples):
        x = torch.linspace(1.0, 10.0, steps=num_pts)
        y = torch.sin(x / 2.0) + 0.1 * torch.randn(num_pts)
        z = torch.zeros(num_pts)
        trajectories[i] = torch.stack([x, y, z], dim=1)
    return images, trajectories


# ---------- 3. 控制转换 ----------
def trajectory_to_controls(traj):
    controls = []
    for i in range(len(traj) - 1):
        dx = traj[i + 1][0] - traj[i][0]
        dy = traj[i + 1][1] - traj[i][1]

        steer_rad = math.atan2(dy.item(), dx.item())
        steer_deg = np.clip(math.degrees(steer_rad), -450, 450)

        throttle = np.clip(dx.item() / 2.0, 0.0, 1.0)
        brake = 1.0 if dx.item() < 0.1 else 0.0

        controls.append({
            "steer": steer_deg,
            "throttle": throttle,
            "brake": brake
        })
    return controls


# ---------- 4. CAN 编码 ----------
def encode_can_message(control):
    steer = int(control['steer'] * 10)
    throttle = int(control['throttle'] * 255)
    brake = int(control['brake'] * 255)

    can_frame = {
        "id": 0x2E4,
        "data": [
            (steer >> 8) & 0xFF,
            steer & 0xFF,
            throttle,
            brake,
            0x00, 0x00, 0x00, 0x00
        ]
    }
    return can_frame


# ---------- 5. 主流程 ----------
def main():
    # 初始化模型、优化器
    model = PlaningNetwork(M=1, num_pts=10)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # 数据加载
    images, trajectories = generate_synthetic_data()
    dataset = TensorDataset(images, trajectories)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 训练模型
    print("=== Training Model ===")
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for img, traj in tqdm(loader):
            _, pred_traj = model(img)
            pred_traj = pred_traj[:, 0]
            loss = loss_fn(pred_traj, traj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: loss={total_loss / len(loader):.4f}")

    # 模拟预测
    print("\n=== Predicting Trajectory ===")
    model.eval()
    test_img = torch.randn(1, 6, 224, 224)
    with torch.no_grad():
        _, predicted_traj = model(test_img)
        traj = predicted_traj[0, 0]
        print("Predicted trajectory (x,y,z):")
        print(traj)

    # 控制转换
    print("\n=== Converting Trajectory to Controls ===")
    controls = trajectory_to_controls(traj)
    for i, c in enumerate(controls[:3]):
        print(f"Step {i}: {c}")

    # 编码 CAN 消息
    print("\n=== Encoding to CAN Frames ===")
    for i, c in enumerate(controls[:3]):
        frame = encode_can_message(c)
        print(f"Step {i}: {frame}")


if __name__ == "__main__":
    main()


'''
=== Converting Trajectory to Controls ===
将轨迹点转换为车辆控制指令（转向、油门、刹车）
# Step 0: 轻微转向，最大油门，无刹车
{'steer': 4.947883697195622, 'throttle': 1.0, 'brake': 0.0}

# Step 1: 转角加大至约 36.6°，油门减小至约 78%，仍未刹车
{'steer': 36.64371727559848, 'throttle': 0.7793508768081665, 'brake': 0.0}

# Step 2: 急剧转弯（157°，接近 U 型弯），完全松开油门，全力刹车
{'steer': 157.47007515019106, 'throttle': 0.0, 'brake': 1.0}


=== Encoding to CAN Frames ===
将控制指令编码为 CAN 帧格式（8 字节）
# Step 0:
# steer = 0 * 256 + 49 = 49 → 49 / 10 = 4.9 度
# throttle = 255 → 255 / 255 = 1.0（最大油门）
# brake = 0 → 没有刹车
{'id': 740, 'data': [0, 49, 255, 0, 0, 0, 0, 0]}

# Step 1:
# steer = 1 * 256 + 110 = 366 → 366 / 10 = 36.6 度
# throttle = 198 / 255 ≈ 0.78
# brake = 0
{'id': 740, 'data': [1, 110, 198, 0, 0, 0, 0, 0]}

# Step 2:
# steer = 6 * 256 + 38 = 1574 → 1574 / 10 = 157.4 度
# throttle = 0 → 无油门
# brake = 255 → 最大刹车
{'id': 740, 'data': [6, 38, 0, 255, 0, 0, 0, 0]}

背后的数据流程如下：
1. 轨迹点（trajectory）
python
复制
编辑
trajectory = [
    (0.0, 0.0),   # 起点
    (1.0, 1.0),   # 中间点
    (2.0, 0.0)    # 终点（例如是一个 S 型路线）
]
这个轨迹中包含了 3 个点，每个点表示车辆计划要经过的位置。

2. 每个点转换为控制指令（steer, throttle, brake）
程序会为这 3 个点逐一计算：

转向角（steer）：根据相邻两个点的方向变化

油门（throttle）：例如越直越快，越弯越慢

刹车（brake）：急弯可能需要减速或刹车

3. 控制指令编码为 CAN 帧
之后，每组控制指令都被编码成一个长度为 8 的 data 数组，表示可以通过 CAN 总线发送到车辆控制器的原始数据。

✅ 所以结果中有：
Step	原始轨迹点	控制指令	CAN 数据帧
0	(0.0, 0.0)	steer=4.9°, throttle=1.0, brake=0.0	[0, 49, 255, 0, 0, 0, 0, 0]
1	(1.0, 1.0)	steer=36.6°, throttle=0.78, brake=0.0	[1, 110, 198, 0, 0, 0, 0, 0]
2	(2.0, 0.0)	steer=157.4°, throttle=0.0, brake=1.0	[6, 38, 0, 255, 0, 0, 0, 0]

实际有多少 step，取决于你提供了多少轨迹点（trajectory points）。
例如，如果你提供了：

Trajectory 点数量	控制 step 数
3 个点	3 个 step
50 个点	50 个 step
100 个点	100 个 step

⏱️ 一个 step 表示的时间？
一般自动驾驶控制频率是：

20Hz：每秒处理 20 次控制，即 每个 step ≈ 50ms

10Hz：每个 step ≈ 100ms

50Hz：每个 step ≈ 20ms（更细粒度控制）

FSD（Tesla）中每秒多少个 step？
Tesla 的 FSD 中没有完全开源的代码，但根据多个公开文献、论文与 Dojo 相关介绍，FSD 的 trajectory 是预测未来 6 秒，每秒 20Hz（即每秒 20 个点）：

预测时间长度：6 秒

控制频率：20 Hz（50ms 一个控制）

step 数量：6s × 20Hz = 120 steps

这 120 个 trajectory points 是由神经网络输出，用于控制 module 下发控制命令（steering, acceleration, braking）。

✅ openpilot 中的 step 说明
openpilot 是 Comma.ai 的开源自动驾驶系统，它的行为预测（modeld）和控制模块（controlsd）有开源代码。

trajectory 时间范围：最多预测 2.5 秒

step 时间间隔：0.05 秒（即 20Hz）

所以：

arduino
复制
编辑
2.5 秒 ÷ 0.05 秒 = 50 个 step
openpilot 的模型会输出：

python
复制
编辑
model_outputs['plan']['positions']
model_outputs['plan']['velocities']
model_outputs['plan']['acceleration']
通常每个都包含 33 ~ 50 个 step（点），实际数量取决于版本和配置。

✅ 汇总对比
系统	预测时长	频率	step 数
Tesla FSD	~6秒	20Hz	~120
openpilot	2.5秒	20Hz	50
你目前 demo	不确定（例子是 3）	不确定	3
'''
