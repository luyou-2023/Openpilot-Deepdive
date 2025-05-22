import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 参数设置
NUM_SAMPLES = 1000      # 训练样本数量
NUM_STEPS = 50          # 轨迹步数（模拟 2.5 秒 @ 20Hz）
LR = 0.001              # 学习率
EPOCHS = 100            # 训练轮数

# === 生成训练数据 ===
def generate_dataset():
    X, y = [], []
    for _ in range(NUM_SAMPLES):
        angle = random.uniform(-0.2, 0.2)
        step = random.randint(0, NUM_STEPS - 1)
        t = step * 0.05  # 每步间隔 0.05 秒

        x = math.cos(angle) * t * 5.0
        y_ = math.sin(angle) * t * 5.0

        steer = math.degrees(math.atan2(y_, x + 1e-6))
        throttle = max(0.0, 1.0 - (step / NUM_STEPS))
        brake = 1.0 if step > NUM_STEPS * 0.8 else 0.0

        X.append([x, y_])
        y.append([steer, throttle, brake])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# === 定义模型结构（简单 MLP） ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

# === 训练模型 ===
def train(model, X, y):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
    return model

# === 模拟轨迹（未来 2.5 秒）===
def generate_trajectory(start_x=0, start_y=0):
    traj = []
    angle = 0.0
    for i in range(NUM_STEPS):
        angle += random.uniform(-0.01, 0.01)
        x = start_x + 5.0 * math.cos(angle) * i * 0.05
        y = start_y + 5.0 * math.sin(angle) * i * 0.05
        traj.append((x, y))
    return traj

# === 模型预测控制 ===
def predict_controls(model, trajectory):
    model.eval()
    inputs = torch.tensor(trajectory, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs).numpy()
    controls = []
    for o in outputs:
        controls.append({
            'steer': float(o[0]),
            'throttle': float(o[1]),
            'brake': float(o[2])
        })
    return controls

# === 编码为 CAN 帧 ===
def encode_can(controls):
    frames = []
    for i, cmd in enumerate(controls):
        steer = int(cmd['steer']) & 0x1FF
        throttle = int(cmd['throttle'] * 255)
        brake = int(cmd['brake'] * 255)
        data = [i % 256, steer >> 1, ((steer & 1) << 7) | throttle, brake, 0, 0, 0, 0]
        frames.append({'id': 0x2E4, 'data': data})
    return frames

# === 主程序 ===
if __name__ == '__main__':
    print("=== Generating Dataset ===")
    X_train, y_train = generate_dataset()

    print("=== Training PyTorch Model ===")
    model = MLP()
    model = train(model, X_train, y_train)

    print("=== Generating Trajectory ===")
    traj = generate_trajectory()

    print("=== Predicting Controls ===")
    controls = predict_controls(model, traj)

    print("=== Encoding to CAN Frames ===")
    can_frames = encode_can(controls)

    print("\n=== Example Output ===")
    for i in range(3):
        print(f"Step {i}: {traj[i]} -> {controls[i]}")
    for i in range(3):
        print(f"Step {i}: {can_frames[i]}")
