# dog_rl

四足机器狗 PPO 强化学习项目，sim-to-real 全流程。

## 目标

用 TOE_dog2 MuJoCo 模型 + 自实现 PPO，训练 trot 步态策略，最终部署到实机（12 × 达妙 J8009-2EC + LinkX-4C）。

## 核心方案

**位置控制（方案 A）**：PPO 输出 12 维目标关节角 q_des ∈ [-1,1]
- 仿真：MuJoCo `<position>` actuator + kp=60 / kv=2
- 实机：MIT 模式固定 kp=60 / kd=2，下发策略 q_des
- 仿真和实机 PD 完全一致，sim-to-real 难度最低

详见 [agents.md](./agents.md) 的「核心决策」一节。

## 快速开始

### 环境准备

```powershell
# 复用 deep-learning 项目的 venv（已装 PyTorch）
d:\data\Trae\deep-learning\.venv\Scripts\Activate.ps1

# 验证依赖
python -c "import mujoco, gymnasium, torch; print(mujoco.__version__, gymnasium.__version__, torch.__version__)"
```

### 训练

```powershell
python train.py --config configs/default.yaml
```

### 评估（MuJoCo 可视化）

```powershell
python eval.py --checkpoint checkpoints/best.pt
```

### 导出策略（给 C# 加载）

```powershell
python export_policy.py --checkpoint checkpoints/best.pt --output policy.onnx
```

## 项目状态

规划阶段（2026-09-20）。技术路线、核心决策、待办清单已就位，下一步执行 Step 1（写 `envs/dog_env.py` 最小骨架，验证模型加载 + 站立不塌）。

完整决策记录与进展见 [agents.md](./agents.md)。

## 关联项目

| 项目 | 路径 | 用途 |
|---|---|---|
| robot-dog | `d:\data\Trae\robot-dog\` | 硬件控制（C# / CAN / scan_tool）+ MuJoCo 模型源 TOE_dog2 |
| deep-learning | `d:\data\Trae\deep-learning\` | HF Deep RL Course 学习脚本（14_ppo_mujoco.py 是本项目模板） |
