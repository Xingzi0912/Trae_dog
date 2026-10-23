# dog_rl

四足机器狗 PPO 强化学习项目，sim-to-real 全流程。

## 目标

用 TOE_dog2 MuJoCo 模型 + 自实现 PPO，训练 trot 步态策略，最终部署到实机（12 × 达妙 J8009-2EC + LinkX-4C）。

## 核心方案

**位置控制（方案 A）**：PPO 输出 12 维目标关节角 q_des ∈ [-1,1]

- 仿真：MuJoCo `<position>` actuator + kp=150 / kv=2
- 实机：MIT 模式固定 kp=150 / kd=2，下发策略 q_des
- 仿真和实机 PD 完全一致，sim-to-real 难度最低

详见 [agents.md](./agents.md) 的「核心决策」一节。

## 快速开始

### 环境

训练跑在 Linux + NVIDIA GPU 上（MJX），依赖见 `requirements-jax.txt`：

```bash
pip install -r requirements-jax.txt   # mujoco 与 mujoco-mjx 必须同版本
```

CPU 环境只用于验证脚本与离屏渲染。

### 训练（正式入口 `train_jax.py`）

```bash
# 从头训：20M 步 / 2048 并行 env
python -u train_jax.py --steps 20000000 --n_envs 2048

# 续训：从 checkpoint 恢复（σ 起点取当前值，保证单调不升）
python -u train_jax.py --steps 9000000 --n_envs 2048 --resume
```

> `train.py` + `envs/dog_env.py` + `algos/ppo.py` 是旧 PyTorch/CPU 路径，仅保留作
> 一致性基准，不是正式入口；`configs/` 为空占位。

### 验证

```bash
python scripts/test_env_mjx.py            # 环境自检（关节/高度/语义/DR）
python scripts/verify_mjx_consistency.py  # MJX vs CPU MuJoCo 零策略轨迹
python scripts/verify_reward.py           # 奖励分项
python scripts/verify_fk.py               # FK 与姿态估计对拍
```

### 行为验收（离屏渲染）

```bash
python -u scripts/render_jax.py --video --steps 1600 \
  --cmds "0:0,0,0; 200:1.0,0,0; 500:0,0.4,0; 700:-0.4,0,0; 900:0,0,1.5; 1150:0.8,0,1.0; 1400:0,0,0"
```

### 实机

先标定，再站立：

```bash
python3 scripts/calibrate_stance.py stance   # 站姿 -> deploy/stance_calibration.json
python3 scripts/calibrate_stance.py lie      # 趴姿 -> deploy/lie_calibration.json
python3 scripts/stand_up.py
```

标定文件随机器生成，已加入 `.gitignore`；新机器必须先跑标定。

## 项目状态

v24：MJX 2048 env 并行 PPO（教师 99 维）+ 61 维学生蒸馏，速度跟踪 track ≈ 0.62。
完整决策与实验账本见 [agents.md](./agents.md)。

## 关联项目

| 项目          | 路径                          | 用途                                                        |
| ------------- | ----------------------------- | ----------------------------------------------------------- |
| robot-dog     | `d:\data\Trae\robot-dog\`     | 硬件控制（C# / CAN / scan_tool）+ MuJoCo 模型源 TOE_dog2    |
| deep-learning | `d:\data\Trae\deep-learning\` | HF Deep RL Course 学习脚本（14_ppo_mujoco.py 是本项目模板） |
