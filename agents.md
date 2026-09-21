# dog_rl —— 机器狗 PPO 训练项目

## 项目概述

- **目标**：用 TOE_dog2 MuJoCo 模型 + 自实现 PPO，训练 trot 步态策略，最终部署到实机（12 × 达妙 J8009-2EC + LinkX-4C）。
- **起点**：2026-09-20
- **来源项目**：
  - 模型：`d:\data\Trae\robot-dog\TOE_dog2\`（保持原位，绝对路径加载）
  - PPO 模板：`d:\data\Trae\deep-learning\14_ppo_mujoco.py`（在 HalfCheetah-v5 跑通）
  - 硬件规格：`d:\data\Trae\robot-dog\AGENTS.md`
- **Python 环境**：复用 `d:\data\Trae\deep-learning\.venv`（PyTorch 已装，RTX 3060 12GB）

---

## 核心决策（所有代码前必须确认）

### 决策 1：控制方案 = 位置控制（方案 A）⭐ 最重要

PPO 输出 **12 维目标关节角 q_des ∈ [-1,1]**，**不直接发力矩**。

- **仿真端**（MuJoCo XML）：用 `<position>` actuator + kp/kd
  ```xml
  <actuator>
    <position name="FL_hip" joint="FL_hip_joint" kp="60" kv="2"/>
    <!-- 12 个关节同样配置 -->
  </actuator>
  ```
- **实机端**（C# MIT 模式）：固定 `kp=60, kd=2, dq_des=0, tau_ff=0`，下发策略输出 q_des

**理由**：

- J8009-2EC MIT 模式天然带 PD（不是纯力矩电机），方案 A 与硬件能力匹配
- 仿真和实机 PD 完全一致，sim-to-real 难度最低
- 遇到障碍物电机 PD 自然产生阻力，不会失控

**淘汰方案 B（纯力矩）原因**：

- 电机起步易抖飞，撞坏齿轮箱
- 仿真没建模减速比摩擦、温升、backlash，力矩控制对模型误差极敏感
- PPO 必须自己学会"减速时反向输出"，1M 步预算学不动

**升级路径**：方案 A 跑通后再升级方案 C（PPO 输出 q_des + tau_ff，仿真自实现 PD + 前馈），ANYmal/Unitree Go2 走此路。

### 决策 2：训练框架 = 复用 14 号脚本自实现 PPO

- 不用 stable-baselines3 / rsl_rl
- 从 `14_ppo_mujoco.py` 拆分到 `algos/networks.py`（PolicyNetwork/ValueNetwork）+ `algos/ppo.py`（collect_rollout/compute_gae/update_ppo）
- 学习价值最大，sim-to-real 全流程可控

### 决策 3：模型文件路径

- TOE_dog2 保持原位 `d:/data/Trae/robot-dog/TOE_dog2/`，被 `robot-dog/AGENTS.md` 引用
- 通过绝对路径加载 `scene.xml`，不复制模型文件，避免多版本同步问题

### 决策 4：Python 环境

- 复用 `d:/data/Trae/deep-learning/.venv`（PyTorch 已装）
- 如需独立 venv 再新建，目前优先复用

---

## 目录结构（规划）

```
d:\data\Trae\dog_rl\
├── agents.md               # 本文件（决策 + 待办 + 进展）
├── README.md               # 项目概述 + 快速上手
├── envs\                   # 阶段①：Gym 环境封装层
│   ├── __init__.py
│   ├── dog_env.py          # DogEnv 主类（gym.Env 子类）
│   ├── obs_utils.py        # 48 维观测构造 + RunningMeanStd 归一化
│   └── reward.py           # 前进/能耗/抖动/姿态 各分量
├── algos\                  # 阶段②：PPO 算法层（从 14 号脚本拆分）
│   ├── __init__.py
│   ├── ppo.py              # collect_rollout / compute_gae / update_ppo
│   └── networks.py         # PolicyNetwork(48→512→512→12) + ValueNetwork
├── configs\
│   └── default.yaml        # 超参集中管理
├── utils\
│   ├── obs_normalizer.py   # RunningMeanStd
│   └── logger.py           # tensorboard 封装
├── train.py                # 训练入口
├── eval.py                 # 加载 checkpoint 看 MuJoCo 可视化跑
├── export_policy.py        # 导出策略为 ONNX（给 C# 端加载）
├── checkpoints\            # .pt 文件输出
└── logs\                   # tensorboard 日志
```

子目录等真要写代码时再创建，保持初期目录干净。

---

## 技术路线三层

| 层                 | 工程量 | 复用度         | 关键产出                             |
| ------------------ | ------ | -------------- | ------------------------------------ |
| ① Gym 环境封装     | 最大   | 0% 全新        | `DogEnv` 类（obs/12维action/reward） |
| ② PPO 训练         | 小     | 90% 照搬 14 号 | 策略 .pt 文件                        |
| ③ Sim-to-Real 部署 | 中     | 0% 接 C# CAN   | 实机 trot                            |

---

## 关键参数清单

### 观测空间设计（48 维，业界标准）

- 本体状态 10 维：base 高度 z、姿态（四元数→欧拉 或 直接 4 元）、线速度 3、角速度 3
- 关节状态 24 维：12 位置 + 12 速度
- 上一步动作 12 维（助稳定）
- 默认关节偏置 12 维（PPO 攻势项，鼓励回归站立位）

### 动作空间

- `Box(-1, 1, (12,), float32)` —— 12 维 q_des，乘上关节限位后下发

### 奖励函数（稠密 + 多分量）

```
r = +1.0  × forward_x_speed            # 前进速度（主目标）
  - 0.05 × Σ action²                   # 能耗惩罚
  - 0.001 × Σ action_diff²             # 抖动惩罚（与上一步差）
  + 0.5  × upright_bonus(roll,pitch<0.4)  # 别摔倒
```

### 终止条件

- `terminated`：base z < 0.25m（本体塌了）
- `truncated`：1000 步上限

### 域随机化（Reset 时）

- 初始关节角 + 小噪声
- 摩擦系数 0.6~1.5
- 本体质量 ±10%
- 地面随机推动力（轻微扰）

### PPO 超参（相对 14 号脚本的变化）

| 项           | 14 号值        | 改成     | 理由               |
| ------------ | -------------- | -------- | ------------------ |
| ENV_ID       | HalfCheetah-v5 | DogEnv() | 自定义环境         |
| state_dim    | 17             | 48       | 观测维度升级       |
| action_dim   | 6              | 12       | 12 关节            |
| 网络规模     | 256×256        | 512×512  | 观测复杂           |
| TOTAL_STEPS  | 1M             | 20M~50M  | 步态学习慢         |
| ENTROPY_BETA | 0.0            | 0.01     | 局部最优多，加探索 |

---

## 待办清单（按执行顺序）

### Step 1：让模型站起来 ⭐ 拦路虎，先验证

- [x] 创建 `dog_rl` 项目骨架（envs/algos/utils/configs 子目录 + 空 `__init__.py`）
- [x] 写 `envs/dog_env.py` 最小骨架：加载 `scene.xml` + reset 到站立位
- [x] 跑零策略 1000 步，确认机器狗不塌
- [x] 如果塌：调整 `dog.xml` 初始 qpos 或在 reset 里设 standing pose（未塌，keyframe home 一次到位）
- [x] MuJoCo viewer 可视化确认姿态正确（scripts/view_stand.py）

### Step 2：手动验证 reward 设计

- [x] 给固定 q_des 让狗往前挪，验证 forward reward 递增（手动摆腿实际产生后退 vx=-0.16，reward 正确给负，方向语义验证通过）
- [x] 验证能耗/抖动惩罚在合理范围（energy 正常；jerk 在 dt=0.002 下偏弱，Step 4/5 再调）
- [x] 验证摔倒时 terminated 触发（外力矩 30 N·m，step 258 触发，upright 同步归零）

### Step 3：接入 PPO 训练

- [ ] 从 `14_ppo_mujoco.py` 拆出 `algos/networks.py` + `algos/ppo.py`
- [ ] 改 state_dim=48, action_dim=12
- [ ] 网络规模 256→512
- [ ] 先 1M 步看曲线对不对（reward 上行、loss 收敛）

### Step 4：obs 归一化 + 域随机化

- [ ] 实现 `RunningMeanStd` 在线归一化（IMU 量级 ~10 vs 关节角 ~1，必须归一化）
- [ ] 实现域随机化（摩擦/质量/初始姿态）

### Step 5：长训

- [ ] 20M~50M 步训练
- [ ] tensorboard 监控 reward / value loss / entropy / KL
- [ ] 检查点保存（best + last）

### Step 6：sim-to-real 准备导出

- [ ] `export_policy.py` 导出策略为 ONNX（仅 mu_head，σ 推理不需要）
- [ ] 在 `robot-dog/sdk/` 加 ONNX Runtime C# 加载代码
- [ ] 关节顺序映射确认：URDF/MJCF 关节顺序 → (CH0~3, MotorID 0x01~03)
- [ ] 50Hz 控制周期（PPO dt=0.02s）
- [ ] 读反馈帧 → 拼成 48 维 obs → 推理得 12 维 action → MIT 力矩下发

---

## 关联资源

| 用途                 | 路径                                                    |
| -------------------- | ------------------------------------------------------- |
| MuJoCo 模型源        | `d:/data/Trae/robot-dog/TOE_dog2/xml/scene.xml`         |
| 狗本体定义           | `d:/data/Trae/robot-dog/TOE_dog2/xml/dog.xml`           |
| URDF（备用）         | `d:/data/Trae/robot-dog/TOE_dog2/urdf/dog.urdf`         |
| PPO 训练模板         | `d:/data/Trae/deep-learning/14_ppo_mujoco.py`           |
| HalfCheetah 探索脚本 | `d:/data/Trae/deep-learning/13_explore_halfcheetah.py`  |
| 硬件规格与电机映射   | `d:/data/Trae/robot-dog/AGENTS.md`                      |
| Python venv          | `d:/data/Trae/deep-learning/.venv`                      |
| C# 扫描工具          | `d:/data/Trae/robot-dog/scan_tool/Program.cs`           |
| C# 电机控制 demo     | `d:/data/Trae/robot-dog/sdk/.../CSharp/demo/Program.cs` |

## 实机关节顺序映射（待确认）

```
策略动作 12 维顺序 ↔ (CAN通道, MotorID)
建议约定 URDF/MJCF 关节顺序：
  [FL_hip, FL_thigh, FL_calf,
   FR_hip, FR_thigh, FR_calf,
   RL_hip, RL_thigh, RL_calf,
   RR_hip, RR_thigh, RR_calf]

实机映射（来自 robot-dog/AGENTS.md）：
  CH0=RL, CH1=RR, CH2=FL(?), CH3=FR(?)
  MotorID: 髋=0x01, 大腿=0x02, 小腿=0x03
```

---

## 进展记录

### 2026-09-20 — 项目创建

- 创建 `d:\data\Trae\dog_rl\` 目录
- 写 `agents.md` 记录核心决策（位置控制方案 A、复用 14 号脚本、模型原位加载、venv 复用）
- 写 `README.md` 项目概述
- 技术路线与待办清单已就位，准备进入新对话执行 Step 1

**下一步（在新对话执行）**：

1. 创建子目录骨架（envs/algos/utils/configs）
2. 写 `envs/dog_env.py` 最小骨架
3. 跑零策略验证模型加载 + 站立不塌

### 2026-09-20 — Step 1 完成

- 写 `envs/dog_positional.xml`：基于 TOE_dog2/xml/dog.xml 改造，actuator 从 `<motor>` 改为 `<position kp=60 kv=2>`，meshdir 改绝对路径，启用 keyframe home
- 写 `envs/scene_positional.xml`：包含自定义 dog_positional.xml
- 写 `envs/dog_env.py` 最小骨架（DogEnv 类）：加载 scene + keyframe home reset + 零策略 step + base z 终止条件
- 写 `scripts/view_stand.py`：MuJoCo 交互式 viewer 看站立姿态
- 跑零策略 1000 步验证通过 ✅，狗没塌（commit `1a817d6`）

**Step 1 待办全部勾选，下一步 = Step 2**

### 2026-09-21 — Step 2 reward 函数实现

- 在 `envs/dog_env.py` 加 `_compute_reward(action)` 方法（4 分量）：
  - `+1.0 × forward_x_speed`（base 线速度 x 分量，世界系）
  - `-0.05 × Σ action²`（能耗惩罚）
  - `-0.001 × Σ (action - last_action)²`（抖动惩罚）
  - `+0.5 × upright_bonus`（roll, pitch < 0.4 时给 +0.5，二值）
- 加 `_quat_to_rpy` 静态方法（四元数 → roll/pitch/yaw）
- `step()` 调用 `_compute_reward` 替代 `reward=0.0`，`reset()` 初始化 `_last_action`
- 写 `scripts/verify_reward.py` 3 场景验证脚本：
  1. 零策略：验证 reward ≈ +0.5/step（只有 upright）
  2. 周期摆腿（5Hz 对角 trot 摆腿）：验证 forward/energy/jerk 都触发且符号正确
  3. 持续外力矩 30 N·m 绕 x 轴：验证 terminated 触发 + upright 归零

**Step 2 待办清单待跑完 verify_reward.py 后再勾**

**下一步**：跑 `scripts/verify_reward.py`，根据 ✅/⚠️/❌ 标记判断 reward 设计是否合理

### 2026-09-21 — Step 2 验证通过

跑 `scripts/verify_reward.py` 3 场景结果：

**场景 1（零策略 500 步）**

- upright mean = +0.5000 ✅，energy/jerk = 0 ✅，狗没塌（base_z ∈ [0.270, 0.289]）✅
- ⚠️ forward total = **-25.9**（mean -0.052/step）：不是 bug，是 reset 启动瞬态。
  keyframe home z=0.27 与重力下静平衡位置有微小偏差，PD 启动瞬间狗腿压缩/后坐，
  前 ~100 步（0.2s）产生负 vx，step 300 后稳态 reward ≈ +0.503（forward≈0）。
  PPO 学相对优势，每 episode 固定偏置不影响策略梯度；如后续影响训练可在 reset 后加热身步。

**场景 2（5Hz 对角 trot 摆腿 500 步）**

- forward total = -80.1（vx mean = **-0.16 m/s，狗在后退**）：手动设计的 sin 相位
  方向反了，但 reward **正确给出负值** → forward reward 的方向语义验证通过。
  PPO 会自己学正确相位，不需要手调摆腿脚本。
- energy total = -10.88 ✅ 符号/量级正确
- ⚠️ jerk total = -0.0009 几乎为 0：dt=0.002s 下相邻 step action 差 ≈ 0.025，
  权重 0.001 太弱。**Step 4/5 调参时考虑提到 0.01~0.1 或按 action_rate/step 归一化**。

**场景 3（持续 30 N·m 绕 x 外力矩）**

- roll 单调上升 0.001 → 1.553，step 120 roll=0.4 时 upright 归零，
  step 258 base_z=0.2491 触发 terminated ✅
- terminated 与 upright_bonus 联动正确

**Step 2 结论：reward 4 分量方向语义全部正确，无需改代码。遗留两个调参项记到 Step 4/5：**

1. jerk 权重 0.001 在 dt=0.002 下偏弱
2. reset 启动瞬态 forward 偏置（先观察 PPO 是否敏感，不急着修）

**当前未 commit 的改动**：`envs/dog_env.py`（reward）、`scripts/verify_reward.py`（新）、`agents.md`

**下次（Step 3）**：从 `d:/data/Trae/deep-learning/14_ppo_mujoco.py` 拆出
`algos/networks.py`（PolicyNetwork 48→512→512→12 + ValueNetwork）和
`algos/ppo.py`（collect_rollout / compute_gae / update_ppo），
同时把 DogEnv obs 从 19 维 qpos 占位升级到 48 维规范观测。
