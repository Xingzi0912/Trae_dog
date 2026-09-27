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
- ~~通过绝对路径加载 `scene.xml`，不复制模型文件，避免多版本同步问题~~
- **2026-09-22 变更（服务器部署）**：dog_rl 仓库自包含——STL 复制到
  `envs/assets/`（18.3MB），dog_positional.xml 的 meshdir 改相对路径 `"assets"`
  （相对该 XML 所在 envs/ 目录，include 场景已实测）。克隆即跑，无需在
  服务器上重建 d:/data/... 目录结构。TOE_dog2 原位资产仍保留不动。

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

### 观测空间设计（58 维，2026-09-22 决策确认）

> 原写"48 维业界标准"，但下列分量加总实为 58；经确认按 **58 维全量版**实现
> （关节位置用绝对角 + 保留常量偏置）。所有速度投影到 **base 机身系**。

- 本体状态 10 维：base 高度 z、roll/pitch/yaw 3、机身系线速度 3、机身系角速度 3
- 关节状态 24 维：12 位置（绝对角）+ 12 速度
- 上一步动作 12 维（助稳定）
- 默认关节偏置 12 维（常量，PPO 攻势项，鼓励回归站立位）

obs 索引：[0]=z，[1:4]=rpy，[4:7]=lin_vel，[7:10]=ang_vel，
[10:22]=q，[22:34]=dq，[34:46]=last_action，[46:58]=q_default。

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
| state_dim    | 17             | 58       | 观测维度升级       |
| action_dim   | 6              | 12       | 12 关节            |
| 网络规模     | 256×256        | 512×512  | 观测复杂           |
| TOTAL_STEPS  | 1M             | 20M~50M  | 步态学习慢         |
| ENTROPY_BETA | 0.0            | 0.01     | 局部最优多，加探索 |

---

## 决策 5：指令条件策略（手柄遥控 vx/vy/yaw）+ 61 维新观测（2026-09-26 冻结）

**背景**：58 维模型（9/22 checkpoint）只会自主向前，无控制语义，不作为部署候选。
新模型接收速度指令，训练后实机用手柄遥控前进/横移/转向/停车。**旧 58 维
checkpoint 不兼容，从头训，不覆盖旧文件**（新模型命名 `dog_cmd_mjx_*`）。

### 5.1 观测布局：58 → 61 维（含 projected gravity 改造）

姿态观测 **obs[1:4] 从 rpy 改为 projected gravity**（重力向量在机身系投影）：
BMI088 为 6 轴 IMU，EKF 无磁力计修正，yaw 纯陀螺积分必漂移（静止 0.013rad/s
→ 20s 漂 0.1~0.3rad）。projected gravity 数学上不含 yaw，漂移天然免疫；
转向跟踪用 gyro ωz（obs[9]，无漂移）+ yaw_rate 指令即可。FK 估计器高度公式
只用 R 第三行、速度全在机身系，不依赖 yaw，改动无副作用。

| 索引    | 量（全部机身系，除 base_z）                | 仿真来源                | 实机来源                                 |
| ------- | ------------------------------------------ | ----------------------- | ---------------------------------------- |
| [0]     | base_z（世界系高度）                       | xpos[base].z            | FK 估计器（支撑脚约束，已验证 0.3mm）    |
| [1:4]   | **projected gravity g_b**（站立≈[0,0,-1]） | Rᵀ·[0,0,-1]             | IMU 四元数算 Rᵀ·[0,0,-1]（yaw 丢弃）     |
| [4:7]   | 机身系线速度 vx/vy/vz                      | Rᵀ·qvel[0:3]            | FK 估计器（动态误差 2.3cm/s）            |
| [7:10]  | 机身系角速度                               | Rᵀ·qvel[3:6]            | IMU gyro 直出（rad/s）                   |
| [10:22] | 12 关节角（**sim 系**）                    | qpos[7:]                | pos16 反馈 → **real→sim 逆映射**         |
| [22:34] | 12 关节速度（**sim 系**）                  | qvel[6:]                | vel12 → S×dq_real + 一阶轻低通(~20Hz)    |
| [34:46] | last_action                                | 缓存上帧输出            | 同（自身已知量）                         |
| [46:58] | q_default 常量                             | **固定** [0,0.9,-1.8]×4 | **必须填同一仿真常量**，禁止填实机标定值 |
| [58:61] | **指令 [vx_cmd, vy_cmd, yaw_rate_cmd]**    | 训练随机采样            | 手柄（单位 m/s, m/s, rad/s）             |

实机关节反馈逆映射（run_policy.py 必须实现，记忆此前只有正向动作映射）：

```
q_sim[i]  = q_default_sim[i] + S[i] × (q_real[i] − q_default_real_eff[i])
dq_sim[i] = S[i] × dq_real[i]
```

`q_default_real_eff` = stance_calibration 零偏折叠 posture_trim（仅小腿）后的
有效零偏。数据流顺序：**反馈 → 逆映射 → FK + 拼 obs**，不能反。
vel12 量化分辨率 ~0.022rad/s，远小于 DR 噪声 0.15，量级安全。

### 5.2 指令采样与三段课程（仿真训练）

最终分布：vx∈[-0.5,+1.5]、vy∈[-0.5,+0.5]、yaw_rate∈[-2.0,+2.0]（单位
m/s、rad/s）；每次采样 **15~20% 概率给零指令**（显式学习停车站立）；
episode 内每 **2~4s 重采样**一次（学习中途切换方向，用 scan carry 实现）。

| 阶段 | vx          | vy          | yaw_rate    |
| ---- | ----------- | ----------- | ----------- |
| 初期 | [0, 0.8]    | 0           | [-0.5, 0.5] |
| 中期 | [-0.3, 1.2] | [-0.3, 0.3] | [-1.5, 1.5] |
| 末期 | [-0.5, 1.5] | [-0.5, 0.5] | [-2.0, 2.0] |

切换步数等冒烟 + 1~2M 步曲线后标定。

### 5.3 奖励（替换旧 forward 分量；速度跟踪为主导信号）

```python
r = 1.0 * exp(-(vx_err² + vy_err²) / 0.25)   # 线速度跟踪（机身系）
  + 0.5 * exp(-(yaw_err²) / 0.25)            # 转向跟踪
  + 0.5 * upright                            # |g_x|,|g_y| < sin(0.4)≈0.39 时=1（与旧阈值等价）
  - 0.05 * Σ action²                          # 能耗
  - 0.01 * Σ (action-last_action)²            # 抖动（旧债 0.001→0.01）
  - 1.0  * vz_body²                           # 抑制上下颠
  - 0.05 * (ωx² + ωy²)                        # 抑制 roll/pitch 抖动（小权重，防"不动最稳"）
```

教训（经验库）：稳定/平滑项权重不得喧宾夺主，否则策略收敛静止退化解；
v1 **不加** foot airtime / 步态相位奖励，不硬编码 trot。终止条件不变
（z<0.25，1000 步截断）。DR 中 [1:4] 噪声从 rpy 0.01rad 改为重力向量
分量噪声 0.02~0.05；去掉 yaw 漂移项。

### 5.4 评估（脚本化指令序列，替代只看 episode return）

站立2s → 直行 vx=0.8（4s）→ 走转 vx=0.5/yaw=0.8（4s）→ 刹车停2s →
横移/后退段；10 种子。指标：各段速度 RMSE、姿态角、是否摔倒。
初定阈值：直行 vx RMSE<0.2、yaw RMSE<0.3、松指令 2s 内停稳；训完校准。
成功基线：评估均值 ≥1500 且 10 种子最低 ≥800 + 视觉步态自然（沿用旧标准）。

---

## 决策 6：GPU 训练走 MJX（JAX），不用 Isaac/PhysX（2026-09-26 冻结）

- **瓶颈是 MuJoCo 物理（mj_step 仅 CPU），不是网络**（actor 仅 30 万参数）。
  单改 `.to(cuda)` 无提速；CPU 20env ≈140 step/s，20M 步 ~40h。
- 选 **MuJoCo MJX（JAX）**：加载同一 MJCF，位置 PD/摩擦/DR 调参成果同源保留，
  sim-to-real 一致性远好于换 PhysX（Isaac Lab/Genesis 否，等于换项目）。
- 目标 2048 env 起（4090，按吞吐/显存调 4096），预期 3万~8万 step/s，
  20M 步分钟级。**训练机 = 4090 Linux 服务器**（pip jax[cuda12]+mujoco-mjx+flax）。
- **PPO 用 Flax 自己写**（沿用决策2"自实现可控"）：GAE 按 env 分离、
  done mask/bootstrap、KL 早停 0.02~0.03、lr 从 anneal_steps 独立衰减、
  熵系数 0.003、obs_rms 归一化；rollout/minibatch 尺寸按大批量重标。
- checkpoint 双存：Flax 原生（续训）+ **部署包 .npz**（MLP 权重 + obs_rms，
  NUC 端 numpy 矩阵乘推理，无需 JAX）。命名 `checkpoints/dog_cmd_mjx_best.pt/.npz`。

### MJX 移植要点（2026-09-26 阶段 1 已实测，mujoco-mjx 3.13.0 / jax 0.11.2）

- 新建 `envs/dog_mjx.xml` + `envs/scene_mjx.xml`（原文件不动）。
  本地 venv（Python3.13）装 CPU 版 jax 0.11.2 + **mujoco-mjx==3.13.0**
  （版本必须与 mujoco 精确一致；阿里云镜像有，清华镜像滞后停在 3.2.2）。
  注意 TRAE 沙箱默认禁止写 deep-learning venv，pip 需沙箱外执行。
- **实测的最终适配清单**（探针矩阵 scripts/probe*mjx*\*.py 实证，非猜测）：
  1. **椭圆锥保留** `cone="elliptic" impratio="100"`——3.13 原生支持，零 warning；
     改金字塔锥会导致 RR_hip 站立偏 4.7°、base_z 低 1cm（impratio=100 的近零
     侧向摩擦是点足设定，绝不能丢）
  2. 积分器加 `integrator="implicitfast"`（与 Euler 站立轨迹实测等价，批量
     rollout 更稳；CPU 同 XML 同积分器对比）
  3. **cylinder→capsule 是唯一硬性改动**：`CYLINDER-BOX collisions not
implemented`（站立即触发自碰撞）；size/pos/quat 原样保留
  4. **condim=6 保留**（3.13 实测支持；比 condim=3 更贴近原模型站立高度）
  5. 删除 `<sensor>` 段（训练不用，obs 直接从 data 构造）
  - frictionloss=0.2 / margin=0.001 / actuatorfrcrange 全部原生支持，零 warning
- **一致性校验通过**（scripts/verify_mjx_consistency.py，零策略 2s）：
  引擎差（CPU-MJX vs MJX 同 XML）关节 max 0.187°/base 0.38mm；
  适配差（原 XML vs MJX-XML，均 CPU）关节 max 0.36°/base 0.21mm；
  终态 base_z 三者 0.27247/0.27249/0.27250 几乎重合。图 logs/mjx_consistency.png。
  ⚠️ 仅验证站立；阶段 2 动态步行后需再跑一次动态一致性（capsule 自碰撞
  包膜在摆腿时差异可能更大）。
- 新文件 `envs/dog_env_mjx.py`、`algos/ppo_jax.py`、`algos/networks_jax.py`，
  全部 vmap/scan/jit 化；旧 CPU dog_env.py 保留两个用途：
  ①MJX 一致性基准 ②训练后加载 npz 权重渲染视频（MJX 离屏渲染麻烦）。
- 零策略语义/时间尺度不变：q_des=q_default+action×0.25、frame_skip=10、
  dt=0.002、50Hz；DR（摩擦/质量/kp/kv/初态扰动/动作延迟/观测噪声）全向量化。

### 实施阶段

0. 4090 服务器装 jax[cuda12]/mujoco-mjx/flax，代码 tar 同步
1. dog_mjx.xml 适配 + mjx.test_model() + **MJX/CPU 动力学一致性校验**
2. `envs/dog_env_mjx.py`：2048 env 向量化 + 61 维观测 + DR + 指令课程
3. `algos/ppo_jax.py`/`networks_jax.py`：Flax PPO + 双格式 checkpoint
4. 指令奖励训练（目标 20M 步）+ 脚本化评估
5. CPU MuJoCo 加载 npz 渲染验证步态/指令跟踪
6. 实机 `run_policy.py` + 手柄（训练达标后再写）

### 实机部署（阶段6）手柄与安全

- pygame 读手柄（Xbox/北通类优先）：左摇杆上下=vx、左右=vy；右摇杆左右=yaw；
  死区 0.1；缩放 1.5/0.5/2.0；指令斜率限制防猛打杆。
- 安全：使能后默认零指令站稳；**肩键 deadman（按住才放行非零指令，松开停车）**；
  独立按键急停失能；起立仍走 stand_up.py，站稳再切 run_policy.py。
- ⚠️ **contact[4] 支撑脚掩码待实现**（实机无力传感器）：优先髋部高度几何法
  （FK 各脚相对 base 的 z，最低两只判支撑），电流阈值备选；trot 摆动相若
  全脚判支撑会拉偏 FK 高度/速度，不可省。

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

- [x] 从 `14_ppo_mujoco.py` 拆出 `algos/networks.py` + `algos/ppo.py`
- [x] 改 state_dim=58, action_dim=12（58 维全量决策，见关键参数清单）
- [x] 网络规模 256→512
- [x] frame_skip=10（500Hz 物理 → 50Hz 策略）+ 速度改 base 机身系投影
- [x] 20,480 步冒烟测试通过（管线全通，评估 487.8 ≈ 零策略基线）
- [ ] 先 1M 步看曲线对不对（reward 上行、loss 收敛；140 step/s ≈ 2h）

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

## 实机关节顺序映射（2026-09-25 已确认）

```
策略动作 12 维顺序（URDF/MJCF 关节顺序）：
  [FL_hip, FL_thigh, FL_calf,
   FR_hip, FR_thigh, FR_calf,
   RL_hip, RL_thigh, RL_calf,
   RR_hip, RR_thigh, RR_calf]

实机映射（2026-09-25 用户确认）：
  CH0=FL(左前), CH1=FR(右前), CH2=RL(左后), CH3=RR(右后)
  MotorID: 髋=0x01, 大腿=0x02, 小腿=0x03

即策略动作 idx → (CH, MotorID)：
  0 FL_hip   → (CH0, 0x01)    1 FL_thigh → (CH0, 0x02)    2 FL_calf → (CH0, 0x03)
  3 FR_hip   → (CH1, 0x01)    4 FR_thigh → (CH1, 0x02)    5 FR_calf → (CH1, 0x03)
  6 RL_hip   → (CH2, 0x01)    7 RL_thigh → (CH2, 0x02)    8 RL_calf → (CH2, 0x03)
  9 RR_hip   → (CH3, 0x01)   10 RR_thigh → (CH3, 0x02)   11 RR_calf → (CH3, 0x03)
```

### 符号矩阵 S（2026-09-26 identify_motors.py 实测）

仿真正方向约定（dog_positional.xml）：髋=向左+，大腿=向后+，小腿=曲腿+（向后摆）。

实机实测正方向 → S = real 相对 sim 的符号：

| 关节 | 实机正方向 | S | | 关节 | 实机正方向 | S |
| ---- | ---------- | - | | ---- | ---------- | - |
| FL 髋 | 向左+ | +1 | | FR 髋 | 向左+ | +1 |
| FL 大腿 | 向后+ | +1 | | FR 大腿 | 向前+ | -1 |
| FL 小腿 | 曲腿+ | +1 | | FR 小腿 | 曲腿- | -1 |
| RL 髋 | 向左- | -1 | | RR 髋 | 向左- | -1 |
| RL 大腿 | 向后+ | +1 | | RR 大腿 | 向后- | -1 |
| RL 小腿 | 曲腿+ | +1 | | RR 小腿 | 曲腿- | -1 |

按 12 维动作顺序展开：
`S = [+1,+1,+1,  +1,-1,-1,  -1,+1,+1,  -1,-1,-1]`

部署换算公式：
`q_des_real[i] = q_default_real[i] + S[i] × (q_des_sim[i] - q_default_sim[i])`
（q_default_real 来自 deploy/stance_calibration.json，q_default_sim=[0,0.9,-1.8]×4）

### 站立位标定结果（2026-09-26 calibrate_stance.py 实测）

`q_default_real`（策略 12 维顺序，rad）：
`[+0.1326, +0.7616, +0.0917,  -0.1104, -0.7624, -0.1482,  -0.0856, -2.1639, -0.0933,  -0.0551, -0.5537, -0.0692]`

一致性核对：

- 4 髋 ≈ 0（±0.13 内）✅；4 小腿隐含零偏左右镜像整齐（左 ≈+1.8 / 右 ≈-1.9）✅
- 大腿 FL/FR/RR 按 S 折算 ≈ +0.55~0.76 ✅
- ⚠️ RL_thigh = -2.16，与同号的 FL_thigh(+0.76) 差 2.9 rad ≈ 166°，判定为该电机装配零偏不同（q_default_real 已吸收，不影响部署）；站立测试时重点观察左后腿

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

### 2026-09-22 — Step 3 PPO 接入与冒烟

- 先提交 Step 2 checkpoint（commit `b410a14`）
- **obs 决策修正**：原"48 维"清单加总实为 58，确认按 **58 维全量版**
  （本体10 + 关节24 + 上步动作12 + 常量偏置12），速度一律投影到 base 机身系
- `envs/dog_env.py` 改造：
  - frame_skip=10（物理 500Hz → 策略 50Hz，单局 1000 step = 20s），
    子步循环中途塌倒即停
  - `_get_kinematics()`：xmat 旋转矩阵 R，机身系 v/ω = Rᵀ × 世界系 qvel
  - reward forward 改用机身系 vx；obs 升级 58 维
- 新建 `algos/networks.py`：PolicyNetwork/ValueNetwork（hidden 512，
  Actor 299k / Critic 293k 参数）
- 新建 `algos/ppo.py`：PPOConfig 数据类 + collect_rollout/compute_gae/
  update_ppo/evaluate（entropy_beta=0.01）
- 新建 `train.py`：支持 `python train.py [总步数]`，best/final checkpoint
  落 checkpoints/，学习曲线落 logs/dog_ppo_curve.png
- **20,480 步冒烟通过**：评估 487.8±0.0（≈零策略基线 500）、H=10.91
  （σ≈0.6 理论值）、vloss=1.0 无异常；⚠ clip_frac=42% / kl=0.039 偏高，
  1M 步重点观察是否回落到健康区
- 速度 140 step/s → 1M 步约 2 小时

**下一步**：前台跑 `python train.py`（1M 步），看 reward 是否上行、
clip/kl 是否回落、σ 是否健康分化。

- **部署修复（同日）**：发现 meshdir Windows 绝对路径在 Linux 服务器必挂，
  已复制 assets 进仓库（envs/assets，18.3MB）+ meshdir 改相对路径，
  本地 20,480 步重测通过（结果与绝对路径版完全一致 487.8）。
- 部署目标：服务器 i9-14900K + RTX 4090（CUDA 12.8 驱动），克隆到
  ~/Sxy_bigdog/Trae_dog，独立 venv + cu128 torch。

### 2026-09-26 — 实机标定与站立测试

- 符号矩阵 S 实测完成（identify_motors.py）：`S=[+1,+1,+1, +1,-1,-1, -1,+1,+1, -1,-1,-1]`，详见上方映射节
- 站姿标定完成（calibrate_stance.py stance → deploy/stance_calibration.json），
  一致性核对通过；⚠️ RL_thigh 零偏与其他腿差 ≈2.9 rad（装配零偏，q_default_real 已吸收）
- 站立保持测试通过（stand_hold.py，146s 无发散）：kp=60 时小腿静差 ≈0.12 rad 偏软，
  实机需 kp=100（高于仿真值属允许方向）；stand_hold.py 支持命令行 kp/kd 参数
- calibrate_stance.py 加名称参数（stance/lie）+ 无反馈拒绝保存保护 + 轮询 5 轮
- 趴姿标定完成（lie → deploy/lie_calibration.json，12 台全到位，左右对称性 ✅）
- 新建 stand_up.py：趴姿→(Enter)→10s 余弦平滑起立→站姿保持→(Enter)→10s 趴下→失能，
  单人免扶狗操作；Ctrl+C 为应急立即失能（狗会摔）
- 站立调平完成（stand_up.py STAND 阶段在线微调）：实机初始标定存在前高后低，
  经微调确认后腿小腿需抬高 `trim_rear=+0.1 rad`（仿真系），前腿 `trim_front=0.0`。
  参数固化到 deploy/posture_trim.json，部署时折叠进零偏：
  `q_default_eff[i] = q_default_real[i] + S[i] * (-trim)（仅小腿关节）`。
- 调平姿态复现确认（kp=150/kd=2，STAND 阶段，用户目视水平 ✅）：
  稳态误差髋/大腿 ≤0.064 rad，小腿残差 0.036~0.114 rad（最大 RR_calf），
  小腿误差不随 kp 增大缩小（力矩/电压上限迹象）。误差网格已存档进
  posture_trim.json 的 validation 字段，作为部署站立基线参考。
- stand_up.py 升级：启动自动加载 deploy/posture_trim.json 作为初始微调
  （load_trim()），起立即达调平姿态，无需再手动输 `r 0.1`；r/f 在线微调
  在此基础叠加；启动/退出打印含初始加载量。
- 教训：JSON 数字不允许前导 `+` 号（`+0.0359` 触发 JSONDecodeError），
  posture_trim.json 曾因此在 NUC 解析失败（崩溃发生在使能电机前，无风险），
  已修复。
- NUC 网络变更：原 192.168.1.100 已失效，新地址 192.168.155.27（2026-09-26 用户确认）。
  NUC 关机后无法连接属正常，下次部署前先确认开机并 `hostname -I` 核对。
- ✅ 已解决（2026-09-26 用户确认）：修复版（无 `+` 号）posture_trim.json 已重新同步到
  NUC，stand_up.py 运行正常，无需再执行 sed 修复。

### 2026-09-26 — 新方向冻结：指令条件策略 + MJX GPU 训练（决策 5/6）

- 9/22 的 58 维模型仅会自主前进、无控制语义，用户决定**不作为部署候选**；
  新目标 = 手柄遥控（vx/vy/yaw_rate 三维全向指令，含后退/原地转/停车）。
- 61 维观测布局与实机传感器逐维核对完成（见决策 5.1 对照表）：全部有信号
  来源，但发现并定案 4 项：
  1. **obs[1:4] rpy → projected gravity**（BMI088 无磁力计，yaw 必漂移）；
  2. 关节反馈 real→sim 逆映射公式定案（run_policy.py 待实现）；
  3. 关节速度 S×dq_real + ~20Hz 低通；
  4. obs[46:58] 实机必须固定填仿真常量 [0,0.9,-1.8]×4。
- 训练后端：CPU 20env 的瓶颈是 mj_step（非网络），定案 **MJX/JAX 重写**
  （非 Isaac/PhysX，保 MuJoCo 同源动力学），4090 Linux 服务器训练，
  Flax 自写 PPO，checkpoint 双存（Flax + 部署用 .npz）。
- 冻结内容详见上方决策 5（61 维布局/指令课程/跟踪奖励/脚本化评估）与
  决策 6（MJX 移植要点/阶段 0~6/手柄安全/contact 掩码待办）。
- **下一步（待用户发话）**：阶段 0/1 —— 4090 服务器环境 + dog_mjx.xml
  适配 + `mjx.test_model()` 报告 + MJX/CPU 零策略动力学一致性校验。

### 2026-09-26（晚）— MJX 阶段 1 完成：模型适配 + 一致性校验双关通过

- 本地 venv 装好 CPU 版 jax/jaxlib 0.11.2 + mujoco-mjx 3.13.0（与 mujoco
  3.13.0 精确配对）+ flax 0.12.9；阿里云镜像可用，TRAE 沙箱需外放 pip。
- 产出：`envs/dog_mjx.xml`、`envs/scene_mjx.xml`、
  `scripts/test_mjx_compat.py`（put_model/make_data/jit 三关）、
  `scripts/verify_mjx_consistency.py`（A/B/C 三组轨迹对比+出图）、
  `scripts/probe_mjx_options.py`、`scripts/probe_mjx_geometry.py`（实测探针，
  推翻"必须金字塔锥/condim=3"的预判）。
- 最终适配仅 3 处实质改动：积分器 implicitfast、cylinder→capsule（唯一硬限制，
  `CYLINDER-BOX collisions not implemented`）、删 sensor 段；椭圆锥
  impratio=100 与 condim=6 均保留。
- 校验结果：纯引擎差 0.187°/0.38mm，模型适配差 0.36°/0.21mm，站立高度三方
  重合（图 logs/mjx_consistency.png）。
- **下一步**：阶段 0 收尾——同步到 4090 服务器装 CUDA 版 jax，跑 1024/2048
  env 的 mjx.step 吞吐冒烟（拿到真实 step/s 再定 env 数与 rollout 尺寸）；
  之后进入阶段 2 写 `envs/dog_env_mjx.py`（61 维观测/DR/指令课程）。
  遗留：动态步行一致性（摆腿时 capsule 自碰撞包膜差异）阶段 2 后复测。
