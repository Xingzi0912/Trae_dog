"""DogEnv —— 机器狗 Gymnasium 环境（Step 4：域随机化版）

控制链路：
    PPO 输出 12 维 action ∈ [-1,1]
      → q_des = q_default + action × 0.25
      → XML <position> actuator（kp=60, kv=2）PD 输出力矩
    action = 0 时 q_des = q_default = [0, 0.9, -1.8] × 4，维持站立姿态。

时间尺度：
    物理步长 = MuJoCo 默认 0.002s（500Hz）
    frame_skip = 10 → 策略频率 50Hz（dt=0.02s，对齐 agents.md 实机控制周期）
    1000 个 policy step = 20s（legged_gym 标准单局时长）

观测空间（58 维全量版，base 机身系）：
    [ 0]    base_z 高度（世界系）                    1
    [ 1: 4] roll / pitch / yaw                       3
    [ 4: 7] base 线速度（机身系）                    3
    [ 7:10] base 角速度（机身系）                    3
    [10:22] 12 关节位置（绝对角）                   12
    [22:34] 12 关节速度                             12
    [34:46] 上一步动作（助稳定）                    12
    [46:58] q_default 常量偏置（攻势项）            12
    合计 58。

reward 4 分量（agents.md）：
    r = +1.0  × forward_x_speed（机身系 vx）
      - 0.05  × Σ action²
      - 0.001 × Σ (action - last_action)²
      + 0.5  × upright_bonus（|roll|,|pitch| < 0.4）

Step 4 域随机化（每次 reset 重新采样，见 RandConfig）：
    ① 地面摩擦 floor.geom_friction[0] ∈ [0.5, 1.8]
       （MuJoCo 接触摩擦 = 两 geom 相乘，足底 0.9 → 接触对 0.45~1.62）
    ② 机身质量 ×[0.8, 1.2]
    ③ PD 增益 kp/kv 各 ×[0.8, 1.2]
    ④ 初始状态：base z/姿态、关节角 ±0.05rad、关节速度 ±0.5rad/s 扰动
    ⑤ 动作延迟 0~2 个策略步（0~40ms，对齐实机通信延迟）
    ⑥ 观测高斯噪声（速度类为主，模拟 IMU/编码器误差）

调用：
    env = DogEnv()                       # 默认开启域随机化
    env = DogEnv(rand_config=RandConfig(enabled=False))  # 标称环境
    obs, info = env.reset(seed=0)
    obs, r, term, trunc, info = env.step(np.zeros(12))
"""

from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from pathlib import Path


@dataclass
class RandConfig:
    """域随机化参数（范围为【每局独立采样】的均匀分布）

    范围设计参考 legged_gym go2 默认 DR，按本狗 6.5kg/位置控制做了收敛：
        - 摩擦/质量/PD：±20% 起步，避免一开 DR 就训不动
        - 初态扰动：让评估 10 局不再是同一条轨迹（eval std 有意义）
        - 延迟/噪声：实机 sim2real 的主要 gap 源
    """
    enabled: bool = True
    # ① 地面摩擦（绝对值，floor geom friction[0]）
    friction_range: tuple = (0.5, 1.8)
    # ② 机身质量（乘性）
    base_mass_range: tuple = (0.8, 1.2)
    # ③ PD 增益（乘性，整局各关节共用一个尺度）
    kp_scale_range: tuple = (0.8, 1.2)
    kv_scale_range: tuple = (0.8, 1.2)
    # ④ 初始状态
    init_base_z_range: tuple = (0.27, 0.32)   # keyframe 标称 0.27
    init_rpy_noise: float = 0.1               # rad，三轴同幅
    init_joint_noise: float = 0.05            # rad
    init_base_vel_noise: float = 0.1          # m/s（线速度）/ rad/s（角速度）
    init_jointvel_noise: float = 0.5          # rad/s
    # ⑤ 动作延迟（策略步，整数闭区间）
    action_delay_range: tuple = (0, 2)
    # ⑥ 观测噪声总开关/尺度（1.0=默认强度，0=关闭）
    obs_noise_scale: float = 1.0


# 各观测分量的加性高斯噪声标准差（对应 58 维布局，× obs_noise_scale）
# 参考实机传感器精度：IMU 速度 ≈0.1m/s、陀螺仪 ≈0.05rad/s、关节编码器位置 ≈0.01rad
_OBS_NOISE_STD = np.concatenate([
    [0.01],                              #  0    base_z
    np.full(3, 0.01),                    #  1:4  rpy
    np.full(3, 0.10),                    #  4:7  机身系线速度
    np.full(3, 0.05),                    #  7:10 机身系角速度
    np.full(12, 0.01),                   # 10:22 关节位置
    np.full(12, 0.15),                   # 22:34 关节速度
    np.zeros(12),                        # 34:46 last_action（自身已知量）
    np.zeros(12),                        # 46:58 q_default（常量）
]).astype(np.float64)


class DogEnv(gym.Env):
    """机器狗环境 - 50Hz 策略 / 58 维观测 / 位置控制 / 域随机化"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # 默认站立关节角，顺序 [FL, FR, RL, RR] × [hip, thigh, calf]
    # 与 dog_positional.xml 的 keyframe home 的 ctrl 一致
    Q_DEFAULT = np.array(
        [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8],
        dtype=np.float64,
    )
    ACTION_SCALE = 0.25  # q_des = q_default + action × 0.25
    FRAME_SKIP = 10      # 0.002s × 10 = 0.02s（50Hz 策略）
    OBS_DIM = 58

    def __init__(self, scene_path: str = None, render_mode: str = None,
                 rand_config: RandConfig = None):
        if scene_path is None:
            scene_path = str(Path(__file__).parent / "scene_positional.xml")
        if rand_config is None:
            rand_config = RandConfig()
        self.rand_cfg = rand_config

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # 12 维动作 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # 58 维规范观测（见模块 docstring 布局）
        # 外部 RunningMeanStd 做归一化，本空间保持原始物理尺度
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )

        # 模型结构自检（加载后立刻报错好排查 XML 问题）
        assert self.model.nq == 19, f"Expected nq=19 (7 freejoint + 12 joints), got {self.model.nq}"
        assert self.model.nv == 18, f"Expected nv=18 (6 free + 12 joints), got {self.model.nv}"
        assert self.model.nu == 12, f"Expected nu=12 actuators, got {self.model.nu}"

        # keyframe home id（reset 时用 mj_resetDataKeyframe 一步到位）
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        assert self._home_key_id >= 0, "keyframe 'home' not found in XML"

        # base_link body id 用于读 base z / xmat 姿态旋转矩阵
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        assert self._base_body_id >= 0, "body 'base_link' not found in XML"

        # floor geom id（摩擦随机化目标，scene_positional.xml 中命名 floor）
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        assert self._floor_geom_id >= 0, "geom 'floor' not found in scene XML"

        # 随机化要改的标称模型参数（每次 reset 先恢复再采样，保证各局独立）
        self._nom_body_mass = self.model.body_mass.copy()
        self._nom_geom_friction = self.model.geom_friction.copy()
        self._nom_gainprm = self.model.actuator_gainprm.copy()
        self._nom_biasprm = self.model.actuator_biasprm.copy()

        # 终止阈值与步数上限（agents.md 关键参数清单）
        self._z_terminate = 0.25  # base z < 0.25m 视为塌倒
        self._max_steps = 1000
        self._step_count = 0

        # 上一步动作（reward 抖动惩罚需要；reset 时重置）
        self._last_action = np.zeros(12, dtype=np.float64)

        # 动作延迟队列（reset 时按采样延迟重建）
        self._ctrl_queue = deque()

        self._renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 用 keyframe home 重置：自动设好 qpos + ctrl 到站立姿态
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        # 恢复标称模型参数，再按新一局采样（DR 关闭时即为标称环境）
        self.model.body_mass[:] = self._nom_body_mass
        self.model.geom_friction[:] = self._nom_geom_friction
        self.model.actuator_gainprm[:] = self._nom_gainprm
        self.model.actuator_biasprm[:] = self._nom_biasprm
        if self.rand_cfg.enabled:
            self._apply_randomization()

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._last_action = np.zeros(12, dtype=np.float64)  # 抖动惩罚基准

        # 重建动作延迟队列：延迟 d 步 → 预置 d 个 q_default，前 d 步执行默认姿态
        d = self._sample_delay()
        self._ctrl_queue = deque(
            [self.Q_DEFAULT.copy() for _ in range(d)], maxlen=d + 1
        )

        obs = self._get_obs()
        info = {"base_z": float(self.data.xpos[self._base_body_id, 2])}
        return obs, info

    def step(self, action):
        # action ∈ [-1, 1]^12 → q_des = q_default + action × 0.25
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_des = self.Q_DEFAULT + action * self.ACTION_SCALE

        # 动作延迟：新指令入队，执行队首（最旧）指令；delay=0 时即时执行
        self._ctrl_queue.append(q_des)
        self.data.ctrl[:] = self._ctrl_queue[0]

        # 10 个物理子步 = 1 个策略步（500Hz → 50Hz）
        # 中途塌倒立即停止子步进，停在塌倒时刻的状态
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            if float(self.data.xpos[self._base_body_id, 2]) < self._z_terminate:
                break

        self._step_count += 1
        kin = self._get_kinematics()
        terminated = kin["base_z"] < self._z_terminate
        truncated = self._step_count >= self._max_steps
        reward, reward_info = self._compute_reward(action, kin)
        obs = self._get_obs()
        info = {"base_z": kin["base_z"], "step": self._step_count, **reward_info}
        # 更新上一步动作（必须放在 _compute_reward 之后，否则抖动惩罚会用错基准）
        self._last_action = action.copy()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------
    # 域随机化：每次 reset 采样一组新参数
    # ------------------------------------------------------------
    def _apply_randomization(self):
        cfg = self.rand_cfg
        rng = self.np_random

        # ① 地面摩擦
        fric = rng.uniform(*cfg.friction_range)
        self.model.geom_friction[self._floor_geom_id, 0] = fric

        # ② 机身质量（只改 base_link；腿部质量保持标称）
        m_scale = rng.uniform(*cfg.base_mass_range)
        self.model.body_mass[self._base_body_id] = \
            self._nom_body_mass[self._base_body_id] * m_scale

        # ③ PD 增益
        #    position actuator：gainprm[:,0]=kp，biasprm[:,1]=-kp，biasprm[:,2]=-kv
        kp_scale = rng.uniform(*cfg.kp_scale_range)
        kv_scale = rng.uniform(*cfg.kv_scale_range)
        self.model.actuator_gainprm[:, 0] = self._nom_gainprm[:, 0] * kp_scale
        self.model.actuator_biasprm[:, 1] = self._nom_biasprm[:, 1] * kp_scale
        self.model.actuator_biasprm[:, 2] = self._nom_biasprm[:, 2] * kv_scale

        # ④ 初始状态扰动
        # base 高度 / 姿态
        self.data.qpos[2] = rng.uniform(*cfg.init_base_z_range)
        rpy0 = rng.uniform(-cfg.init_rpy_noise, cfg.init_rpy_noise, size=3)
        self.data.qpos[3:7] = self._rpy_to_quat(rpy0)
        # 关节位置：加扰后 clip 回关节限位（jnt_range 第 0 行是 freejoint，跳过）
        self.data.qpos[7:] += rng.uniform(
            -cfg.init_joint_noise, cfg.init_joint_noise, size=12
        )
        lo = self.model.jnt_range[1:, 0]
        hi = self.model.jnt_range[1:, 1]
        self.data.qpos[7:] = np.clip(self.data.qpos[7:], lo, hi)
        # 初始速度
        self.data.qvel[0:3] = rng.uniform(
            -cfg.init_base_vel_noise, cfg.init_base_vel_noise, size=3
        )
        self.data.qvel[3:6] = rng.uniform(
            -cfg.init_base_vel_noise, cfg.init_base_vel_noise, size=3
        )
        self.data.qvel[6:] = rng.uniform(
            -cfg.init_jointvel_noise, cfg.init_jointvel_noise, size=12
        )

    def _sample_delay(self) -> int:
        """采样本局动作延迟（策略步）；DR 关闭时恒为 0"""
        if not self.rand_cfg.enabled:
            return 0
        lo, hi = self.rand_cfg.action_delay_range
        return int(self.np_random.integers(int(lo), int(hi) + 1))

    # ------------------------------------------------------------
    # 运动学：世界系状态 + 投影到 base 机身系（50Hz 每 policy step 算一次）
    # ------------------------------------------------------------
    def _get_kinematics(self) -> dict:
        base_z = float(self.data.xpos[self._base_body_id, 2])
        quat = self.data.qpos[3:7]
        roll, pitch, yaw = self._quat_to_rpy(quat)

        # freejoint qvel: [0:3]=世界系线速度, [3:6]=世界系角速度
        # xmat = base body 相对世界系的旋转矩阵 R（行主序 9 个数）
        # 机身系向量 v_b = R^T @ v_w
        R = self.data.xmat[self._base_body_id].reshape(3, 3)
        lin_vel_base = R.T @ self.data.qvel[0:3]
        ang_vel_base = R.T @ self.data.qvel[3:6]

        return {
            "base_z": base_z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "lin_vel_base": lin_vel_base,
            "ang_vel_base": ang_vel_base,
        }

    def _compute_reward(self, action: np.ndarray, kin: dict) -> tuple[float, dict]:
        """reward 4 分量（agents.md）

        r = +1.0  × forward_x_speed              # 前进速度（机身系 vx，主目标）
          - 0.05 × Σ action²                      # 能耗惩罚
          - 0.001 × Σ (action - last_action)²     # 抖动惩罚
          + 0.5  × upright_bonus                  # 别摔倒（roll,pitch<0.4）
        """
        # 前进速度：机身系 vx。狗转向后"前进"语义仍正确
        forward_x_speed = float(kin["lin_vel_base"][0])

        # 能耗（action 已 clip 到 [-1,1]）
        action_sq_sum = float(np.sum(action ** 2))

        # 抖动（与上一步动作差）；policy step=0.02s 后该惩罚尺度比 Step 2 合理
        action_diff = action - self._last_action
        action_diff_sq_sum = float(np.sum(action_diff ** 2))

        # 姿态：upright 二值奖励
        roll, pitch = kin["roll"], kin["pitch"]
        upright = 1.0 if (abs(roll) < 0.4 and abs(pitch) < 0.4) else 0.0

        r_forward = 1.0 * forward_x_speed
        r_energy = -0.05 * action_sq_sum
        r_jerk = -0.001 * action_diff_sq_sum
        r_upright = 0.5 * upright
        reward = r_forward + r_energy + r_jerk + r_upright

        info = {
            "reward_forward": r_forward,
            "reward_energy": r_energy,
            "reward_jerk": r_jerk,
            "reward_upright": r_upright,
            "forward_x_speed": forward_x_speed,
            "roll": float(roll),
            "pitch": float(pitch),
            "upright": upright,
        }
        return reward, info

    @staticmethod
    def _quat_to_rpy(quat: np.ndarray) -> np.ndarray:
        """四元数 [w, x, y, z] → [roll, pitch, yaw]（ZYX 顺序，弧度）

        标准 ZYX 欧拉角分解，机器狗只关心 roll/pitch（pitch=前后倾，roll=左右倾）。
        yaw 用于转向（trot 直行时不重要）。
        """
        w, x, y, z = quat
        # roll 绕 x 轴
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        # pitch 绕 y 轴（asin 有数值范围限制，需 clip）
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        # yaw 绕 z 轴
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    @staticmethod
    def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
        """[roll, pitch, yaw]（弧度）→ 四元数 [w, x, y, z]，_quat_to_rpy 的逆"""
        r, p, y = rpy
        cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
        cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
        cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
        ])

    def _get_obs(self):
        """构造 58 维观测（布局见模块 docstring）；DR 开启时叠加传感器噪声"""
        kin = self._get_kinematics()
        obs = np.concatenate([
            [kin["base_z"]],                 #  0   base z（世界系）
            [kin["roll"], kin["pitch"], kin["yaw"]],       #  1:4
            kin["lin_vel_base"],             #  4:7  机身系线速度
            kin["ang_vel_base"],             #  7:10 机身系角速度
            self.data.qpos[7:],              # 10:22 关节位置（绝对角）
            self.data.qvel[6:],              # 22:34 关节速度
            self._last_action,               # 34:46 上一步动作
            self.Q_DEFAULT,                  # 46:58 默认偏置常量
        ])
        if self.rand_cfg.enabled and self.rand_cfg.obs_noise_scale > 0:
            obs = obs + _OBS_NOISE_STD * self.rand_cfg.obs_noise_scale \
                       * self.np_random.standard_normal(self.OBS_DIM)
        return obs.astype(np.float32)

    def render(self):
        if self.render_mode != "rgb_array":
            raise gym.error.UnsupportedMode(
                f"render_mode={self.render_mode!r}（当前仅支持 rgb_array；"
                "交互式可视化用 mujoco.viewer，见 scripts/view_stand.py）"
            )
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ============================================================
# 自检：标称环境（关 DR）零策略跑满 1000 step，打印 base z
# 直接 `python dog_env.py` 运行
# 期望：base_z 始终 > 0.25m，狗不塌；观测为 58 维
# ============================================================
if __name__ == "__main__":
    env = DogEnv(rand_config=RandConfig(enabled=False))
    obs, info = env.reset()
    print(f"[reset] base_z = {info['base_z']:.4f} m （站立目标 ≈ 0.27m）")
    print(f"[reset] obs {obs.shape[0]} 维")
    print(f"        [0] base_z      = {obs[0]:.4f}")
    print(f"        [1:4] rpy       = {obs[1:4]}")
    print(f"        [4:7] lin vel b = {obs[4:7]}")
    print(f"        [7:10] ang vel b= {obs[7:10]}")
    print(f"        [10:22] joint q = {obs[10:22]}")
    print(f"        [46:58] q_def   = {obs[46:58]}")
    print()

    z_history = [info["base_z"]]
    action = np.zeros(12, dtype=np.float32)  # 零策略
    print("开始跑零策略 1000 policy step（frame_skip=10，20s 仿真）...")
    for step in range(1000):
        obs, r, terminated, truncated, info = env.step(action)
        z_history.append(info["base_z"])
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:4d}  base_z = {info['base_z']:.4f} m")
        if terminated:
            print(
                f"\n[terminated at step {step+1}] base_z={info['base_z']:.4f} m "
                f"< {env._z_terminate}m，狗塌了！"
            )
            break
    else:
        print(f"\n[truncated at step {env._step_count}] 完成 1000 步，狗没塌 ✅")

    z_arr = np.array(z_history)
    print()
    print("=" * 50)
    print("总结：")
    print(f"  base_z  min={z_arr.min():.4f}  max={z_arr.max():.4f}  final={z_arr[-1]:.4f}")
    print(f"  波动幅度 (max-min) = {z_arr.max()-z_arr.min():.4f} m")
    if z_arr.min() >= env._z_terminate:
        print(f"  验证结果：✅ 零策略能 hold 住站立姿态（始终 > {env._z_terminate}m）")
    else:
        print(f"  验证结果：❌ 狗塌了（z 曾跌破 {env._z_terminate}m），需调整 PD 或初始姿态")
    env.close()
