"""DogEnv —— 机器狗 Gymnasium 环境（Step 3：接入 PPO 版）

控制链路：
    PPO 输出 12 维 action ∈ [-1,1]
      → q_des = q_default + action × 0.25
      → XML <position> actuator（kp=60, kv=2）PD 输出力矩
    action = 0 时 q_des = q_default = [0, 0.9, -1.8] × 4，维持站立姿态。

时间尺度（Step 3 新增）：
    物理步长 = MuJoCo 默认 0.002s（500Hz）
    frame_skip = 10 → 策略频率 50Hz（dt=0.02s，对齐 agents.md 实机控制周期）
    1000 个 policy step = 20s（legged_gym 标准单局时长）

观测空间（Step 3 决策：58 维全量版，base 机身系）：
    [ 0]    base_z 高度（世界系）                    1
    [ 1: 4] roll / pitch / yaw                       3
    [ 4: 7] base 线速度（机身系）                    3
    [ 7:10] base 角速度（机身系）                    3
    [10:22] 12 关节位置（绝对角）                   12
    [22:34] 12 关节速度                             12
    [34:46] 上一步动作（助稳定）                    12
    [46:58] q_default 常量偏置（攻势项）            12
    合计 58。
    说明：agents.md 原写"48 维"但其分量清单加总为 58，2026-09-22 确认按
          58 维全量实现（关节位置用绝对值 + 保留常量偏置）。

reward 4 分量（agents.md）：
    r = +1.0  × forward_x_speed（机身系 vx，Step 3 从世界系改投影）
      - 0.05  × Σ action²
      - 0.001 × Σ (action - last_action)²
      + 0.5  × upright_bonus（|roll|,|pitch| < 0.4）

暂不实现：obs 归一化 + 域随机化（Step 4）。

调用：
    env = DogEnv()
    obs, info = env.reset()
    obs, r, term, trunc, info = env.step(np.zeros(12))
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from pathlib import Path


class DogEnv(gym.Env):
    """机器狗环境 - 50Hz 策略 / 58 维观测 / 位置控制"""

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

    def __init__(self, scene_path: str = None, render_mode: str = None):
        if scene_path is None:
            scene_path = str(Path(__file__).parent / "scene_positional.xml")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # 12 维动作 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # 58 维规范观测（见模块 docstring 布局）
        # Step 4 会在外部加 RunningMeanStd 归一化，本空间保持原始物理尺度
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

        # 终止阈值与步数上限（agents.md 关键参数清单）
        self._z_terminate = 0.25  # base z < 0.25m 视为塌倒
        self._max_steps = 1000
        self._step_count = 0

        # 上一步动作（reward 抖动惩罚需要；reset 时重置）
        self._last_action = np.zeros(12, dtype=np.float64)

        self._renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 用 keyframe home 重置：自动设好 qpos + ctrl 到站立姿态
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._last_action = np.zeros(12, dtype=np.float64)  # 抖动惩罚基准
        obs = self._get_obs()
        info = {"base_z": float(self.data.xpos[self._base_body_id, 2])}
        return obs, info

    def step(self, action):
        # action ∈ [-1, 1]^12 → q_des = q_default + action × 0.25
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_des = self.Q_DEFAULT + action * self.ACTION_SCALE
        self.data.ctrl[:] = q_des

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
        # 前进速度：机身系 vx。狗转向后"前进"语义仍正确（Step 3 改）
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

    def _get_obs(self):
        """构造 58 维观测（布局见模块 docstring）"""
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
# 自检：零策略跑满 1000 个 policy step（20s 仿真），打印 base z
# 直接 `python dog_env.py` 运行
# 期望：base_z 始终 > 0.25m，狗不塌；观测为 58 维
# ============================================================
if __name__ == "__main__":
    env = DogEnv()
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
