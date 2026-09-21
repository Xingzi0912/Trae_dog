"""DogEnv —— 机器狗 Gymnasium 环境最小骨架（Step 1）

只验证一件事：零策略下机器狗能否在 MuJoCo 中维持站立姿态不塌。

零策略语义（agents.md 决策1）：
    action = 0  →  q_des = q_default + 0 × 0.25 = q_default
    q_default = [0, 0.9, -1.8] × 4  （hip=0, thigh=0.9, calf=-1.8，四条腿一致）
PD 控制器（kp=60, kv=2，由 XML 的 <position> actuator 实现）应能 hold 住这个姿态。

本文件暂不实现：
    - 48 维观测（Step 2）—— 当前 obs 直接返回 qpos 占位
    - reward（Step 2）—— 当前恒为 0
    - 域随机化（Step 4）

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
    """机器狗环境最小骨架 - Step 1 验证零策略站立不塌"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # 默认站立关节角，顺序 [FL, FR, RL, RR] × [hip, thigh, calf]
    # 与 dog_positional.xml 的 keyframe home 的 ctrl 一致
    Q_DEFAULT = np.array(
        [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8],
        dtype=np.float64,
    )
    ACTION_SCALE = 0.25  # q_des = q_default + action × 0.25

    def __init__(self, scene_path: str = None, render_mode: str = None):
        if scene_path is None:
            scene_path = str(Path(__file__).parent / "scene_positional.xml")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        # 12 维动作 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Step 1 占位 obs：直接返回 qpos（19 维 = 7 free + 12 joint）
        # Step 2 会改成 48 维（本体状态 + 关节 + 上一步动作 + 默认偏置）
        self._nq = self.model.nq
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._nq,), dtype=np.float32
        )

        # 模型结构自检（加载后立刻报错好排查 XML 问题）
        assert self._nq == 19, f"Expected nq=19 (7 freejoint + 12 joints), got {self._nq}"
        assert self.model.nv == 18, f"Expected nv=18 (6 free + 12 joints), got {self.model.nv}"
        assert self.model.nu == 12, f"Expected nu=12 actuators, got {self.model.nu}"

        # keyframe home id（reset 时用 mj_resetDataKeyframe 一步到位）
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        assert self._home_key_id >= 0, "keyframe 'home' not found in XML"

        # base_link body id 用于读 base z
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
        mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        base_z = float(self.data.xpos[self._base_body_id, 2])
        terminated = base_z < self._z_terminate
        truncated = self._step_count >= self._max_steps
        reward, reward_info = self._compute_reward(action)
        info = {"base_z": base_z, "step": self._step_count, **reward_info}
        # 更新上一步动作（必须放在 _compute_reward 之后，否则抖动惩罚会用错基准）
        self._last_action = action.copy()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        """reward 4 分量（agents.md 第 113-118 行）

        r = +1.0  × forward_x_speed              # 前进速度（主目标）
          - 0.05 × Σ action²                      # 能耗惩罚
          - 0.001 × Σ (action - last_action)²     # 抖动惩罚
          + 0.5  × upright_bonus                  # 别摔倒（roll,pitch<0.4）

        Step 2 阶段：obs 维度暂未扩展，reward 内部直接从 data 读状态，
        不依赖 obs 数组。Step 3 接 PPO 时再统一 obs 48 维规范化。
        """
        # 前进速度：base 线速度 x 分量（世界系，keyframe home 时 qvel[0]=0）
        # Step 2 验证阶段狗不转向，世界系 = base 系；Step 3 再投影到 base 系
        forward_x_speed = float(self.data.qvel[0])

        # 能耗（action 已 clip 到 [-1,1]）
        action_sq_sum = float(np.sum(action ** 2))

        # 抖动（与上一步动作差）
        action_diff = action - self._last_action
        action_diff_sq_sum = float(np.sum(action_diff ** 2))

        # 姿态：四元数 → roll, pitch（MuJoCo quat 顺序 [w, x, y, z]）
        quat = self.data.qpos[3:7]
        roll, pitch, _ = self._quat_to_rpy(quat)
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
        # Step 1 占位：直接返回 qpos。Step 2 改成 48 维完整观测。
        return self.data.qpos.copy().astype(np.float32)

    def render(self):
        if self.render_mode != "rgb_array":
            raise gym.error.UnsupportedMode(
                f"render_mode={self.render_mode!r}（Step 1 仅支持 rgb_array；"
                "交互式可视化用 mujoco.viewer，见 Step 1.5）"
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
# 自检：跑零策略 1000 步，打印 base z 曲线（Step 1.3 验证）
# 直接 `python dog_env.py` 运行
# 期望：1000 步后 base_z 仍 > 0.25m，狗不塌
# ============================================================
if __name__ == "__main__":
    env = DogEnv()
    obs, info = env.reset()
    print(f"[reset] base_z = {info['base_z']:.4f} m （站立目标 ≈ 0.27m）")
    print(f"[reset] obs (qpos 19 维) = {obs}")
    print(f"        前 7 维 (freejoint xyz+quat): {obs[:7]}")
    print(f"        后 12 维 (关节角): {obs[7:]}")
    print()

    z_history = [info["base_z"]]
    action = np.zeros(12, dtype=np.float32)  # 零策略
    print("开始跑零策略 1000 步 ...")
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
