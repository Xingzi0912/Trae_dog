"""DogEnvMJX —— 机器狗 MJX(JAX) 向量化环境（阶段 2，agents.md 决策 5/6）

与旧 dog_env.py（CPU gym.Env，58 维）的关系：
    同一几何/质量/执行器参数（scene_mjx.xml），同一零策略语义与时间尺度
    （q_des = q_default + action×0.25、frame_skip=10、dt=0.002、50Hz）。
    本环境为纯函数式 JAX 实现：reset/step 全部 jit+vmap，单卡 4090 目标
    2048 env；旧环境保留作一致性基准与训练后渲染。

与旧环境的关键差异：
  1. 观测 61 维（决策 5.1）：rpy → projected gravity，尾部追加 3 维速度指令
  2. obs[7:10] 直接取 qvel[3:6] —— freejoint 角速度本身就是机身系
     （2026-10-10 探针实证：mj_objectVelocity 世界系角速度 == R @ qvel[3:6]，与
     四元数积分约定一致；旧 env 的 R.T @ qvel[3:6] 在站立时 R≈I 不可见，但语义
     是错的）。新语义与实机 IMU gyro 直出天然对齐，部署零转换。
  3. obs[34:46] last_action 修正旧环境的 off-by-one：obs_t 携带的是产生该状态
     的动作 a_t（旧环境 obs 里是 a_{t-1}）
  4. 指令 [vx, vy, yaw_rate]：每个 env 独立采样，episode 内每 2~4s
     （100~200 策略步）重采样，17% 概率零指令（决策 5.2）。课程收缩由训练脚本
     通过 EnvConfig 的 cmd_* 范围参数实现，环境本身始终按给定范围采样
  5. 奖励改速度跟踪高斯（决策 5.3），v1 不加步态相位/腾空奖励
  6. DR 分两层（2026-10-13 重构 + K 池化，纯性能原因）：
     a) 模型级（摩擦/质量/kp/kv）在 vmap **外层**注入。实测（probe_batch，N=2048）：
        模型一旦按 env 批化——无论全批化 (0,0) 还是旧版「vmap 内 replace」的混批
        ——mjx.step 快路径立即失效，吞吐从 21,000 掉到 ~660 env·物理子步/s（31 倍）；
        共享**非批化**模型（in_axes=None 广播）则 20,963。
     b) 为在保住快路径的同时恢复 DR 多样度，引入 **K 池化**（EnvConfig.dr_pools=K）：
        外层 K 份池模型（vmap 出按池批化的 (K,...) 模型），池内 n_envs/K 个 env 共享
        一份；state/obs 对外仍是 (N,)，仅 K>1 时内部 reshape 成 (K, N/K)。
        实测 K∈{128,256,512} 仍是快路径（~1.2x 开销），K∈{8,32,64} 或 K=2048
        （=per-env 全批化）掉慢路径（见 scripts/bench_mjx_throughput.py）。
        K=1 即 a) 的「整批共享」路径（最快，无 DR 多样度）；训练默认 K=256。
     c) 便宜的 per-env DR 全部保留：初态扰动/动作延迟(0~2 策略步)/观测噪声/
        指令采样都在 Data/state 里逐 env 采样，与模型无关

观测布局（61 维，除 base_z 外全部机身系）：
    [ 0]    base_z（世界系高度）= qpos[2]
    [ 1: 4] projected gravity g_b = Rᵀ·[0,0,-1]（= -R[2,:]，站立≈[0,0,-1]）
    [ 4: 7] 机身系线速度 = Rᵀ·qvel[0:3]
    [ 7:10] 机身系角速度 = qvel[3:6]（freejoint 角速度本就是机身系，见上）
    [10:22] 12 关节角  = qpos[7:]
    [22:34] 12 关节速度 = qvel[6:]
    [34:46] last_action（产生当前状态的动作）
    [46:58] q_default 常量（2026-10-23 起为加电站立位，见 Q_DEFAULT；实机必须填同一常量）
    [58:61] 指令 [vx, vy, yaw_rate]（m/s, m/s, rad/s）

用法（训练循环骨架）：
    env = DogEnvMJX(n_envs=2048)
    state, obs = env.reset(jax.random.PRNGKey(0))
    state, obs, rew, term, trunc, info = env.step(state, actions)
    # episode 结束（term|trunc）的 env 由训练脚本做 auto-reset：
    rng, k = jax.random.split(rng)
    fresh, fresh_obs = env.reset(k)          # 或缓存一份复用
    state = tree_where_done(done, fresh, state)  # done = term|trunc
    obs   = tree_where_done(done, fresh_obs, obs)

注意：mjx.step 不支持子步中途塌倒早停（scan 固定 10 步），与旧环境在摔倒
瞬间的轨迹会有 ≤9 物理步的尾部差异，对训练无影响。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from envs.gait import (gait as gait_fn, gait_ik as gait_ik_fn,
                       gait_ik_phase as gait_ik_phase_fn,
                       gait_freq as gait_freq_fn)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvConfig:
    """环境与 DR 参数（范围参照旧 dog_env.py RandConfig，agents.md 决策 5/6）

    指令课程：训练脚本收缩 cmd_vx/cmd_vy/cmd_yaw 后重建 env 即可
    （初期 vx[0,0.8]/vy 0/yaw ±0.5 → 末期默认全范围）。
    """
    # 时间尺度
    frame_skip: int = 10          # 0.002s×10 = 0.02s（50Hz 策略）
    max_steps: int = 1000         # 20s 截断
    z_terminate: float = 0.25     # base_z 低于此值判塌倒
    action_scale: float = 0.25    # 旧：q_des = q_default + action×0.25
    residual_scale: float = 0.2   # 残差 RL：q_des = q_default + gait + action×0.2
    use_residual: bool = True     # True=残差模式（基准步态+残差），False=旧模式
    # 基准步态实现："sine"=v24 正弦相位（仅前进构型稳定）；"ik"=v25 IK 全向
    # 步进（支撑相匀速后移 + 摆动相抬脚回位，覆盖后退/横移/转向）
    gait_mode: str = "sine"
    # 指令分布（默认末期全范围）
    cmd_vx: tuple = (-0.5, 1.5)
    cmd_vy: tuple = (-0.5, 0.5)
    cmd_yaw: tuple = (-2.0, 2.0)
    zero_cmd_prob: float = 0.17            # 零指令概率（15~20% 取中）
    cmd_resample_steps: tuple = (100, 200)  # 2~4s @50Hz，episode 内重采样
    # 奖励权重（决策 5.3）
    w_track: float = 1.0
    w_yaw: float = 0.5
    w_upright: float = 0.5
    w_energy: float = 0.05
    w_jerk: float = 0.01
    w_vz: float = 1.0
    w_rp_rate: float = 0.05
    track_sigma2: float = 0.25      # 线速度跟踪高斯分母
    # yaw 跟踪高斯分母（2026-10-24 从 track_sigma2 拆出，原共用一个 σ=0.5 rad/s）。
    # 实测航向漂移只有 0.063 rad/s，落在 exp(-0.063²/0.25)=0.984 里只损失 0.008 分，
    # 策略没有动力修；收紧到 0.10（σ≈0.32）才压得住。
    yaw_sigma2: float = 0.25
    # >0 时 yaw 奖励改用**线性**形式 w_yaw*clip(1-|e|/yaw_lin_scale, 0, 1)。
    # 动机（2026-10-24 实测）：高斯在零点附近是平的，实测航向漂移 0.063 rad/s
    # 在 σ²=0.25 下只损失 0.008 分、σ²=0.10 也只损失 0.019 分 → 策略没有动力修。
    # 线性项梯度恒定（-w/scale），1.0 rad/s 量程下每 rad/s 扣 0.5 分，压得住漂移。
    yaw_lin_scale: float = 0.0
    upright_thresh: float = 0.39    # |g_x|,|g_y| < sin(0.4rad)，与旧阈值等价
    # 域随机化
    dr_enabled: bool = True
    # 模型级 DR 池数 K：1=整批 env 共享一份（最快，无 DR 多样度）；K>1=外层 K 份
    # 池模型、池内 n_envs/K 个 env 共享（DR 多样度=K）。实测 2048 env 下
    # K∈{128,256,512} 仍走快路径（~1.2x 开销），K∈{8,32,64,2048} 掉慢路径（见
    # scripts/bench_mjx_throughput.py --modes kpool）
    dr_pools: int = 1
    # 足底 geom friction[0] 绝对值（4 只脚同一采样值）。足底 priority=1：
    # 接触参数完全取 priority 高的一方，地板摩擦不参与——DR 必须打在足底上
    friction_range: tuple = (0.5, 1.8)
    base_mass_range: tuple = (0.8, 1.2)    # base_link 质量乘性
    kp_scale_range: tuple = (0.8, 1.2)
    kv_scale_range: tuple = (0.8, 1.2)
    init_base_z: tuple = (0.30, 0.34)
    init_rpy_noise: float = 0.1            # rad
    init_joint_noise: float = 0.05         # rad
    init_vel_noise: float = 0.1            # m/s 与 rad/s
    init_jointvel_noise: float = 0.5       # rad/s
    action_delay_range: tuple = (0, 2)     # 策略步
    obs_noise_scale: float = 1.0           # 0 = 关观测噪声
    gravity_noise: float = 0.03            # projected gravity 分量噪声(0.02~0.05)


class DRParams(NamedTuple):
    """模型级 DR 的 per-env 参数（episode 内恒定，reset 时重采样）"""
    fric: jax.Array        # 足底 geom friction[0]（4 只脚同一值）
    mass_scale: jax.Array  # base_link 质量乘子
    kp_scale: jax.Array
    kv_scale: jax.Array


class EnvState(NamedTuple):
    """单个 env 的完整状态（vmap 后第 0 维为 env 批次轴）"""
    dx: mjx.Data
    last_action: jax.Array   # (12,)
    command: jax.Array       # (3,) [vx, vy, yaw_rate]
    delay_buf: jax.Array     # (3,12) 动作延迟队列，尾部最新
    delay: jax.Array         # int32 ∈ [0,2]
    resample_in: jax.Array   # int32 距下次指令重采样剩余策略步
    episode_steps: jax.Array
    gait_phase: jax.Array    # 标量 ∈[0,1) 基准步态周期相位（累加，指令变化不跳变）
    dr: DRParams
    rng: jax.Array


# ---------------------------------------------------------------------------
# 小工具（纯函数，可 jit）
# ---------------------------------------------------------------------------
def _quat_to_mat(q: jax.Array) -> jax.Array:
    """wxyz 四元数 → 旋转矩阵 R（body→world）。与 xmat 对拍误差 5.6e-17。"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rpy_to_quat(rpy: jax.Array) -> jax.Array:
    """[roll,pitch,yaw] → wxyz 四元数（reset 初态扰动用）"""
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = jnp.cos(r * 0.5), jnp.sin(r * 0.5)
    cp, sp = jnp.cos(p * 0.5), jnp.sin(p * 0.5)
    cy, sy = jnp.cos(y * 0.5), jnp.sin(y * 0.5)
    return jnp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def tree_where_done(done: jax.Array, fresh, current):
    """按 env 在两个同构 pytree（批次轴在第 0 维）之间二选一（auto-reset 用）"""
    def pick(f, c):
        return jnp.where(done.reshape((-1,) + (1,) * (f.ndim - 1)), f, c)
    return jax.tree_util.tree_map(pick, fresh, current)


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
class DogEnvMJX:
    """MJX 向量化机器狗环境：reset/step 均为 jit+vmap 纯函数"""

    # 加电站立位（2026-10-23 kp=150 实测标定；替换原 [0,0.9,-1.8]×4，与 keyframe home 一致）
    Q_DEFAULT = np.array([-0.0134, 0.8611, -1.7340, 0.0160, 0.8531, -1.7100,
                          0.0088, 0.8801, -1.7210, -0.0133, 0.9111, -1.6946])
    OBS_DIM = 61
    ACT_DIM = 12

    def __init__(self, n_envs: int, config: EnvConfig = None,
                 scene_path: str = None):
        self.n_envs = int(n_envs)
        self.cfg = config or EnvConfig()
        self.n_pools = int(self.cfg.dr_pools)
        assert self.n_pools >= 1 and self.n_envs % self.n_pools == 0, \
            f"dr_pools={self.n_pools} 必须整除 n_envs={self.n_envs}"
        if scene_path is None:
            scene_path = str(Path(__file__).parent / "scene_mjx.xml")

        m = mujoco.MjModel.from_xml_path(scene_path)
        assert m.nq == 19 and m.nv == 18 and m.nu == 12, \
            f"模型维度异常 nq={m.nq} nv={m.nv} nu={m.nu}"
        home_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert home_id >= 0, "keyframe 'home' 缺失"
        self._base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        assert self._base_id >= 0
        # 4 只足底 geom id（摩擦 DR 目标；足底 priority=1，地板摩擦不参与接触对，
        # 改 floor.geom_friction 无效——2026-10-23 代码审查实锤后修复）
        foot_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        ]
        assert all(i >= 0 for i in foot_ids), "foot geoms (FL/FR/RL/RR_foot) 缺失"
        self._foot_ids = jnp.asarray(foot_ids)

        # 4 条腿 calf body id（用于几何法 contact 检测）
        # foot geom 在 calf body 下，偏移 [0, 0, -0.24]
        self._calf_ids = jnp.asarray([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FL_calf"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FR_calf"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "RL_calf"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "RR_calf"),
        ])
        self._foot_offset = jnp.array([-0.002, 0.0, -0.24])  # 与 XML foot pos 一致
        self._contact_thresh = jnp.float32(0.05)  # 脚 z < 此值判接触

        # 模型级 DR 的标称值（_apply_dr 以此恢复再乘/设）
        self._nom_base_mass = jnp.float32(m.body_mass[self._base_id])
        self._nom_foot_fric = jnp.float32(m.geom_friction[foot_ids[0], 0])
        self._nom_gain0 = jnp.asarray(m.actuator_gainprm[:, 0].copy())   # kp
        self._nom_bias1 = jnp.asarray(m.actuator_biasprm[:, 1].copy())   # -kp
        self._nom_bias2 = jnp.asarray(m.actuator_biasprm[:, 2].copy())   # -kv
        self._jnt_lo = jnp.asarray(m.jnt_range[1:, 0].copy())
        self._jnt_hi = jnp.asarray(m.jnt_range[1:, 1].copy())

        # 基准状态：keyframe home + mj_forward → put_data（mjx.make_data 对本模型
        # condim=6 接触对报 NotImplementedError，必须走 put_data 路径）
        d0 = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d0, home_id)
        mujoco.mj_forward(m, d0)
        self._dx_base = mjx.put_data(m, d0)
        self._mx = mjx.put_model(m)

        self._q_default = jnp.asarray(self.Q_DEFAULT)
        cfg = self.cfg
        self._obs_noise_std = jnp.asarray(np.concatenate([
            [0.01],                            #  0    base_z
            np.full(3, cfg.gravity_noise),     #  1:4  projected gravity
            np.full(3, 0.10),                  #  4:7  机身系线速度
            np.full(3, 0.05),                  #  7:10 机身系角速度
            np.full(12, 0.01),                 # 10:22 关节位置
            np.full(12, 0.15),                 # 22:34 关节速度
            np.zeros(12),                      # 34:46 last_action（自身已知量）
            np.zeros(12),                      # 46:58 q_default（常量）
            np.zeros(3),                       # 58:61 指令（自身已知量）
        ]).astype(np.float32))

        # reset/step 的 jit 目标内部按 n_pools 静态分支：K=1 → 单层 vmap + 标量
        # DR；K>1 → 外层 K 池嵌套 vmap（池内共享一份池模型）
        self._reset_batch = jax.jit(self._reset_impl)
        self._step_batch = jax.jit(self._step_impl)
        # 当前 rollout 的模型级 DR（脚本/单测走这两个默认值；训练脚本每 update
        # 显式 resample/传参，避免闭包常量随 update 变化触发 mega-jit 重编译）
        self.resample_dr(jax.random.PRNGKey(0))

    # --------------------------------------------------------------
    # 模型级 DR（vmap 外层，标量 K=1 或 (K,) 池化）——见文件头「关键差异 6」
    # --------------------------------------------------------------
    def _sample_one_dr(self, key: jax.Array) -> DRParams:
        """采样**一份标量**模型级 DR（池内共享，整 rollout 恒定）"""
        cfg = self.cfg
        if not cfg.dr_enabled:
            return DRParams(fric=self._nom_foot_fric, mass_scale=jnp.float32(1.0),
                            kp_scale=jnp.float32(1.0), kv_scale=jnp.float32(1.0))
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return DRParams(
            fric=jax.random.uniform(k1, (), minval=cfg.friction_range[0],
                                    maxval=cfg.friction_range[1]),
            mass_scale=jax.random.uniform(k2, (), minval=cfg.base_mass_range[0],
                                          maxval=cfg.base_mass_range[1]),
            kp_scale=jax.random.uniform(k3, (), minval=cfg.kp_scale_range[0],
                                        maxval=cfg.kp_scale_range[1]),
            kv_scale=jax.random.uniform(k4, (), minval=cfg.kv_scale_range[0],
                                        maxval=cfg.kv_scale_range[1]),
        )

    def sample_model_dr(self, key: jax.Array, k: int = None) -> DRParams:
        """采样 K=n_pools 份模型级 DR（跨 update 覆盖 DR 分布，整 rollout 恒定）

        K=1 → 标量叶子（非批化模型，用 in_axes=None 广播）；
        K>1 → 叶子形状 (K,)，每份供一个池使用（见 build_model / _reset_impl）
        """
        k = self.n_pools if k is None else int(k)
        if k == 1:
            return self._sample_one_dr(key)
        return jax.vmap(self._sample_one_dr)(jax.random.split(key, k))

    def build_model(self, dr_model: DRParams) -> mjx.Model:
        """把模型级 DR 注入共享模型 → 供 reset/step 使用的模型

        n_pools=1 → 标量 DR → **非批化单模型**（in_axes=None 广播给整批 env）；
        n_pools>1 → (K,) DR → vmap 出**按池批化**的 (K,...) 模型，配合
        _step_impl 的嵌套 vmap（外层 K 池、池内共享一份池模型）。
        实测 2048 env：K∈{128,256,512} 仍是快路径（~1.2x），K=1 最快。
        """
        if self.n_pools == 1:
            return self._mx if not self.cfg.dr_enabled \
                else self._apply_dr(self._mx, dr_model)
        return jax.vmap(self._apply_dr, in_axes=(None, 0))(self._mx, dr_model)

    def resample_dr(self, key: jax.Array) -> DRParams:
        """重采样本 rollout 的模型级 DR（脚本/单测便捷入口，会改 env 属性）"""
        self._dr_model = self.sample_model_dr(key)
        self._mx_eff = self.build_model(self._dr_model)
        return self._dr_model

    # --------------------------------------------------------------
    # 池化：外部 state/obs 始终 (N,)；内部 K>1 时才 reshape 成 (K, N/K)
    # --------------------------------------------------------------
    def _to_pool(self, x):
        """(N, ...) → (K, N/K, ...)"""
        k = self.n_pools
        return jax.tree_util.tree_map(
            lambda a: a.reshape((k, self.n_envs // k) + a.shape[1:]), x)

    def _from_pool(self, x):
        """(K, N/K, ...) → (N, ...)"""
        return jax.tree_util.tree_map(
            lambda a: a.reshape((self.n_envs,) + a.shape[2:]), x)

    def _reset_impl(self, keys: jax.Array, dr: DRParams):
        """jit 目标：K=1 → 单层 vmap 广播标量 DR；K>1 → 外层 K 池、池内共享"""
        if self.n_pools == 1:
            return jax.vmap(self._reset_one, in_axes=(0, None))(keys, dr)
        k = self.n_pools
        kk = keys.reshape((k, self.n_envs // k) + keys.shape[1:])
        state, obs = jax.vmap(
            jax.vmap(self._reset_one, in_axes=(0, None)),
            in_axes=(0, 0))(kk, dr)
        return self._from_pool(state), self._from_pool(obs)

    def _step_impl(self, mx: mjx.Model, state: EnvState, action: jax.Array):
        """jit 目标：K=1 → 单层 vmap 广播单模型；K>1 → 外层 K 池、池内共享"""
        if self.n_pools == 1:
            return jax.vmap(self._step_one, in_axes=(None, 0, 0))(
                mx, state, action)
        out = jax.vmap(
            jax.vmap(self._step_one, in_axes=(None, 0, 0)),
            in_axes=(0, 0, 0))(mx, self._to_pool(state), self._to_pool(action))
        return self._from_pool(out)

    # --------------------------------------------------------------
    # 对外 API（rng 单 key 传入，内部分拆到 n_envs）
    # --------------------------------------------------------------
    def reset(self, rng: jax.Array, dr_model: DRParams = None):
        """→ (state, obs)，叶子形状 (n_envs, ...)；dr_model 缺省用 self._dr_model"""
        dr_model = self._dr_model if dr_model is None else dr_model
        return self._reset_batch(jax.random.split(rng, self.n_envs), dr_model)

    def step(self, state: EnvState, action: jax.Array, mx_model: mjx.Model = None):
        """action (n_envs,12) → (state, obs, reward, terminated, truncated, info)

        mx_model 缺省用 self._mx_eff；训练时由 mega-jit 内的上层传入（见
        ppo_jax.make_update），保证模型走显式参数而非闭包常量（否则随 update 变化
        的模型会触发 mega-jit 重编译）。K>1 时为按池批化的 (K,...) 模型。
        """
        mx_model = self._mx_eff if mx_model is None else mx_model
        return self._step_batch(mx_model, state, action)

    def get_privileged_obs(self, state):
        """批量获取特权信息 (n_envs, 38)，供教师网络使用"""
        return jax.vmap(self._get_privileged_obs)(state.dx, state.dr)

    def dr_per_env(self, dr_model: DRParams) -> DRParams:
        """把模型级 DR（标量 / (K,) 池）展开成 per-env (n_envs,) 叶子

        只用于特权观测与日志；真正进物理的是 build_model 的批化模型。
        """
        if self.n_pools == 1:
            return jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (self.n_envs,)), dr_model)
        rep = self.n_envs // self.n_pools
        return jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, rep, axis=0), dr_model)

    def sync_state_dr(self, state: EnvState, dr_model: DRParams) -> EnvState:
        """把 state.dr 同步为当前模型级 DR（每 update 调用一次）

        物理模型每个 update 都会重建，而 state 里的 dr 只在 reset 时写入。
        特权观测从 state.dr 读取 DR 真值，不同步就会长期使用过期参数。
        """
        return state._replace(dr=self.dr_per_env(dr_model))

    # --------------------------------------------------------------
    # per-env 纯函数（vmap 映射目标）
    # --------------------------------------------------------------
    def _sample_command(self, key) -> jax.Array:
        cfg = self.cfg
        k1, k2, k3, k4 = jax.random.split(key, 4)
        cmd = jnp.array([
            jax.random.uniform(k1, (), minval=cfg.cmd_vx[0], maxval=cfg.cmd_vx[1]),
            jax.random.uniform(k2, (), minval=cfg.cmd_vy[0], maxval=cfg.cmd_vy[1]),
            jax.random.uniform(k3, (), minval=cfg.cmd_yaw[0], maxval=cfg.cmd_yaw[1]),
        ])
        zero = jax.random.bernoulli(k4, cfg.zero_cmd_prob)
        return jnp.where(zero, jnp.zeros(3), cmd)

    def _reset_one(self, key, dr: DRParams):
        cfg = self.cfg
        k_init, k_cmd, k_delay, k_int, k_noise, k_phase = jax.random.split(key, 6)

        # 模型级 DR 由调用方在 vmap 外层采样后广播进来（build_model 用它注入共享
        # 模型）；state.dr 只保留这份标量供特权观测使用，不再参与模型构造
        if cfg.dr_enabled:
            delay = jax.random.randint(k_delay, (), cfg.action_delay_range[0],
                                       cfg.action_delay_range[1] + 1)
        else:
            delay = jnp.zeros((), jnp.int32)

        # 初态扰动（在 keyframe home 基准上叠加）
        dx = self._dx_base
        if cfg.dr_enabled:
            k1, k2, k3, k4, k5, k6 = jax.random.split(k_init, 6)
            qpos0 = dx.qpos
            z = jax.random.uniform(k1, (), minval=cfg.init_base_z[0],
                                   maxval=cfg.init_base_z[1])
            quat = _rpy_to_quat(jax.random.uniform(
                k2, (3,), minval=-cfg.init_rpy_noise, maxval=cfg.init_rpy_noise))
            jq = jnp.clip(
                qpos0[7:] + jax.random.uniform(
                    k3, (12,), minval=-cfg.init_joint_noise,
                    maxval=cfg.init_joint_noise),
                self._jnt_lo, self._jnt_hi)
            qpos = jnp.concatenate([qpos0[0:2], z[None], quat, jq])
            qvel = jnp.concatenate([
                jax.random.uniform(k4, (3,), minval=-cfg.init_vel_noise,
                                   maxval=cfg.init_vel_noise),
                jax.random.uniform(k5, (3,), minval=-cfg.init_vel_noise,
                                   maxval=cfg.init_vel_noise),
                jax.random.uniform(k6, (12,), minval=-cfg.init_jointvel_noise,
                                   maxval=cfg.init_jointvel_noise),
            ])
            dx = dx.replace(qpos=qpos, qvel=qvel)

        lo, hi = cfg.cmd_resample_steps
        state = EnvState(
            dx=dx,
            # 必须强类型（weak_type=False）：jnp.zeros 产出的是弱类型 float 数组，而
            # 一次 env.step 之后 last_action 由网络输出算出、变成强类型，jit 缓存键
            # （含 avals.weak_type）随之改变 → mega-jit 多编译一次（实测 128 env
            # update 0/1 = 18.2s/17.3s，之后才落到 6.1s）
            last_action=jnp.array(np.zeros(12, dtype=np.float32)),
            command=self._sample_command(k_cmd),
            delay_buf=jnp.repeat(self._q_default[None, :], 3, axis=0),
            delay=delay,
            resample_in=jax.random.randint(k_int, (), lo, hi + 1),
            episode_steps=jnp.zeros((), jnp.int32),
            # 相位随机初始化：让策略见过任意起始相位，避免只适应"从 0 开始"
            gait_phase=jax.random.uniform(k_phase, (), minval=0.0, maxval=1.0),
            dr=dr,
            rng=k_noise,
        )
        obs = self._get_obs(dx, state.last_action, state.command, k_noise)
        return state, obs

    def _apply_dr(self, mx: mjx.Model, dr: DRParams) -> mjx.Model:
        """把**标量**模型级 DR 注入共享模型（只在 vmap 外层调用，见 build_model）

        必须在 vmap 外调用：一旦在 vmap 内按 lane 构造模型（旧实现），XLA 会把
        模型当成逐 env 变化的量，mjx.step 快路径失效（实测慢 31 倍，见文件头）。
        """
        gf = mx.geom_friction.at[self._foot_ids, 0].set(dr.fric)
        bm = mx.body_mass.at[self._base_id].set(self._nom_base_mass * dr.mass_scale)
        ag = mx.actuator_gainprm.at[:, 0].set(self._nom_gain0 * dr.kp_scale)
        ab = (mx.actuator_biasprm.at[:, 1].set(self._nom_bias1 * dr.kp_scale)
                                   .at[:, 2].set(self._nom_bias2 * dr.kv_scale))
        return mx.replace(geom_friction=gf, body_mass=bm,
                          actuator_gainprm=ag, actuator_biasprm=ab)

    def _get_obs(self, dx: mjx.Data, last_action: jax.Array,
                 command: jax.Array, noise_key) -> jax.Array:
        cfg = self.cfg
        R = _quat_to_mat(dx.qpos[3:7])
        g_b = -R[2, :]                     # Rᵀ·[0,0,-1]
        lin_b = R.T @ dx.qvel[0:3]         # 机身系线速度
        ang_b = dx.qvel[3:6]               # 机身系角速度（freejoint 本就机身系）
        obs = jnp.concatenate([
            dx.qpos[2:3],                  #  0   base_z
            g_b,                           #  1:4
            lin_b,                         #  4:7
            ang_b,                         #  7:10
            dx.qpos[7:],                   # 10:22
            dx.qvel[6:],                   # 22:34
            last_action,                   # 34:46
            self._q_default,               # 46:58
            command,                       # 58:61
        ])
        if cfg.dr_enabled and cfg.obs_noise_scale > 0:
            obs = obs + self._obs_noise_std * cfg.obs_noise_scale \
                        * jax.random.normal(noise_key, (self.OBS_DIM,))
        return obs

    PRIV_DIM = 38

    def _get_privileged_obs(self, dx: mjx.Data, dr: DRParams) -> jax.Array:
        """特权信息 38 维（教师网络专用，实机不可得）

        布局：
            [ 0: 3] 精确 base 线速度（机身系）= Rᵀ·qvel[0:3]
            [ 3: 6] 精确 base 角速度（机身系）= qvel[3:6]
            [ 6:18] 精确关节位置 = qpos[7:]
            [18:30] 精确关节速度 = qvel[6:]
            [30:34] contact[4]（脚是否着地，几何法：foot_z < 阈值）
            [34]    足底摩擦系数
            [35]    机身质量乘子
            [36:38] base x/y 世界位置
        """
        R = _quat_to_mat(dx.qpos[3:7])
        lin_b = R.T @ dx.qvel[0:3]
        ang_b = dx.qvel[3:6]

        # 几何法 contact：4 只脚的世界 z 坐标
        # foot_world = calf_xpos + calf_xmat @ foot_offset
        calf_xpos = dx.xpos[self._calf_ids]        # (4, 3)
        calf_xmat = dx.xmat[self._calf_ids].reshape(4, 3, 3)  # (4, 3, 3)
        foot_z = calf_xpos[:, 2] + (calf_xmat @ self._foot_offset)[:, 2]
        contact = (foot_z < self._contact_thresh).astype(jnp.float32)  # (4,)

        return jnp.concatenate([
            lin_b,               #  0: 3
            ang_b,               #  3: 6
            dx.qpos[7:],         #  6:18
            dx.qvel[6:],         # 18:30
            contact,             # 30:34
            dr.fric[None],       # 34
            dr.mass_scale[None], # 35
            dx.qpos[0:2],        # 36:38
        ])

    def _step_one(self, mx: mjx.Model, state: EnvState, action: jax.Array):
        cfg = self.cfg
        # mx 由 vmap 外层构建好（build_model），此处不做任何 per-lane 模型改动

        # 动作 → q_des → 延迟队列（delay 步后生效，0 步即即时）
        action = jnp.clip(action, -1.0, 1.0)
        if cfg.use_residual:
            # 残差模式：q_des = q_default + 基准步态 + action × residual_scale
            base_gait = (gait_ik_phase_fn(state.gait_phase, state.command)
                         if cfg.gait_mode == "ik"
                         else gait_fn(state.episode_steps, state.command))
            q_des = self._q_default + base_gait + action * cfg.residual_scale
        else:
            # 旧模式：q_des = q_default + action × action_scale
            q_des = self._q_default + action * cfg.action_scale
        buf = jnp.roll(state.delay_buf, -1, axis=0).at[-1].set(q_des)
        ctrl = buf[2 - state.delay]

        # 10 个物理子步 = 1 个策略步（无中途塌倒早停，见模块 docstring）
        # v21：qvel 数值护栏。每个物理子步后 clamp（±100），极端接触/别腿时
        # 阻止速度发散滚雪球成 inf/NaN；正常步态 |qvel|<30，护栏隐身。
        def _substep(d, _):
            d = mjx.step(mx, d)
            # v21b：护栏触火取证——任何分量触及 ±100 记一次（NaN 比较恒为
            # False，不会误计）。触火率与 alive 下探同频 = 接触爆炸实锤。
            fired = jnp.any(jnp.abs(d.qvel) >= 100.0)
            return d.replace(qvel=jnp.clip(d.qvel, -100.0, 100.0)), fired
        dx = state.dx.replace(ctrl=ctrl)
        dx, fired_arr = jax.lax.scan(_substep, dx, None,
                                     length=cfg.frame_skip)
        qvel_fire_cnt = jnp.sum(fired_arr.astype(jnp.float32))

        # 运动学（观测/奖励共用）
        R = _quat_to_mat(dx.qpos[3:7])
        g_b = -R[2, :]
        lin_b = R.T @ dx.qvel[0:3]
        ang_b = dx.qvel[3:6]
        base_z = dx.qpos[2]

        # 奖励（决策 5.3：速度跟踪主导 + 姿态/能耗/平滑小权重）
        cmd = state.command   # 本步追踪的是进入本步时的指令
        vx_err = lin_b[0] - cmd[0]
        vy_err = lin_b[1] - cmd[1]
        yaw_err = ang_b[2] - cmd[2]
        r_track = cfg.w_track * jnp.exp(-(vx_err ** 2 + vy_err ** 2)
                                        / cfg.track_sigma2)
        if cfg.yaw_lin_scale > 0:      # 静态分支，trace 期求值，不影响 jit
            r_yaw = cfg.w_yaw * jnp.clip(
                1.0 - jnp.abs(yaw_err) / cfg.yaw_lin_scale, 0.0, 1.0)
        else:
            r_yaw = cfg.w_yaw * jnp.exp(-(yaw_err ** 2) / cfg.yaw_sigma2)
        upright = ((jnp.abs(g_b[0]) < cfg.upright_thresh)
                   & (jnp.abs(g_b[1]) < cfg.upright_thresh))
        r_upright = cfg.w_upright * upright.astype(jnp.float32)
        r_energy = -cfg.w_energy * jnp.sum(action ** 2)
        r_jerk = -cfg.w_jerk * jnp.sum((action - state.last_action) ** 2)
        r_vz = -cfg.w_vz * lin_b[2] ** 2
        r_rp = -cfg.w_rp_rate * (ang_b[0] ** 2 + ang_b[1] ** 2)
        reward = r_track + r_yaw + r_upright + r_energy + r_jerk + r_vz + r_rp

        steps = state.episode_steps + 1
        # 基准步态相位累加（指令变化只改推进速度，相位本身连续）
        # 策略步长 = XML timestep × frame_skip = 0.002 × 10 = 0.02 s
        gait_phase = (state.gait_phase
                      + cfg.frame_skip * 0.002 * gait_freq_fn(state.command)) % 1.0
        terminated = base_z < cfg.z_terminate
        truncated = steps >= cfg.max_steps

        # 指令重采样（reward 之后；新指令进 obs[58:61] 供下一步决策）
        rng, k_cmd, k_int, k_noise = jax.random.split(state.rng, 4)
        resample_in = state.resample_in - 1
        do = resample_in <= 0
        lo, hi = cfg.cmd_resample_steps
        command = jnp.where(do, self._sample_command(k_cmd), cmd)
        resample_in = jnp.where(do, jax.random.randint(k_int, (), lo, hi + 1),
                                resample_in)

        new_state = EnvState(
            dx=dx, last_action=action, command=command, delay_buf=buf,
            delay=state.delay, resample_in=resample_in, episode_steps=steps,
            gait_phase=gait_phase,
            dr=state.dr, rng=rng,
        )
        obs = self._get_obs(dx, action, command, k_noise)
        priv = self._get_privileged_obs(dx, state.dr)
        info = {
            "base_z": base_z,
            "lin_vel_b": lin_b, "ang_vel_b": ang_b, "command": cmd,
            "r_track": r_track, "r_yaw": r_yaw, "r_upright": r_upright,
            "r_energy": r_energy, "r_jerk": r_jerk, "r_vz": r_vz, "r_rp": r_rp,
            "qvel_fire_cnt": qvel_fire_cnt,
            "privileged_obs": priv,
        }
        return new_state, obs, reward, terminated, truncated, info
