"""正弦 trot 基准步态生成器（Residual RL 用）

trot 步态：对角腿同步（FL+RR 一组，FR+RL 一组），两组相位差 π。

最终 q_des = q_default + gait(phase, command)
网络只学残差：q_des = q_default + gait + action × residual_scale

步态参数（站立位 [0, 0.9, -1.8] 基础上摆动）：
    - hip:   A=0.10 rad，vy 驱动左右摆
    - thigh: A=0.35 rad，vx 驱动前后摆（主要推进关节）
    - calf:  A=0.35 rad，与 thigh 配合屈膝
频率：由 vx 决定，vx=0 时 2Hz，vx=1.5 时 4Hz
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

Q_DEFAULT = np.array([0.0, 0.9, -1.8] * 4)
DT = 0.02  # 策略步长

# 步态幅值（rad）
A_HIP = 0.10
A_THIGH = 0.35
A_CALF = 0.35

# 频率范围（Hz）
FREQ_MIN = 2.0
FREQ_MAX = 4.0


def gait(episode_steps: jax.Array, command: jax.Array) -> jax.Array:
    """生成基准 trot 步态

    Args:
        episode_steps: (N,) 策略步数（标量或批次）
        command: (N, 3) [vx, vy, yaw_rate]

    Returns:
        (N, 12) 基准关节角增量（相对 q_default）
    """
    # 频率由 vx 决定：vx ∈ [-0.5, 1.5] → freq ∈ [2, 4] Hz
    vx = command[..., 0]
    vy = command[..., 1]
    yaw = command[..., 2]
    t = jnp.clip((vx + 0.5) / 2.0, 0.0, 1.0)  # 归一化到 [0,1]
    freq = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * t

    # 相位
    phase = (episode_steps * DT * freq * 2 * jnp.pi) % (2 * jnp.pi)

    # 对角腿相位差 π
    phase_a = phase               # FL, RR
    phase_b = phase + jnp.pi      # FR, RL

    # 幅值调制
    # vx 影响 thigh 前后摆幅值
    amp_thigh = A_THIGH * jnp.clip(jnp.abs(vx) / 1.0, 0.0, 1.0)
    amp_thigh = jnp.maximum(amp_thigh, A_THIGH * 0.3)  # 最小摆动
    # vy 影响 hip 左右摆
    amp_hip = A_HIP * jnp.clip(jnp.abs(vy) / 0.5, 0.0, 1.0)
    # yaw 影响左右腿 hip 差（转向时内外侧步幅不同）
    yaw_hip = A_HIP * 0.5 * jnp.clip(yaw / 2.0, -1.0, 1.0)

    # 每条腿的关节增量
    # FL (phase_a): hip +0, thigh sin, calf sin
    # FR (phase_b): hip -0, thigh sin, calf sin
    # RL (phase_b): hip +0, thigh sin, calf sin
    # RR (phase_a): hip -0, thigh sin, calf sin

    # hip: vy 正向左移 → 左腿外展(hip+)，右腿内收(hip-)
    hip_sign = jnp.sign(vy + 1e-8)
    fl_hip = amp_hip * hip_sign + yaw_hip
    fr_hip = -amp_hip * hip_sign - yaw_hip
    rl_hip = amp_hip * hip_sign - yaw_hip
    rr_hip = -amp_hip * hip_sign + yaw_hip

    # thigh: 前后摆，vx 正向后蹬(thigh-) 与前摆(thigh+)
    fl_thigh = amp_thigh * jnp.sin(phase_a)
    fr_thigh = amp_thigh * jnp.sin(phase_b)
    rl_thigh = amp_thigh * jnp.sin(phase_b)
    rr_thigh = amp_thigh * jnp.sin(phase_a)

    # calf: 与 thigh 反相，屈膝
    fl_calf = -A_CALF * jnp.sin(phase_a)
    fr_calf = -A_CALF * jnp.sin(phase_b)
    rl_calf = -A_CALF * jnp.sin(phase_b)
    rr_calf = -A_CALF * jnp.sin(phase_a)

    gait_q = jnp.stack([
        fl_hip, fl_thigh, fl_calf,   # FL
        fr_hip, fr_thigh, fr_calf,   # FR
        rl_hip, rl_thigh, rl_calf,   # RL
        rr_hip, rr_thigh, rr_calf,   # RR
    ], axis=-1)

    return gait_q
