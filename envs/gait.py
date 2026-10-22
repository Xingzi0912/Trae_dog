"""正弦 trot 基准步态生成器（Residual RL 用）

trot 步态：对角腿同步（FL+RR 一组，FR+RL 一组），两组相位差 π。

最终 q_des = q_default + gait(phase, command)
网络只学残差：q_des = q_default + gait + action × residual_scale

步态参数（站立位 [0, 0.9, -1.8] 基础上摆动）：
    - hip:   A=0.10 rad，vy/yaw 静态偏置（横移/转向重心调整）
    - thigh: A=0.35 rad，vx 驱动前后摆（主要推进关节）
    - calf:  A=0.35 rad，与 thigh 反相屈膝
频率：由 vx 决定，vx=0 时 2.5Hz，vx=1.5 时 4Hz

v24 方向语义修复（2026-10-22 探针判决，/tmp/probe_gait.py）：
    旧版 thigh/calf 为对称正弦（无方向偏置项，注释"vx 正向后蹬"与实现
    不符），净推进方向由摆动-触地时序的时间不对称决定——实测 cmd=+1 时
    200 步净位移 -4.5m（后退！）、cmd=0 漂移 -1.16m、cmd=-1 直接摔倒。
    vx 只进频率/幅值不进方向，定向推进全靠残差（±0.2 rad 对抗整个步态），
    是 track 卡 0.46-0.49 平台与"前进指令狗倒退"的根因。修复：
    ① 相位翻转：thigh/calf 整体乘 dir 系数（保持两者 π 反相不变），
      dir=-1（vx≥0）时净推进 +x；探针旧输出对应 dir=+1。
    ② 后退语义：vx<0 时 dir=+1（回到旧相位 = -x 推进），负指令真后退。
    ③ 幅值速度门控：amp_thigh/amp_calf ∝ |vx|，删除 thigh 最小摆动
      （A_THIGH×0.3）与 calf 恒定全幅——零指令 q_des=q_default 纯站立，
      修复"零速腿抖"（calf ±0.35 rad 全幅 2Hz 摆动，残差压不干净）。
      vx 过零时幅值→0，方向切换天然连续。
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

    # 幅值调制（v24：∝|vx| 速度门控，零指令幅值=0 纯站立）
    amp_thigh = A_THIGH * jnp.clip(jnp.abs(vx) / 1.0, 0.0, 1.0)
    amp_calf = A_CALF * jnp.clip(jnp.abs(vx) / 1.0, 0.0, 1.0)
    # vy 影响 hip 左右摆
    amp_hip = A_HIP * jnp.clip(jnp.abs(vy) / 0.5, 0.0, 1.0)
    # yaw 影响左右腿 hip 差（转向时内外侧步幅不同）
    yaw_hip = A_HIP * 0.5 * jnp.clip(yaw / 2.0, -1.0, 1.0)

    # 方向系数（v24）：dir=-1 → thigh=-sin（净推进 +x，探针旧版 +sin 朝 -x）；
    # vx<0 → dir=+1（旧相位，净推进 -x = 后退）
    direction = jnp.where(vx >= 0.0, -1.0, 1.0)

    # hip: vy 正向左移 → 左腿外展(hip+)，右腿内收(hip-)
    hip_sign = jnp.sign(vy + 1e-8)
    fl_hip = amp_hip * hip_sign + yaw_hip
    fr_hip = -amp_hip * hip_sign - yaw_hip
    rl_hip = amp_hip * hip_sign - yaw_hip
    rr_hip = -amp_hip * hip_sign + yaw_hip

    # thigh/calf：整体方向翻转，两者保持 π 反相（屈膝配合不变）
    fl_thigh = direction * amp_thigh * jnp.sin(phase_a)
    fr_thigh = direction * amp_thigh * jnp.sin(phase_b)
    rl_thigh = direction * amp_thigh * jnp.sin(phase_b)
    rr_thigh = direction * amp_thigh * jnp.sin(phase_a)

    fl_calf = -direction * amp_calf * jnp.sin(phase_a)
    fr_calf = -direction * amp_calf * jnp.sin(phase_b)
    rl_calf = -direction * amp_calf * jnp.sin(phase_b)
    rr_calf = -direction * amp_calf * jnp.sin(phase_a)

    gait_q = jnp.stack([
        fl_hip, fl_thigh, fl_calf,   # FL
        fr_hip, fr_thigh, fr_calf,   # FR
        rl_hip, rl_thigh, rl_calf,   # RL
        rr_hip, rr_thigh, rr_calf,   # RR
    ], axis=-1)

    return gait_q
