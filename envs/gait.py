"""正弦 trot 基准步态生成器（Residual RL 用）

trot 步态：对角腿同步（FL+RR 一组，FR+RL 一组），两组相位差 π。

最终 q_des = q_default + gait(phase, command)
网络只学残差：q_des = q_default + gait + action × residual_scale

步态参数（站立位 [0, 0.9, -1.8] 基础上摆动）：
    - hip:   A=0.10 rad，vy/yaw 静态偏置（横移/转向重心调整）
    - thigh: A=0.35 rad，vx 驱动前后摆（主要推进关节）
    - calf:  A=0.35 rad，与 thigh 的相位差 δ 决定推进方向
频率：由 |vx| 决定，2~4Hz

v24 方向语义修复（2026-10-22 探针判决）：
    旧版 thigh/calf 为对称正弦，净推进方向由摆动-触地时序的不对称决定——
    实测 cmd=+1 时 200 步净位移 -4.5m（后退！）、cmd=0 漂移 -1.16m。
    v24 的修复是**相位差 δ 硬切换**（不是整体乘方向系数，后者等价于相位
    平移 π、拓扑不变，已实测证伪）：
      δ=0（vx≥0）→ 净推进 +x；δ=π（vx<0）→ 净推进 -x；
      中间相位 π/4~3π/4 全部失稳。
    幅值按 |vx| 门控，零指令 q_des=q_default 纯站立，修复"零速腿抖"。

v25（2026-10-23 评审后）：gait_ik() 用逆运动学直接编码全向步进。
    实测证明 sine 版只在前向成立：后退半轴本身会塌、vy/yaw 不产生净位移、
    速度上限 ~0.27 m/s。gait_ik 覆盖后退/横移/转向与全程速度，
    但它是几何精确参考，低刚度（旧仿真 kp=60）下 PD 静差会直接压塌。
    默认仍为 "sine"；ik 仅在 probe_gait 判决（全指令存活 + 跟踪≥60%）
    通过后启用（仿真 kp 已于同日对齐实机 150）。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# 加电站立位（2026-10-23 kp=150 实测标定；替换原 [0,0.9,-1.8]×4）
Q_DEFAULT = np.array([-0.0134, 0.8611, -1.7340, 0.0160, 0.8531, -1.7100,
                      0.0088, 0.8801, -1.7210, -0.0133, 0.9111, -1.6946])
DT = 0.02  # 策略步长

# 步态幅值（rad）
A_HIP = 0.10
A_THIGH = 0.35
A_CALF = 0.35

# 频率范围（Hz）
FREQ_MIN = 2.0
FREQ_MAX = 4.0

# thigh-calf 相位差（rad），v24 探针扫描选定（/tmp/probe_gait2.py）：
# δ=0 → +1.0m/200步（前进，存活）；δ=π → -4.5m/200步（后退，存活）；
# δ∈{π/4, π/2, 3π/4} 全部失稳摔倒
GAIT_DELTA_FWD = 0.0
GAIT_DELTA_BWD = float(np.pi)

# ---------------------------------------------------------------------------
# v25：IK 驱动基准步态（EnvConfig.gait_mode = "ik"）
# ---------------------------------------------------------------------------
# 2026-10-23 评审实测（关 DR / action=0 纯基准步态）：
#   · vx<0 的基准轨迹本身在塌（δ=π 构型低频不稳）
#   · vy/yaw 的基准完全不产生净位移（hip 只给静态偏置，没有踏步）
#   · 基准速度上限 ~0.27 m/s，远低于指令上限 1.5——残差补不了这个缺口
# 因此把"朝指令方向的连续步进"直接编码进基准：支撑相脚相对机身匀速后移，
# 摆动相抬脚回位；横移/转向由每只脚的期望速度
#     v_i = (vx - yaw·y_i, vy + yaw·x_i)
# 直接给出。基准覆盖全方向与全程速度，残差只做修正。
#
# ⚠️ 2026-10-23 第二轮评审（当时仿真 kp=60）：本轨迹是**几何精确**参考，
#    kp=60 的 PD 静差约 3.5cm（脚中心 −0.251 vs 标称 −0.286），直接吃掉
#    落点精度 → 关 DR、action=0 时所有非零指令 17~38 步内摔倒；
#    kp=100/150/250 存活；IK_LIFT=0 可活 120 步（抬脚是触发器）。
#    同日已把仿真 kp 从 60 对齐到实机的 150（见 XML），因此该否决结论
#    必须在 kp=150 下重新 probe；评审在 kp=150 测到速度跟踪仅 ~30%，
#    疑支撑相脚打滑。
#    → 在 probe_gait.py 判决（全指令存活 + 跟踪≥60%）之前，禁止用于训练。
L1 = 0.22               # 髋→膝连杆长（m）
L2 = 0.24               # 膝→脚连杆长（m）
# 2026-10-23：随加电姿态标定更新（旧值 0.0156665 / -0.2859406 对应手扶标定的
# 旧 q_default）。新值 = 新站立位下四足端在 hip 系的均值（MuJoCo 实测），使
# gait_ik(零指令) ≈ 0。
FOOT_X_NOM = 0.008037     # 脚端相对 hip 关节的 x（新站立位，FK 实测均值）
FOOT_Z_NOM = -0.302601    # 同上 z
THIGH_Y_OFF = 0.1039    # thigh 关节相对 hip 关节的横向偏移
LEG_X = np.array([0.22337, 0.22337, -0.22337, -0.22337])   # 脚在 body 系 x
LEG_Y = np.array([0.1639, -0.1639, 0.1639, -0.1639])       # 脚在 body 系 y
THIGH_Y = np.array([THIGH_Y_OFF, -THIGH_Y_OFF,
                    THIGH_Y_OFF, -THIGH_Y_OFF])            # hip 系横向偏移
LEG_PHASE = np.array([0.0, 0.5, 0.5, 0.0])                 # 对角 trot
IK_DUTY = 0.55          # 支撑相占比
IK_LIFT = 0.06          # 摆动抬脚高度（m）
IK_MAX_STEP = 0.20      # 支撑相脚位移上限（m），防目标超出腿长
# v26（2026-10-28）：IK 规划的步幅在仿真里只兑现 44%（0.3 指令 → 实测 0.132 m/s）。
# 试过把它当"标定损失"补回来（×2.3）→ **实测失败**：fwd_0.3 从存活 0.764m
# 变成摔@76，后退/横移/转向全部开始摔。
# 结论：44% 不是可标定的增益，而是这条腿在该频率下的**实际执行极限**；
# 强行加大步幅只会打滑/失稳。所以保持 1.0，改从"降低指令范围"入手
# （残差能补的缺口有限，见 10-28 IK 训练饱和记录 A=88~92%）。
IK_STRIDE_GAIN = 1.0


def _ik_planar(x, z):
    """二维矢状面 2 连杆 IK（膝走负分支），返回绝对角 (q_thigh, q_calf)"""
    r2 = jnp.maximum(x * x + z * z, 1e-6)
    c2 = jnp.clip((r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0)
    q2 = -jnp.arccos(c2)
    q1 = (jnp.arctan2(-x, -z)
          - jnp.arctan2(L2 * jnp.sin(q2), L1 + L2 * jnp.cos(q2)))
    return q1, q2


def gait_ik(episode_steps: jax.Array, command: jax.Array) -> jax.Array:
    """IK 驱动 trot：直接编码全向步进，返回 (…,12) 相对 q_default 的增量"""
    vx = command[..., 0]
    vy = command[..., 1]
    yaw = command[..., 2]
    speed = jnp.sqrt(vx ** 2 + vy ** 2) + 0.25 * jnp.abs(yaw)

    # 步频随速度提高；零指令时抬脚高度与步幅都门控到 0（纯站立）
    freq = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * jnp.clip(speed / 1.2, 0.0, 1.0)
    cyc = 1.0 / freq
    amp = jnp.clip(speed / 0.3, 0.0, 1.0)

    phase = (episode_steps * DT / cyc) % 1.0               # (N,) 归一化相位
    ph = (phase[..., None] + LEG_PHASE) % 1.0              # (N,4)

    # 每只脚的期望机体速度（含 yaw 切向分量）；支撑相脚相对机身反向移动
    T_st = IK_DUTY * cyc
    vx_i = vx[..., None] - yaw[..., None] * LEG_Y
    vy_i = vy[..., None] + yaw[..., None] * LEG_X
    stride_x = jnp.clip(vx_i * T_st[..., None] * IK_STRIDE_GAIN,
                        -IK_MAX_STEP, IK_MAX_STEP)
    stride_y = jnp.clip(vy_i * T_st[..., None] * IK_STRIDE_GAIN,
                        -IK_MAX_STEP, IK_MAX_STEP)

    stance = ph < IK_DUTY
    fr_st = ph / IK_DUTY
    fr_sw = (ph - IK_DUTY) / (1.0 - IK_DUTY)
    # 支撑相 +d/2 → -d/2 线性；摆动相 -d/2 → +d/2 余弦平滑（两端零速）
    s_sw = 0.5 - 0.5 * jnp.cos(jnp.pi * fr_sw)
    off_x = jnp.where(stance, stride_x * (0.5 - fr_st),
                      stride_x * (-0.5 + s_sw))
    off_y = jnp.where(stance, stride_y * (0.5 - fr_st),
                      stride_y * (-0.5 + s_sw))
    lift = jnp.where(stance, 0.0,
                     amp[..., None] * IK_LIFT * jnp.sin(jnp.pi * fr_sw))

    # 目标脚位（相对 hip 关节）
    x_t = FOOT_X_NOM + off_x
    y_t = THIGH_Y + off_y
    z_t = FOOT_Z_NOM + lift

    # hip 转角把腿平面旋到目标横向，剩余平面内用 2 连杆 IK
    ay = jnp.sqrt(jnp.maximum(y_t * y_t + z_t * z_t
                              - THIGH_Y ** 2, 1e-6))
    z_leg = -ay
    # (y,z) 平面的旋转角须用 atan2(z, y)（R_x 作用在 (y,z) 上，y 相当于 x 轴）
    q_hip = jnp.arctan2(z_t, y_t) - jnp.arctan2(z_leg, THIGH_Y)
    q_thigh, q_calf = _ik_planar(x_t, z_leg)

    q = jnp.stack([q_hip, q_thigh, q_calf], axis=-1).reshape(
        command.shape[:-1] + (12,))
    return q - Q_DEFAULT


def gait(episode_steps: jax.Array, command: jax.Array) -> jax.Array:
    """生成基准 trot 步态

    Args:
        episode_steps: (N,) 策略步数（标量或批次）
        command: (N, 3) [vx, vy, yaw_rate]

    Returns:
        (N, 12) 基准关节角增量（相对 q_default）
    """
    # 频率由 |vx| 决定（v24：负 vx 也走高频，δ=π 后退构型在低频不稳）
    vx = command[..., 0]
    vy = command[..., 1]
    yaw = command[..., 2]
    t = jnp.clip(jnp.abs(vx) / 1.5, 0.0, 1.0)
    freq = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * t

    # 相位
    phase = (episode_steps * DT * freq * 2 * jnp.pi) % (2 * jnp.pi)

    # 对角腿相位差 π
    phase_a = phase               # FL, RR
    phase_b = phase + jnp.pi      # FR, RL

    # 幅值调制（v24：∝|vx| 速度门控，零指令幅值=0 纯站立）
    # v26（2026-10-28 实机判决）：线性门控在低速段幅值太小——vx=0.2 只有
    # 0.070 rad(4°)，实机实测摆腿抬脚不足、力矩涨到 26 N·m 也不产生位移。
    # 改 sqrt 缩放：保持零指令=0、高速端不变（vx≥1.0 仍饱和到 0.35），
    # 低速端翻倍：vx=0.1 → 6.3°，0.2 → 9.0°，0.3 → 11.0°，0.5 → 14.2°。
    amp_scale = jnp.sqrt(jnp.clip(jnp.abs(vx) / 1.0, 0.0, 1.0))
    amp_thigh = A_THIGH * amp_scale
    amp_calf = A_CALF * amp_scale
    # vy 影响 hip 左右摆
    amp_hip = A_HIP * jnp.clip(jnp.abs(vy) / 0.5, 0.0, 1.0)
    # yaw 影响左右腿 hip 差（转向时内外侧步幅不同）
    yaw_hip = A_HIP * 0.5 * jnp.clip(yaw / 2.0, -1.0, 1.0)

    # hip: vy 正向左移 → 左腿外展(hip+)，右腿内收(hip-)
    hip_sign = jnp.sign(vy + 1e-8)
    fl_hip = amp_hip * hip_sign + yaw_hip
    fr_hip = -amp_hip * hip_sign - yaw_hip
    rl_hip = amp_hip * hip_sign - yaw_hip
    rr_hip = -amp_hip * hip_sign + yaw_hip

    # thigh/calf 相位差 δ 按 vx 符号硬切换（v24 探针扫描判决）：
    # 实测稳定极限环只有 δ=0（+1.0m/200步 前进）与 δ=π（-4.5m/200步 后退），
    # 中间相位（π/4~3π/4）全部失稳摔倒。δ=0 轨道退化为直线、无法编码
    # 后退方向，故后退必须切 δ=π；vx=0 时幅值=0，切换天然连续。
    delta = jnp.where(vx >= 0.0, GAIT_DELTA_FWD, GAIT_DELTA_BWD)

    fl_thigh = amp_thigh * jnp.sin(phase_a)
    fr_thigh = amp_thigh * jnp.sin(phase_b)
    rl_thigh = amp_thigh * jnp.sin(phase_b)
    rr_thigh = amp_thigh * jnp.sin(phase_a)

    fl_calf = amp_calf * jnp.sin(phase_a + delta)
    fr_calf = amp_calf * jnp.sin(phase_b + delta)
    rl_calf = amp_calf * jnp.sin(phase_b + delta)
    rr_calf = amp_calf * jnp.sin(phase_a + delta)

    gait_q = jnp.stack([
        fl_hip, fl_thigh, fl_calf,   # FL
        fr_hip, fr_thigh, fr_calf,   # FR
        rl_hip, rl_thigh, rl_calf,   # RL
        rr_hip, rr_thigh, rr_calf,   # RR
    ], axis=-1)

    return gait_q
