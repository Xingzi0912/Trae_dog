#!/usr/bin/env python3
"""probe_gait.py —— 开环基准步态探针（残差置零，关 DR）

目的：判决 EnvConfig.gait_mode 的 "sine"（v24）与 "ik"（v25）谁能让
"指令 → 净运动"在全方向上成立。只跑基准步态（action=0），排除策略影响。

用法（服务器仓库根目录）：
    python3 scripts/probe_gait.py --mode sine
    python3 scripts/probe_gait.py --mode ik

判据（同一指令下对比）：
    · 前进/后退：dx 的符号必须与 vx 一致，步数跑满不摔
    · 横移：dy 的符号必须与 vy 一致
    · 原地转：dyaw 的符号必须与 yaw 一致
    · 零指令：|dx|、|dy|、|dyaw| 都接近 0（允许 ~5cm 落定瞬态）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax                     # noqa: E402
import jax.numpy as jnp        # noqa: E402

from envs.dog_env_mjx import DogEnvMJX, EnvConfig  # noqa: E402
import envs.gait as G                              # noqa: E402

DT = 0.02
SETTLE = 10                    # 前 10 步作为落定瞬态，不计入位移
CMDS = [
    ("stand",    (0.0, 0.0, 0.0)),
    ("fwd_0.3",  (0.3, 0.0, 0.0)),
    ("fwd_1.0",  (1.0, 0.0, 0.0)),
    ("fwd_1.5",  (1.5, 0.0, 0.0)),
    ("bwd_0.3",  (-0.3, 0.0, 0.0)),
    ("bwd_0.5",  (-0.5, 0.0, 0.0)),
    ("left_0.4", (0.0, 0.4, 0.0)),
    ("right_0.4", (0.0, -0.4, 0.0)),
    ("yaw+1.0",  (0.0, 0.0, 1.0)),
    ("yaw-1.0",  (0.0, 0.0, -1.0)),
    ("diag",     (0.8, 0.3, 0.8)),
]


def yaw_of(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def run_one(env, cmd, steps: int):
    st, _ = env.reset(jax.random.PRNGKey(0))
    c = jnp.array([cmd], dtype=jnp.float32)
    act = jnp.zeros((1, 12), dtype=jnp.float32)

    for _ in range(min(SETTLE, steps)):
        st = st._replace(command=c)
        st, _, _, _, _, _ = env.step(st, act)

    q = np.asarray(jax.device_get(st.dx.qpos[0]))
    x0, y0, yaw0 = float(q[0]), float(q[1]), yaw_of(q[3:7])

    for i in range(SETTLE, steps):
        st = st._replace(command=c)
        st, _, _, term, trunc, _ = env.step(st, act)
        if bool(jax.device_get(term[0])):
            q = np.asarray(jax.device_get(st.dx.qpos[0]))
            return (float(q[0] - x0), float(q[1] - y0),
                    wrap(yaw_of(q[3:7]) - yaw0), i + 1, False)

    q = np.asarray(jax.device_get(st.dx.qpos[0]))
    return (float(q[0] - x0), float(q[1] - y0),
            wrap(yaw_of(q[3:7]) - yaw0), steps, True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sine", "ik"], required=True)
    ap.add_argument("--steps", type=int, default=300)
    # --- 实验开关：全部缺省=保持项目现值，只用于离线判决，不写回配置 ---
    ap.add_argument("--kp", type=float, default=None,
                    help="覆盖 12 个位置执行器的 kp（默认用 XML 的 60）")
    ap.add_argument("--fric", type=float, default=None,
                    help="覆盖 4 只足底的切向摩擦（默认 0.9），用于判打滑")
    ap.add_argument("--lift", type=float, default=None,
                    help="覆盖 IK 抬脚高度（默认 %.3f）" % G.IK_LIFT)
    ap.add_argument("--duty", type=float, default=None,
                    help="覆盖 IK 支撑相占比（默认 %.2f）" % G.IK_DUTY)
    ap.add_argument("--max_step", type=float, default=None,
                    help="覆盖 IK 支撑相步幅上限（默认 %.2f m）" % G.IK_MAX_STEP)
    ap.add_argument("--freq_max", type=float, default=None,
                    help="覆盖步频上限（默认 %.1f Hz）" % G.FREQ_MAX)
    ap.add_argument("--stride_gain", type=float, default=None,
                    help="覆盖 IK 规划步幅的标定系数（默认 %.2f）。"
                         "调大 = 步幅更大（对抗打滑），过大会失稳" % G.IK_STRIDE_GAIN)
    args = ap.parse_args()

    # 步态常量必须在首次 step（JIT trace）之前写入
    for name, val in (("IK_LIFT", args.lift), ("IK_DUTY", args.duty),
                      ("IK_MAX_STEP", args.max_step),
                      ("FREQ_MAX", args.freq_max),
                      ("IK_STRIDE_GAIN", args.stride_gain)):
        if val is not None:
            setattr(G, name, val)

    cfg = EnvConfig(
        dr_enabled=False,                 # 标称物理，排除 DR 干扰
        gait_mode=args.mode,
        max_steps=10 ** 9,                # 关 episode 截断
        cmd_resample_steps=(10 ** 9, 10 ** 9),   # 关内部指令重采样
    )
    env = DogEnvMJX(n_envs=1, config=cfg)

    # 模型覆盖（只在探针内生效；训练走 XML + DR）
    mx = env._mx
    if args.kp is not None:
        mx = mx.replace(
            actuator_gainprm=mx.actuator_gainprm.at[:, 0].set(args.kp),
            actuator_biasprm=mx.actuator_biasprm.at[:, 1].set(-args.kp),
        )
    if args.fric is not None:
        mx = mx.replace(
            geom_friction=mx.geom_friction.at[env._foot_ids, 0].set(args.fric))
    env._mx_eff = mx

    print(f"\n[probe] mode={args.mode}, kp={args.kp or 'XML'}, "
          f"fric={args.fric or 'XML(0.9)'}, "
          f"lift={G.IK_LIFT:.3f}, duty={G.IK_DUTY:.2f}, "
          f"max_step={G.IK_MAX_STEP:.2f}, freq_max={G.FREQ_MAX:.1f}")
    n = args.steps - SETTLE

    print(f"\n=== gait_mode = {args.mode}  "
          f"({n} 步 = {n * DT:.1f}s，残差=0，关 DR) ===")
    print(f"{'cmd':<10} {'vx':>5} {'vy':>5} {'yaw':>5} | "
          f"{'dx(m)':>8} {'dy(m)':>8} {'dyaw(deg)':>10} "
          f"{'vx_eff':>7} {'vy_eff':>7} {'结果':>5}")
    print("-" * 82)
    for name, cmd in CMDS:
        dx, dy, dyaw, step_done, alive = run_one(env, cmd, args.steps)
        t = max(1, step_done - SETTLE) * DT
        print(f"{name:<10} {cmd[0]:>5.1f} {cmd[1]:>5.1f} {cmd[2]:>5.1f} | "
              f"{dx:>+8.3f} {dy:>+8.3f} {np.degrees(dyaw):>+10.1f} "
              f"{dx / t:>+7.3f} {dy / t:>+7.3f} "
              f"{'存活' if alive else '摔@%d' % step_done:>5}")
    print()


if __name__ == "__main__":
    main()
