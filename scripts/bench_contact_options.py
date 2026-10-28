#!/usr/bin/env python3
"""bench_contact_options.py —— 接触求解配置的【吞吐 + 站立精度】扫描

背景（2026-10-28）：
    实机 rollout 50s/update ≈ 9,800 env·物理子步/s，而脚本内标定的"快路径"
    也只有 20,963/s —— 对 4090 而言低约两个数量级。首要嫌疑是接触配置：
        cone="elliptic"  impratio="100"  condim="6"
    impratio=100 把摩擦锥拉成极扁形状，条件数差，求解器迭代次数暴涨。

做法：直接改 MjModel 字段（不改 XML 文件），逐变体测：
    ① 吞吐（env·子步/s，与 baseline 的加速比）
    ② 站立一致性（1 秒后 base_z、与 keyframe 的最大关节偏差）

用法：
    python scripts/bench_contact_options.py --envs 2048 --steps 100
    python scripts/bench_contact_options.py --envs 2048 --steps 100 --variants base condim3
    python scripts/bench_contact_options.py --envs 4096 --steps 100 --only base
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENE = ROOT / "envs" / "scene_mjx.xml"
CONE_PYR, CONE_ELL = 0, 1          # mjCONE_PYRAMIDAL / mjCONE_ELLIPTIC

# 变体： (cone, impratio, condim, rolling_friction, 说明)
VARIANTS = {
    "base":      (CONE_ELL, 100.0, 6, 0.01, "现状：椭圆锥 impratio=100 condim=6"),
    "imp10":     (CONE_ELL, 10.0,  6, 0.01, "椭圆锥 impratio=10"),
    "imp1":      (CONE_ELL, 1.0,   6, 0.01, "椭圆锥 impratio=1（各向同性）"),
    "condim3":   (CONE_ELL, 100.0, 3, 0.01, "椭圆锥 condim=3（去掉扭转/滚动摩擦）"),
    "noroll":    (CONE_ELL, 100.0, 6, 0.0,  "椭圆锥 condim=6 但滚动摩擦=0"),
    "pyr":       (CONE_PYR, 100.0, 6, 0.01, "金字塔锥 impratio=100 condim=6"),
    "pyr_c3":    (CONE_PYR, 100.0, 3, 0.01, "金字塔锥 condim=3"),
    "fast":      (CONE_PYR, 1.0,   3, 0.0,  "金字塔锥+impratio=1+condim=3+无滚动（最激进）"),
}


def load_model(cone, impratio, condim, roll=0.01):
    mjm = mujoco.MjModel.from_xml_path(str(SCENE))
    mjm.opt.cone = cone
    mjm.opt.impratio = impratio
    foot_ids = [i for i in range(mjm.ngeom)
                if (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_foot")]
    if not foot_ids:
        raise RuntimeError("找不到 *_foot geom，检查 scene_mjx.xml")
    for i in foot_ids:
        mjm.geom_condim[i] = condim
        if roll != 0.01:                 # 只在需要时改，避免浮点噪声
            mjm.geom_friction[i, 2] = roll
    return mjm, foot_ids


def make_batch(mx, n_envs):
    dx = mjx.make_data(mx)
    if mx.nkey > 0:
        qpos = np.asarray(mx.key_qpos[0])
        ctrl = np.asarray(mx.key_ctrl[0])
    else:
        qpos = np.asarray(mx.qpos0)
        ctrl = np.zeros(mx.nu)
    dx = dx.replace(qpos=jnp.asarray(qpos), ctrl=jnp.asarray(ctrl))
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_envs,) + x.shape), dx)


def bench(cone, impratio, condim, roll, n_envs, n_steps, settle):
    mjm, _ = load_model(cone, impratio, condim, roll)
    mx = mjx.put_model(mjm)
    dx = make_batch(mx, n_envs)
    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    dx = step(mx, dx)                      # 编译 + 预热（不计时）
    jax.block_until_ready(dx.qpos)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        dx = step(mx, dx)
    jax.block_until_ready(dx.qpos)
    dt = time.perf_counter() - t0

    ms = dt / n_steps * 1e3
    rate = n_envs * n_steps / dt

    # 继续跑到 settle 步，检查站立稳态
    for _ in range(settle):
        dx = step(mx, dx)
    jax.block_until_ready(dx.qpos)
    bz = float(np.asarray(dx.qpos[:, 2]).mean())
    q = np.asarray(dx.qpos[0])
    q_ref = np.asarray(mjm.key_qpos[0])
    dq_max = float(np.abs(q[7:] - q_ref[7:]).max())
    return ms, rate, bz, dq_max


def main():
    ap = argparse.ArgumentParser(description="接触配置吞吐 + 站立精度扫描")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=100, help="计时用的物理子步数")
    ap.add_argument("--settle", type=int, default=500,
                    help="站立一致性检查前先跑多少子步（默认 500 = 1 秒）")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--only", default=None, help="只跑一个变体名")
    args = ap.parse_args()

    names = [args.only] if args.only else args.variants
    print(f"MJX {jax.__version__} | devices={jax.devices()} | envs={args.envs} "
          f"| 计时 {args.steps} 子步")
    print(f"{'变体':<10}{'ms/子步':>10}{'env·子步/s':>14}{'相对base':>10}"
          f"{'base_z':>9}{'max|Δq|':>9}  说明")
    print("-" * 108)

    base_rate = None
    for name in names:
        cone, imp, cd, roll, desc = VARIANTS[name]
        try:
            ms, rate, bz, dq = bench(cone, imp, cd, roll, args.envs,
                                     args.steps, args.settle)
            if base_rate is None:
                base_rate = rate
            sp = rate / base_rate
            print(f"{name:<10}{ms:>10.2f}{rate:>14.0f}{sp:>9.2f}x"
                  f"{bz:>9.4f}{dq:>9.4f}  {desc}")
        except Exception as e:
            print(f"{name:<10}{'FAIL':>10}  {type(e).__name__}: {e}")
    print("-" * 108)
    print("判据：① 相对 base 的加速比 ≥3x 才值得考虑换；")
    print("      ② base_z 与 base 相差 <0.005m 且 max|Δq| <0.05rad 才算精度可接受；")
    print("      ③ 两个条件同时满足 → 直接改 XML 并重跑训练对比。")


if __name__ == "__main__":
    main()
