"""MJX / CPU 动力学一致性校验（阶段 1 地基关）

三组零策略轨迹（同 home keyframe 初态、ctrl=站立常量、1000 物理步=2s）：
  A. CPU MuJoCo + 原 scene_positional.xml（椭圆锥/cylinder/Euler）
  B. CPU MuJoCo + 新 scene_mjx.xml（金字塔锥/capsule/implicitfast）
  C. MJX       + 新 scene_mjx.xml

对比：
  B-vs-C = 纯引擎差（求解器不同导致）
  A-vs-B = 模型适配差（XML 兼容性改动导致）
通道：base xyz(3) + quat(4) + 12 关节角，共 qpos 19 维。
图：logs/mjx_consistency.png

用法：python scripts/verify_mjx_consistency.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
ENVS = ROOT / "envs"
N_STEPS = 1000          # 0.002s × 1000 = 2s
KEY_HOME = "home"


def load_cpu(scene_name: str):
    model = mujoco.MjModel.from_xml_path(str(ENVS / scene_name))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEY_HOME)
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def rollout_cpu(model, data) -> np.ndarray:
    traj = np.zeros((N_STEPS + 1, model.nq))
    traj[0] = data.qpos.copy()
    for i in range(N_STEPS):
        mujoco.mj_step(model, data)
        traj[i + 1] = data.qpos.copy()
    return traj


def rollout_mjx(model, data) -> np.ndarray:
    mx = mjx.put_model(model)
    dx0 = mjx.put_data(model, data)

    @jax.jit
    def run(dx):
        def step(d, _):
            d = mjx.step(mx, d)
            return d, d.qpos
        _, traj = jax.lax.scan(step, dx, None, length=N_STEPS)
        return traj

    traj = np.asarray(run(dx0))
    return np.concatenate([data.qpos[None, :], traj], axis=0)


def diff_report(name: str, t1: np.ndarray, t2: np.ndarray):
    """打印两组轨迹的逐通道误差（四元数部分用向量整体误差）"""
    d = np.abs(t1 - t2)
    print(f"\n--- {name} ---")
    print(f"  base xyz : max={d[:, 0:3].max():.5f} m , "
          f"rms={np.sqrt((d[:, 0:3]**2).mean()):.5f} m")
    print(f"  quat     : max={d[:, 3:7].max():.6f}, "
          f"rms={np.sqrt((d[:, 3:7]**2).mean()):.6f}")
    print(f"  12 关节角: max={d[:, 7:].max():.5f} rad "
          f"({np.degrees(d[:, 7:].max()):.3f}°), "
          f"rms={np.degrees(np.sqrt((d[:, 7:]**2).mean())):.3f}°")
    # 单关节最大误差，定位是哪条腿
    jmax = d[:, 7:].max(axis=0)
    names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
             "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]
    worst = int(np.argmax(jmax))
    print(f"  最大单关节: {names[worst]} max={jmax[worst]:.5f} rad "
          f"({np.degrees(jmax[worst]):.3f}°)")
    return d


def make_plot(ra, rb, rc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(N_STEPS + 1) * 0.002
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = [
        ("base_z", 2, "m"),
        ("base_x", 0, "m"),
        ("FL_thigh joint pos", 8, "rad"),
    ]
    titles = ["A=CPU original XML", "B=CPU mjx XML", "C=MJX mjx XML"]
    for ax, (ylabel, idx, unit) in zip(axes, labels):
        ax.plot(t, ra[:, idx], label=titles[0], lw=2)
        ax.plot(t, rb[:, idx], "--", label=titles[1], lw=1.5)
        ax.plot(t, rc[:, idx], ":", label=titles[2], lw=1.5)
        ax.set_ylabel(f"{ylabel} ({unit})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Zero-policy 2s: MJX vs CPU consistency")
    out = ROOT / "logs" / "mjx_consistency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\n图已保存：{out}")


def main():
    print(f"jax devices = {jax.devices()}")
    print(f"零策略 {N_STEPS} 物理步 = {N_STEPS*0.002:.1f}s\n")

    # A：CPU + 原模型（独立 data，避免互相污染）
    ma, da = load_cpu("scene_positional.xml")
    # B：CPU + MJX 模型
    mb, db = load_cpu("scene_mjx.xml")
    # C：MJX + MJX 模型（从与 B 相同的初态出发）
    mc, dc = load_cpu("scene_mjx.xml")

    print("rollout A: CPU + scene_positional.xml ...")
    ra = rollout_cpu(ma, da)
    print("rollout B: CPU + scene_mjx.xml ...")
    rb = rollout_cpu(mb, db)
    print("rollout C: MJX + scene_mjx.xml（含 jit 编译）...")
    rc = rollout_mjx(mc, dc)

    print("\n=== 终值 ===")
    for tag, tr in [("A", ra), ("B", rb), ("C", rc)]:
        print(f"  {tag}: base_xyz={tr[-1, :3]}  base_z min={tr[:, 2].min():.4f}")

    diff_report("B vs C（纯引擎差，越小越好）", rb, rc)
    diff_report("A vs B（模型适配差）", ra, rb)

    make_plot(ra, rb, rc)

    # 粗略通过线：站立 2s 内引擎差关节 <0.5°、高度 <1cm；适配差关节 <0.5°
    eng = np.abs(rb - rc)
    adapt = np.abs(ra - rb)
    ok_eng = eng[:, 7:].max() < np.radians(0.5) and eng[:, 2].max() < 0.01
    ok_adapt = adapt[:, 7:].max() < np.radians(0.5)
    print("\n判定阈值：关节 max<0.5°，（引擎差另需 base_z<1cm）")
    print(f"  引擎差：{'✅' if ok_eng else '⚠️ 超阈值，需分析'}")
    print(f"  适配差：{'✅' if ok_adapt else '⚠️ 超阈值，需分析'}")


if __name__ == "__main__":
    main()
