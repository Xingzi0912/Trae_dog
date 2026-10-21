"""爆炸源探测：摔倒后 20ms 盲区内，是谁把 qvel 打爆的？

背景：执行器力矩已被关节层 actuatorfrcrange 钳在 ±20/25 N·m
（probe_act_clamp 两轮实证 25/25），"单次力矩超标"假说不成立。
受力链 qacc = M⁻¹·(qfrc_actuator + qfrc_passive + qfrc_constraint) 里
唯一无界的是 qfrc_constraint —— 接触/限位约束求解器的输出，
actuatorfrcrange 管不到它，且穿透越深修正冲量越大。

本脚本在 MJX（出事的同一引擎）构造三个场景，各跑 10 物理子步
（= 1 个策略步的 20ms 终止判定盲区），逐子步记录：
  max|qvel| / max|qfrc_constraint| / max|qfrc_actuator|

  A. 站立（home, ctrl=q_default）               —— 基线
  B. 高速侧摔（z=0.10, roll=1.2rad, vz=-2.5）   —— 模拟真实摔倒撞地
  C. 静态深穿透（z=0.02，腿穿地 ~0.2m）         —— 极端场景

判定：
  - qfrc_actuator 全程 ≤25    → 执行器无辜（再次佐证）
  - B/C 的 qfrc_constraint ≫ A 且 qvel 冲过护栏线 100 → 接触爆炸实锤
  - 若只有 C 爆、B 不爆      → 爆炸需要"深穿透"特定构型，
                               与 ~10⁹ 环境步才一次吻合

用法：XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/probe_explosion.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "envs" / "scene_mjx.xml"
Q_DEFAULT = np.array([0.0, 0.9, -1.8] * 4)   # 站立 ctrl（环境站立语义）
N_SUB = 10  # 一个策略步 = 10 物理子步 = 20ms 盲区
GUARD = 100.0  # v21 护栏线


def make_state(m, base_z=None, roll=0.0, vz=0.0):
    """从 home keyframe 出发构造指定场景，返回 mjx.Data"""
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, key)
    if base_z is not None:
        d.qpos[2] = base_z
    if roll != 0.0:
        d.qpos[3:7] = [np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0]
    d.qvel[2] = vz
    d.ctrl[:] = Q_DEFAULT
    mujoco.mj_forward(m, d)
    return mjx.put_data(m, d)


def run_scene(name, mx, dx):
    has_qfc = hasattr(dx, "qfrc_constraint")
    step_fn = jax.jit(lambda d: mjx.step(mx, d))
    print(f"\n场景 {name}   (qfrc_constraint 字段: {'有' if has_qfc else '无，跳过该列'})")
    hdr = f"  {'sub':>3} {'max|qvel|':>12}"
    if has_qfc:
        hdr += f" {'max|qfrc_con|':>14}"
    hdr += f" {'max|qfrc_act|':>14}"
    print(hdr)
    for i in range(N_SUB):
        dx = step_fn(dx)
        qv = float(jnp.abs(dx.qvel).max())
        qa = float(jnp.abs(dx.qfrc_actuator).max())
        row = f"  {i+1:>3} {qv:>12.3f}"
        if has_qfc:
            qc = float(jnp.abs(dx.qfrc_constraint).max())
            row += f" {qc:>14.3f}"
        row += f" {qa:>14.3f}"
        if qv >= GUARD:
            row += "  ← 越护栏线"
        print(row)


def main():
    print(f"jax devices = {jax.devices()}")
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    mx = mjx.put_model(m)

    run_scene("A 站立基线（home, ctrl=q_default）",
              mx, make_state(m))
    run_scene("B 高速侧摔（z=0.10, roll=1.2, vz=-2.5）",
              mx, make_state(m, base_z=0.10, roll=1.2, vz=-2.5))
    run_scene("C 静态深穿透（z=0.02）",
              mx, make_state(m, base_z=0.02))

    print("\n判定：")
    print("  · qfrc_act 全程 ≤25          → 执行器无辜")
    print("  · B/C 的 qfrc_con ≫ A 且 qvel 冲线 → 接触约束力 = 爆炸燃料")
    print("  · 仅 C 爆                    → 需深穿透构型，与小概率吻合")


if __name__ == "__main__":
    main()
