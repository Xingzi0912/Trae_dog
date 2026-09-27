"""MJX 兼容性探测（阶段 1）

三道关：
  1. mjx.put_model：捕获 MJX 对不支持特性的 warning
  2. mjx.make_data + keyframe home 初始化
  3. jit 编译 + 零策略（ctrl=keyframe 常量）跑 1000 个物理步，查 NaN/塌倒

用法：
    python scripts/test_mjx_compat.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

ENVS_DIR = Path(__file__).resolve().parent.parent / "envs"
SCENE = str(ENVS_DIR / "scene_mjx.xml")


def main() -> int:
    print(f"jax {jax.__version__}, devices={jax.devices()}")
    print(f"mujoco {mujoco.__version__}, scene = {SCENE}")

    # ---- CPU MuJoCo 加载（MJX 也从同一个 MjModel 转换）----
    mj_model = mujoco.MjModel.from_xml_path(SCENE)
    print(f"nq={mj_model.nq} nv={mj_model.nv} nu={mj_model.nu} "
          f"cone={mj_model.opt.cone} integrator={mj_model.opt.integrator}")

    # ---- 关 1：put_model 警告 ----
    print("\n[1] mjx.put_model ...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mx = mjx.put_model(mj_model)
    if caught:
        print(f"  ⚠️ {len(caught)} 条 warning：")
        for w in caught:
            print(f"    - {w.category.__name__}: {w.message}")
    else:
        print("  ✅ 无 warning")

    # ---- 关 2：make_data + keyframe home ----
    print("\n[2] make_data + keyframe home ...")
    mj_data = mujoco.MjData(mj_model)
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0, "keyframe home 缺失"
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    mujoco.mj_forward(mj_model, mj_data)
    dx0 = mjx.put_data(mj_model, mj_data)
    base_id = mj_model.body("base_link").id
    print(f"  base_z(init) = {float(dx0.xpos[base_id, 2]):.4f}（期望 0.27）")
    print(f"  ctrl[:3] = {dx0.ctrl[:3]}（期望 [0, 0.9, -1.8]）")

    # ---- 关 3：jit 零策略 1000 物理步（2s）----
    print("\n[3] jit 编译 + 零策略 1000 物理步 ...")

    @jax.jit
    def rollout(dx):
        def body(_, d):
            return mjx.step(mx, d)
        return jax.lax.fori_loop(0, 1000, body, dx)

    dx = rollout(dx0)   # 首次调用触发编译
    base_z = float(dx.xpos[base_id, 2])
    has_nan = bool(jp.isnan(dx.qpos).any() or jp.isnan(dx.qvel).any())
    print(f"  base_z(final) = {base_z:.4f}")
    print(f"  NaN = {has_nan}")
    print(f"  qpos[:3] (base xyz) = {dx.qpos[:3]}")

    ok = (not has_nan) and base_z > 0.25
    print("\n" + ("✅ MJX 兼容性通过（零策略 2s 站稳、无 NaN）"
                  if ok else "❌ 未通过（NaN 或塌倒），需按上面 warning/异常改 XML"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
