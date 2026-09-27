"""一次性探针：锥体 × 积分器 2×2 矩阵在 MJX 下的行为

目的：定位 A(原模型) vs B(MJX模型) 零策略 4.5° 适配差的来源。
在 envs/ 下生成临时 XML（保证 include/meshdir 相对路径可用），
每个变体用 MJX 跑零策略 1000 步，打印终值并与 CPU-A 参考对比。
跑完自动删除临时文件。

参考（CPU 原模型 scene_positional.xml）：
  base_z final ≈ 0.2725, base_x ≈ -0.050
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

ENVS = Path(__file__).resolve().parent.parent / "envs"
N_STEPS = 1000

# dog_mjx.xml 的 option 行作为替换目标
OPTION_ANCHOR = '<option cone="pyramidal" integrator="implicitfast" timestep="0.002" />'

VARIANTS = {
    "pyr+implicitfast": '<option cone="pyramidal" integrator="implicitfast" timestep="0.002" />',
    "ellip100+implicitfast": '<option cone="elliptic" impratio="100" integrator="implicitfast" timestep="0.002" />',
    "pyr+euler": '<option cone="pyramidal" integrator="Euler" timestep="0.002" />',
    "ellip100+euler": '<option cone="elliptic" impratio="100" integrator="Euler" timestep="0.002" />',
}


def run_variant(name: str, option_line: str):
    dog_xml = (ENVS / "dog_mjx.xml").read_text(encoding="utf-8")
    assert OPTION_ANCHOR in dog_xml
    dog_v = dog_xml.replace(OPTION_ANCHOR, option_line)
    dog_path = ENVS / f"_probe_{name}_dog.xml"
    scene_path = ENVS / f"_probe_{name}_scene.xml"
    dog_path.write_text(dog_v, encoding="utf-8")
    scene_path.write_text(
        f'<mujoco><include file="{dog_path.name}"/>\n'
        '<statistic center="0 0 0.1" extent="0.8"/>\n'
        '<worldbody><geom name="floor" size="0 0 0.05" type="plane" '
        'friction="1.2 0.1 0.01"/></worldbody></mujoco>\n',
        encoding="utf-8",
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mx = mjx.put_model(model)
        warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]

        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        dx0 = mjx.put_data(model, data)

        @jax.jit
        def run(dx):
            def step(d, _):
                d = mjx.step(mx, d)
                return d, d
            _, traj = jax.lax.scan(step, dx, None, length=N_STEPS)
            return traj

        traj = run(dx0)
        qpos = np.asarray(traj.qpos)
        nan = bool(jp.isnan(qpos).any())
        # RR_hip = 第 10 个关节 → qpos 7+9=16；base_x=0,z=2
        print(f"\n[{name}]")
        if warn_msgs:
            for m in warn_msgs:
                print(f"  WARN {m}")
        print(f"  NaN={nan}  base_x={qpos[-1,0]:+.5f}  base_z={qpos[-1,2]:.5f} "
              f"(min {qpos[:,2].min():.5f})  RR_hip={np.degrees(qpos[-1,16]):+.3f}°")
    except Exception as e:
        print(f"\n[{name}]  ❌ {type(e).__name__}: {e}")
    finally:
        dog_path.unlink(missing_ok=True)
        scene_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("参考 CPU-A：base_z≈0.2725 base_x≈-0.050（原椭圆锥+Euler+cylinder）")
    for nm, line in VARIANTS.items():
        run_variant(nm, line)
