"""一次性探针：原模型几何（cylinder + condim=6）在 MJX 3.13 是否可跑

从 dog_positional.xml 复制，仅做两处最小改动：
  - 删除 <sensor> 段
  - option 加 integrator=implicitfast（cone 保持 elliptic impratio=100）
cylinder/capsule、condim=6 全部保留原样。
能跑且匹配 CPU-A，则 dog_mjx.xml 可进一步最小化差异。
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

ENVS = Path(__file__).resolve().parent.parent / "envs"
N_STEPS = 1000


def main():
    xml = (ENVS / "dog_positional.xml").read_text(encoding="utf-8")
    # option 行加积分器（原行无 integrator）
    xml = xml.replace(
        '<option cone="elliptic" impratio="100" />',
        '<option cone="elliptic" impratio="100" integrator="implicitfast" timestep="0.002" />',
    )
    # 删除 <sensor>...</sensor>
    xml = re.sub(r"\s*<sensor>.*?</sensor>", "", xml, flags=re.DOTALL)

    dog_p = ENVS / "_probe_origgeo_dog.xml"
    scene_p = ENVS / "_probe_origgeo_scene.xml"
    dog_p.write_text(xml, encoding="utf-8")
    scene_p.write_text(
        f'<mujoco><include file="{dog_p.name}"/>\n'
        '<statistic center="0 0 0.1" extent="0.8"/>\n'
        '<worldbody><geom name="floor" size="0 0 0.05" type="plane" '
        'friction="1.2 0.1 0.01"/></worldbody></mujoco>\n',
        encoding="utf-8",
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_p))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mx = mjx.put_model(model)
        for w in caught:
            print(f"WARN {w.category.__name__}: {w.message}")
        if not caught:
            print("put_model: 无 warning")

        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        dx0 = mjx.put_data(model, data)

        @jax.jit
        def run(dx):
            def step(d, _):
                return mjx.step(mx, d), d
            _, traj = jax.lax.scan(step, dx, None, length=N_STEPS)
            return traj

        traj = run(dx0)
        q = np.asarray(traj.qpos)
        print(f"NaN={bool(jp.isnan(q).any())}")
        print(f"base_x={q[-1,0]:+.5f} base_z={q[-1,2]:.5f} (min {q[:,2].min():.5f})")
        # 与 CPU-A 参考对比关节终值
        names = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
                 "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]
        ref_home = np.tile([0, 0.9, -1.8], 4)
        err_deg = np.degrees(np.abs(q[-1, 7:] - ref_home))
        worst = int(np.argmax(err_deg))
        print(f"相对 home 最大关节偏差: {names[worst]} {err_deg[worst]:.3f}°")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
    finally:
        dog_p.unlink(missing_ok=True)
        scene_p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
