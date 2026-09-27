"""MJX GPU 吞吐冒烟测试（阶段 0，4090 服务器）

用法：
    python scripts/bench_mjx_throughput.py                    # 默认 1024/2048/4096 env x 500 物理步
    python scripts/bench_mjx_throughput.py --envs 512 1024 --steps 1000

每个 batch size 单独 jit 编译（编译时间不计入吞吐）。
报告物理步吞吐（envs x steps / 墙钟）与等效策略步吞吐（/FRAME_SKIP=10）。
初始状态用 keyframe home 站立姿势 + ctrl=q_default，让接触求解器真实工作，
比悬空零力矩更贴近训练负载。参考：旧 CPU 20env 约 140 策略步/s。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

FRAME_SKIP = 10
Q_DEFAULT = np.array([0.0, 0.9, -1.8] * 4)          # 站立位目标角
SCENE = Path(__file__).resolve().parent.parent / "envs" / "scene_mjx.xml"


def make_batch(m: mujoco.MjModel, n_envs: int) -> mjx.Data:
    """CPU 端 keyframe home 初始化 + 站立 ctrl，put_data 上设备后广播成批次

    注意不能用 mjx.make_data(mx)：它对 condim=1 接触对（本模型含）+
    椭圆锥直接 NotImplementedError，而 put_data 路径无此限制（运行时正常）。
    """
    d0 = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0, "keyframe home 缺失"
    mujoco.mj_resetDataKeyframe(m, d0, key_id)
    d0.ctrl[:] = Q_DEFAULT                     # 接触求解器真实工作，贴近训练负载
    mujoco.mj_forward(m, d0)
    dx0 = mjx.put_data(m, d0)
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_envs,) + x.shape).copy(), dx0)


def bench(m: mujoco.MjModel, mx: mjx.Model, n_envs: int, steps: int) -> None:
    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    dx = make_batch(m, n_envs)

    t_compile0 = time.perf_counter()
    dx = step(mx, dx)                                        # 触发 jit 编译
    jax.block_until_ready(dx)
    t_compile = time.perf_counter() - t_compile0

    dx = make_batch(m, n_envs)                               # 重置回站立，正式计时
    t0 = time.perf_counter()
    for _ in range(steps):
        dx = step(mx, dx)
    jax.block_until_ready(dx)
    dt = time.perf_counter() - t0

    has_nan = bool(jnp.isnan(dx.qpos).any())
    phys_sps = n_envs * steps / dt
    print(f"envs={n_envs:5d}  steps={steps}  编译 {t_compile:6.1f}s  "
          f"墙钟 {dt:6.2f}s  物理步 {phys_sps:12,.0f}/s  "
          f"策略步 {phys_sps / FRAME_SKIP:10,.0f}/s  NaN={has_nan}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--steps", type=int, default=500,
                    help="每个 batch size 计时的物理步数（=50 策略步）")
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(f"JAX 设备: {dev.platform} / {getattr(dev, 'device_kind', '?')}")

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    mx = mjx.put_model(m)
    for n in args.envs:
        bench(m, mx, n, args.steps)


if __name__ == "__main__":
    main()
