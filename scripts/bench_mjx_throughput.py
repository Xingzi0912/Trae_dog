"""MJX GPU 吞吐冒烟 + 模型级 DR「治本」探针（4090 服务器）

用法：
    # 纯吞吐冒烟（无 DR，const 模式）
    python scripts/bench_mjx_throughput.py --envs 1024 2048 --steps 200 --modes const

    # 模型级 DR 写法对比（2026-10-13 性能定位，见 project_memory）
    python scripts/bench_mjx_throughput.py --envs 2048 --steps 60 \
        --modes const shared perenv kpool --ks 8 32 128 512 2048

每个 mode 单独 jit 编译（编译时间不计入吞吐）。报告物理步吞吐
（envs x steps / 墙钟）与等效策略步吞吐（/FRAME_SKIP=10）。
初始状态用 keyframe home 站立姿势 + ctrl=q_default，让接触求解器真实工作，
比悬空零力矩更贴近训练负载。参考：旧 CPU 20env 约 140 策略步/s。

--modes：同一批 env、同一份初态，只改「模型级 DR 怎么注入」——
    const   共享非批化模型 vmap(mjx.step,(None,0))，模型为闭包常量、无 DR
            —— 快路径基准（N=2048 实测 ≈98ms/20963 env·子步,s）
    shared  jit 内 apply_dr(mx, 标量 dr) → **非批化但 traced** 的单模型，再
            vmap 广播 —— 即当前仓库重构后的训练结构。**本探针的主判据**：
            若 ≈const ⇒ 重构保住快路径（DR 多样度退化为 1）；若 ≫const ⇒
            连「traced 非批化模型」都掉慢路径，需编译期常量模型
    perenv  vmap 内 apply_dr → 模型按 env 全批化（仓库旧写法 / 标准写法）
            —— N 份独立 DR，实测 ≈3103ms/660（慢 31 倍）
    kpool   vmap(vmap(mjx.step,(None,0)),(0,0))：外层 K 份池模型、池内共享
            —— 治本候选；K=N 即退化成 perenv，K=1 即 const。判据：若
            ≈const(97ms) ⇒ 在现有 MJX 后端内可把 DR 多样度恢复到 K 份

---- 结论速查（2026-10-13 probe_batch，N=2048）----
    A 共享非批化 const       : 97.7ms / 20963 env·物理子步每秒
    B 全批化 (0,0)           : 3057ms / 670   （标准写法=死路）
    C 仓库旧混批 _apply_dr   : 3103ms / 660
    ⇒ 只要模型按 env 批化，快路径必失效，且与写法无关。
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

from envs.dog_env_mjx import DogEnvMJX  # noqa: E402

FRAME_SKIP = 10
Q_DEFAULT = np.array([0.0, 0.9, -1.8] * 4)          # 站立位目标角
SCENE = ROOT / "envs" / "scene_mjx.xml"


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


def make_dr_pool(env: DogEnvMJX, k: int, seed: int = 0):
    """采样 k 份标量模型级 DR → DRParams（叶子形状 (k,)）

    用 vmap 驱动 sample_model_dr，保证每个 lane 拿到独立 key（比直接塞批化
    key 更稳妥，避免依赖 jax.random 对 key 批维的隐式语义）。
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), k)
    return jax.vmap(env.sample_model_dr)(keys)


def _build_fn(env, mx, n_envs, mode, k):
    """返回 (jitted_fn, args, note)：fn(*args) 一次完整前向"""
    if mode == "const":
        fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        return fn, (mx, None), "无 DR / 闭包常量模型（快路径基准）"

    if mode == "shared":
        dr = env.sample_model_dr(jax.random.PRNGKey(0))   # 标量
        fn = jax.jit(lambda d, x: jax.vmap(mjx.step, in_axes=(None, 0))(
            env._apply_dr(mx, d), x))
        return fn, (dr, None), "标量 DR→traced 非批化模型（=重构后训练结构）"

    if mode == "perenv":
        drs = make_dr_pool(env, n_envs)
        fn = jax.jit(lambda d, x: jax.vmap(mjx.step, in_axes=(0, 0))(
            jax.vmap(env._apply_dr, in_axes=(None, 0))(mx, d), x))
        return fn, (drs, None), f"N={n_envs} 份独立 DR（全批化）"

    if mode == "kpool":
        assert n_envs % k == 0, f"n_envs={n_envs} 不能被 k={k} 整除"
        drs = make_dr_pool(env, k)
        inner = jax.vmap(mjx.step, in_axes=(None, 0))
        fn = jax.jit(lambda d, x: jax.vmap(inner, in_axes=(0, 0))(
            jax.vmap(env._apply_dr, in_axes=(None, 0))(mx, d), x))
        return fn, (drs, None), f"K={k} 份池模型、池内 {n_envs // k} env 共享"

    raise ValueError(mode)


def bench(env, m, mx, n_envs, steps, mode, k) -> None:
    fn, (model_arg, _), note = _build_fn(env, mx, n_envs, mode, k)

    if mode == "const":
        args = (mx, make_batch(m, n_envs))
    elif mode == "kpool":
        dx = make_batch(m, n_envs)
        dK = jax.tree_util.tree_map(
            lambda x: x.reshape((k, n_envs // k) + x.shape[1:]), dx)
        args = (model_arg, dK)
    else:
        args = (model_arg, make_batch(m, n_envs))

    t0 = time.perf_counter()
    out = fn(*args)                                          # 触发 jit 编译
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()                                 # 正式计时
    for _ in range(steps):
        out = fn(*args)
    jax.block_until_ready(out)
    dt = time.perf_counter() - t0

    has_nan = bool(jnp.isnan(out.qpos).any())
    per_call_ms = dt / steps * 1e3
    phys_sps = n_envs * steps / dt
    tag = f"K={k}" if mode == "kpool" else ""
    print(f"envs={n_envs:5d} {mode:7s}{tag:8s} steps={steps:4d}  "
          f"编译 {t_compile:6.1f}s  单次 {per_call_ms:9.2f}ms  "
          f"物理步 {phys_sps:11,.0f}/s  策略步 {phys_sps / FRAME_SKIP:9,.0f}/s  "
          f"NaN={has_nan}")
    print(f"         └ {note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--steps", type=int, default=100,
                    help="每个 mode 计时的物理步数")
    ap.add_argument("--modes", nargs="+",
                    default=["const", "shared", "perenv"],
                    choices=["const", "shared", "perenv", "kpool"])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128, 512, 2048],
                    help="仅 kpool 模式使用：池数 K 列表")
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(f"JAX 设备: {dev.platform} / {getattr(dev, 'device_kind', '?')}")

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    mx = mjx.put_model(m)
    env = DogEnvMJX(n_envs=1)      # 仅借其 _apply_dr / sample_model_dr

    for n in args.envs:
        for mode in args.modes:
            if mode == "kpool":
                for k in args.ks:
                    if n % k:
                        print(f"跳过 kpool K={k}（n_envs={n} 不整除）")
                        continue
                    bench(env, m, mx, n, args.steps, mode, k)
            else:
                bench(env, m, mx, n, args.steps, mode, 0)


if __name__ == "__main__":
    main()
