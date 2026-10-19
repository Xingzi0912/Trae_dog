"""加载 JAX/Flax checkpoint，渲染 MJX 策略在标称环境（关 DR）下的行走视频

与 scripts/render_policy.py（旧 PyTorch 版）对应，但全 JAX/Flax + MJX。

用法（在服务器项目根目录执行）：
    python scripts/render_jax.py --video --steps 1000
    python scripts/render_jax.py --video --steps 1000 --ckpt checkpoints/dog_cmd_mjx_teacher.pkl
    python scripts/render_jax.py --video --steps 1000 --ckpt checkpoints/dog_cmd_mjx_teacher.pkl --out logs/u37.mp4

视频生成后 scp 回本地：
    scp ubuntu@<server>:/home/ubuntu/Sxy_bigdog/Trae_dog/logs/dog_walk.mp4 .

渲染原理（与训练解耦）：
  1. 创建 n_envs=1 的 DogEnvMJX，关 DR、关噪声
  2. 加载 pickle checkpoint（params + obs_rms）
  3. 每步：obs_rms 归一化 → Actor.apply 取 μ（确定性）→ env.step
  4. MJX state 用 mjx.get_data 转回 mujoco.MjData，Renderer 每帧 render
  5. matplotlib FuncAnimation 合成 mp4（无 ffmpeg 则 gif）
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import mjx
import mujoco
import numpy as np
from flax.training.train_state import TrainState
import optax

from envs.dog_env_mjx import DogEnvMJX, EnvConfig
from algos.networks_jax import make_teacher_actor


def load_ckpt(ckpt_path: str):
    """返回 (actor_params, obs_rms_dict)；obs_rms_dict 可能为 None（学生没有）"""
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    return ck["params"], ck.get("obs_rms")


def make_deterministic_actor_apply(actor):
    """JIT 编译一个只返回 μ 的前向（忽略 log_std、critic）"""

    @jax.jit
    def apply(params, obs):
        mu, _std = actor.apply(params, obs)
        return mu

    return apply


def run_offscreen(env, actor_apply, params, obs_rms_dict,
                  n_steps: int, out_path: Path, seed: int = 0):
    """离屏渲染，matplotlib 存 mp4 或 gif"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        print("  发现 imageio-ffmpeg，用 mp4 输出")
    except ImportError:
        print("  未发现 imageio-ffmpeg，将输出 gif")

    # --- MuJoCo 渲染器 ---
    mj_model = mujoco.MjModel.from_xml_path(env.model_xml)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.2
    cam.elevation = -15
    cam.azimuth = 90

    # --- 环境 reset ---
    rng = jax.random.PRNGKey(seed)
    mjx_model = mjx.put_model(mj_model)
    state, obs = env.reset(rng)  # (1, 99)

    # --- 单步 jit（减少开销）---
    @jax.jit
    def single_step(state, action):
        return env.step(state, action, mx_model=mjx_model)

    # --- obs 归一化 ---
    if obs_rms_dict is not None:
        obs_mean = jnp.asarray(obs_rms_dict["mean"])
        obs_var = jnp.asarray(obs_rms_dict["var"])
    else:
        obs_mean = jnp.zeros_like(obs[0])
        obs_var = jnp.ones_like(obs[0])
        print("  ⚠️ obs_rms 缺失，未做观测归一化（学生 checkpoint 正常）")

    # --- 渲染循环 ---
    frames = []
    total_reward = 0.0
    step = 0
    done = False

    for step in range(1, n_steps + 1):
        # 归一化（99 维）→ Actor μ → clip
        obs_n = (obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        mu = actor_apply(params["actor"]["params"], obs_n)  # (1, 12)
        action = jnp.clip(mu, -1.0, 1.0)

        state, obs, reward, term, trunc, info = single_step(state, action)
        total_reward += float(reward[0])

        # MJX state → mujoco data → render
        mj_data = mjx.get_data(mj_model, state.mjx_data)
        cam.lookat[:] = [mj_data.qpos[0], mj_data.qpos[1], 0.25]
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render())

        if bool(term[0]) or bool(trunc[0]):
            print(f"  提前结束 at step {step}（term={bool(term[0])}, trunc={bool(trunc[0])}）")
            done = True
            break

    renderer.close()

    # --- 合成视频 ---
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axis("off")
    im = ax.imshow(frames[0])
    plt.tight_layout(pad=0)

    def draw(i):
        im.set_array(frames[i])
        return (im,)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=20)  # 50Hz
    if FFMpegWriter.isAvailable():
        out = out_path.with_suffix(".mp4")
        anim.save(out, writer=FFMpegWriter(fps=50, bitrate=2000))
    else:
        out = out_path.with_suffix(".gif")
        anim.save(out, writer=PillowWriter(fps=50))
    plt.close(fig)

    # --- 总结 ---
    mj_data = mjx.get_data(mj_model, state.mjx_data)
    x, y = float(mj_data.qpos[0]), float(mj_data.qpos[1])
    print()
    print(f"  视频已保存: {out}（{len(frames)} 帧）")
    print(f"  步数 {step} | 累计回报 {total_reward:7.1f}")
    print(f"  前进位移 {x:6.2f} m | 横向偏移 {y:5.2f} m | "
          f"平均速度 {x / max(step, 1) * 50:4.2f} m/s")
    print(f"  最终 base_z = {float(mj_data.qpos[2]):.3f} m")
    print(f"  done={done}")


def main():
    ap = argparse.ArgumentParser(description="JAX/Flax + MJX 离屏渲染")
    ap.add_argument("--ckpt", default="checkpoints/dog_cmd_mjx_teacher.pkl",
                    help="Flax checkpoint（params + obs_rms）")
    ap.add_argument("--steps", type=int, default=1000,
                    help="渲染步数（默认 1000 即 20s @50Hz）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/dog_walk",
                    help="视频输出路径（不含扩展名，自动 .mp4 或 .gif）")
    args = ap.parse_args()

    # --- checkpoint ---
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"❌ checkpoint 不存在：{ckpt_path}")
        print(f"   服务器上查一下：ls /home/ubuntu/Sxy_bigdog/Trae_dog/checkpoints/")
        sys.exit(1)
    params, obs_rms_dict = load_ckpt(str(ckpt_path))
    print(f"checkpoint: {ckpt_path}")
    print(f"  obs_rms: {'有' if obs_rms_dict is not None else '无（学生 checkpoint）'}")

    # --- 环境（n_envs=1，关 DR 关噪声）---
    env_cfg = EnvConfig(
        n_envs=1,
        randomize=False,           # 标称环境，关 DR
        obs_noise_scale=0.0,       # 关观测噪声
        obs_delay_steps=0,         # 关延迟
    )
    env = DogEnvMJX(env_cfg)
    print(f"env: n_envs=1, DR=关, noise=0, delay=0")

    # --- Actor ---
    actor = make_teacher_actor(action_dim=12)
    actor_apply = make_deterministic_actor_apply(actor)

    # --- 输出路径 ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_offscreen(env, actor_apply, params, obs_rms_dict,
                  args.steps, out_path, args.seed)


if __name__ == "__main__":
    main()
