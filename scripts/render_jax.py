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
import os
import pickle
import sys
from pathlib import Path

# 必须在 import jax 之前设置
# 公用服务器：JAX 默认占满所有 GPU 显存，这里限制到 10%（渲染不吃 GPU，
# 只是 MJX 仿真可能会 lazy init 一个 GPU context；给最小配额避免干扰他人）
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from flax.training.train_state import TrainState
import optax

from envs.dog_env_mjx import DogEnvMJX, EnvConfig
from algos.networks_jax import make_teacher_actor


def load_ckpt(ckpt_path: str):
    """返回 (actor_params, obs_rms)；obs_rms 是 RunningMeanStd NamedTuple（或 None）"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    return ck["params"], ck.get("obs_rms")


def make_deterministic_actor_apply(actor):
    """JIT 编译一个只返回 μ 的前向（忽略 log_std、critic）"""

    @jax.jit
    def apply(actor_params, obs):
        mu, _std = actor.apply(actor_params, obs)
        return mu

    return apply


def run_interactive(env, actor_apply, params, obs_rms_dict, seed: int = 0):
    """交互式键盘控制（需要本地显示器 + glfw）

    按键：
      W/S  前进/后退 (vx)
      A/D  左移/右移 (vy)
      Q/E  左转/右转 (yaw_rate)
      X    指令归零
      R    重置环境
      Esc  退出
    """
    import glfw

    scene_xml = str(Path(__file__).parent.parent / "envs" / "scene_mjx.xml")
    mj_model = mujoco.MjModel.from_xml_path(scene_xml)
    mjx_model = mjx.put_model(mj_model)

    if obs_rms_dict is not None:
        obs_mean = jnp.asarray(obs_rms_dict.mean)
        obs_var = jnp.asarray(obs_rms_dict.var)
    else:
        obs_mean = jnp.zeros(99)
        obs_var = jnp.ones(99)

    @jax.jit
    def single_step(state, action):
        return env.step(state, action, mx_model=mjx_model)

    cmd = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
    cmd_step = 0.15
    yaw_step = 0.5
    cmd_range = {"vx": (-0.5, 1.5), "vy": (-0.5, 0.5), "yaw": (-2.0, 2.0)}

    def clamp_cmd():
        for k, (lo, hi) in cmd_range.items():
            cmd[k] = max(lo, min(hi, cmd[k]))

    print("  W/S vx | A/D vy | Q/E yaw | X 归零 | R 重置 | Esc 退出")

    rng = jax.random.PRNGKey(seed)
    state, obs = env.reset(rng)

    glfw.init()
    win = glfw.create_window(800, 600, "Dog Interactive", None, None)
    glfw.make_context_current(win)

    def key_cb(_, k, sc, a, m):
        if a != glfw.PRESS:
            return
        if k == glfw.KEY_W:   cmd["vx"] += cmd_step
        elif k == glfw.KEY_S: cmd["vx"] -= cmd_step
        elif k == glfw.KEY_A: cmd["vy"] -= cmd_step
        elif k == glfw.KEY_D: cmd["vy"] += cmd_step
        elif k == glfw.KEY_Q: cmd["yaw"] += yaw_step
        elif k == glfw.KEY_E: cmd["yaw"] -= yaw_step
        elif k == glfw.KEY_X: cmd.update(vx=0.0, vy=0.0, yaw=0.0)
        elif k == glfw.KEY_R:
            nonlocal state, obs, rng
            rng, k2 = jax.random.split(rng)
            state, obs = env.reset(k2)
            print("  重置")
        elif k == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        clamp_cmd()

    glfw.set_key_callback(win, key_cb)

    renderer = mujoco.Renderer(mj_model, height=600, width=800)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.2
    cam.elevation = -15
    cam.azimuth = 90

    try:
        while not glfw.window_should_close(win):
            glfw.poll_events()

            # 拼特权信息 (61+38=99)
            priv = env.get_privileged_obs(state)  # (1, 38)
            teacher_obs = jnp.concatenate([obs, priv], axis=-1)
            # 覆盖指令槽（[58:61] 是标准 obs 里的指令，[99-3=96:99] 也是，两处都改）
            cmd_arr = jnp.array([[cmd["vx"], cmd["vy"], cmd["yaw"]]], dtype=jnp.float32)
            teacher_obs = teacher_obs.at[0, 58:61].set(cmd_arr[0])
            teacher_obs = teacher_obs.at[0, 96:99].set(cmd_arr[0])
            obs_n = (teacher_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
            action = jnp.clip(actor_apply(params["actor"], obs_n), -1.0, 1.0)

            state, obs, reward, term, trunc, info = single_step(state, action)
            if bool(term[0]) or bool(trunc[0]):
                rng, k2 = jax.random.split(rng)
                state, obs = env.reset(k2)

            mj_data = mjx.get_data(mj_model, state.dx)[0]
            cam.lookat[:] = [mj_data.qpos[0], mj_data.qpos[1], 0.25]
            renderer.update_scene(mj_data, camera=cam)
            renderer.render()

            glfw.swap_buffers(win)
            glfw.wait_events_timeout(0.02)

    finally:
        renderer.close()
        glfw.destroy_window(win)
        glfw.terminate()
        print("  退出")


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
    scene_xml = str(Path(__file__).parent.parent / "envs" / "scene_mjx.xml")
    mj_model = mujoco.MjModel.from_xml_path(scene_xml)
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
        obs_mean = jnp.asarray(obs_rms_dict.mean)
        obs_var = jnp.asarray(obs_rms_dict.var)
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
        # 拼特权 → 99 维 → 归一化 → Actor
        priv = env.get_privileged_obs(state)
        teacher_obs = jnp.concatenate([obs, priv], axis=-1)
        obs_n = (teacher_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        mu = actor_apply(params["actor"], obs_n)  # (1, 12)
        action = jnp.clip(mu, -1.0, 1.0)

        state, obs, reward, term, trunc, info = single_step(state, action)
        total_reward += float(reward[0])

        # MJX state → mujoco data → render
        mj_data = mjx.get_data(mj_model, state.dx)[0]
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
    mj_data = mjx.get_data(mj_model, state.dx)[0]
    x, y = float(mj_data.qpos[0]), float(mj_data.qpos[1])
    print()
    print(f"  视频已保存: {out}（{len(frames)} 帧）")
    print(f"  步数 {step} | 累计回报 {total_reward:7.1f}")
    print(f"  前进位移 {x:6.2f} m | 横向偏移 {y:5.2f} m | "
          f"平均速度 {x / max(step, 1) * 50:4.2f} m/s")
    print(f"  最终 base_z = {float(mj_data.qpos[2]):.3f} m")
    print(f"  done={done}")


def main():
    ap = argparse.ArgumentParser(description="JAX/Flax + MJX 渲染（离屏视频 或 交互控制）")
    ap.add_argument("--ckpt", default="checkpoints/dog_cmd_mjx_teacher.pkl")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/dog_walk")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="键盘控制模式（WASD/Q/E 改指令，需本地 glfw + 显示器）")
    ap.add_argument("--video", action="store_true", default=False,
                    help="离屏视频模式（默认）")
    args = ap.parse_args()

    if not args.interactive and not args.video:
        args.video = True  # 默认视频模式

    # --- checkpoint ---
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"❌ checkpoint 不存在：{ckpt_path}")
        print(f"   服务器上查一下：ls /home/ubuntu/Sxy_bigdog/Trae_dog/checkpoints/")
        sys.exit(1)
    params, obs_rms_dict = load_ckpt(str(ckpt_path))
    print(f"checkpoint: {ckpt_path}")
    print(f"  obs_rms: {'有' if obs_rms_dict is not None else '无（学生 checkpoint）'}")

    # --- 环境（n_envs=1，关 DR）---
    env_cfg = EnvConfig(
        dr_enabled=False,          # 标称环境，关 DR
    )
    env = DogEnvMJX(n_envs=1, config=env_cfg)
    print(f"env: n_envs=1, DR=关, model={env.cfg.action_scale}")

    # --- Actor ---
    actor = make_teacher_actor(action_dim=12)
    actor_apply = make_deterministic_actor_apply(actor)

    # --- 输出路径 ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        print("模式: 交互式键盘控制（-i）")
        run_interactive(env, actor_apply, params, obs_rms_dict, args.seed)
    else:
        print("模式: 离屏视频（--video）")
        run_offscreen(env, actor_apply, params, obs_rms_dict,
                      args.steps, out_path, args.seed)


if __name__ == "__main__":
    main()
