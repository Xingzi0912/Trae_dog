"""加载训好的 checkpoint，渲染策略在【标称环境（关 DR）】下的行走

肉眼验收最后一关：确认步态自然、无贴地滑行/乱摆蹭分等刷分行为。

用法（在项目根目录执行）：
    python scripts/render_policy.py                  # 交互式 viewer（需显示器/X11）
    python scripts/render_policy.py --video          # 离屏渲染，存 logs/dog_walk.gif
    python scripts/render_policy.py --video --steps 1000
    python scripts/render_policy.py --ckpt checkpoints/dog_ppo_final.pt

参数：
    --ckpt PATH   checkpoint（默认 checkpoints/dog_ppo_best.pt）
    --steps N     渲染步数（默认 1000，即完整一局 20s）
    --video       离屏模式：无窗口，输出视频到 logs/（无 ffmpeg 时存 GIF）
    --seed N      标称环境 reset seed（默认 0；关 DR 时轨迹确定）
"""

import argparse
import sys
from pathlib import Path

# 让脚本能在项目外被直接 python 调用（不需要安装包）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from envs.dog_env import DogEnv, RandConfig
from algos.networks import PolicyNetwork
from utils.normalizer import RunningMeanStd


def load_policy(ckpt_path: str, device):
    """checkpoint = {"actor": state_dict, "obs_rms": state_dict}"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = PolicyNetwork(state_dim=58, action_dim=12).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_rms = RunningMeanStd(shape=(58,))
    obs_rms.load_state_dict(ckpt["obs_rms"])
    return actor, obs_rms


def deterministic_action(actor, obs_rms, obs, device):
    """与 evaluate() 完全一致：obs_rms 只归一化不更新，取 μ 后 clip"""
    obs_n = obs_rms.normalize(obs)
    state_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu = actor(state_t).mean
    return mu.cpu().numpy().clip(-1.0, 1.0)


def run_interactive(actor, obs_rms, env, device, n_steps):
    import mujoco.viewer

    obs, _ = env.reset(seed=0)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        total = 0.0
        try:
            while viewer.is_running() and step < n_steps:
                action = deterministic_action(actor, obs_rms, obs, device)
                obs, reward, term, trunc, info = env.step(action)
                viewer.sync()
                total += reward
                step += 1
                if term or trunc:
                    print(f"  提前结束 at step {step}（term={term}）")
                    break
        except KeyboardInterrupt:
            print("\n手动退出")
    _print_summary(env, step, total)


def run_offscreen(actor, obs_rms, env, device, n_steps, out_path: Path):
    """离屏渲染 + 跟踪相机（镜头随狗的 xy 平移），matplotlib 存视频/GIF"""
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    # 优先用 imageio-ffmpeg 自带的 ffmpeg（无需系统安装、无需 PATH）
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.2
    cam.elevation = -15
    cam.azimuth = 90

    obs, _ = env.reset(seed=0)
    frames = []
    total = 0.0
    step = 0
    for step in range(1, n_steps + 1):
        action = deterministic_action(actor, obs_rms, obs, device)
        obs, reward, term, trunc, info = env.step(action)
        total += reward

        cam.lookat[:] = [env.data.qpos[0], env.data.qpos[1], 0.25]
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render())
        if term or trunc:
            print(f"  提前结束 at step {step}（term={term}）")
            break
    renderer.close()

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
    print(f"视频已保存: {out}（{len(frames)} 帧）")
    _print_summary(env, step, total)


def _print_summary(env, step, total):
    x, y = env.data.qpos[0], env.data.qpos[1]
    print()
    print(f"  步数 {step} | 累计回报 {total:7.1f}")
    print(f"  前进位移 {x:6.2f} m | 横向偏移 {y:5.2f} m | "
          f"平均速度 {x / max(step, 1) * 50:4.2f} m/s")
    print(f"  最终 base_z = {env.data.qpos[2]:.3f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/dog_ppo_best.pt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, obs_rms = load_policy(args.ckpt, device)

    # 标称环境：关 DR、关观测噪声/延迟，检验策略在"理想世界"的真实步态
    rand_cfg = RandConfig(enabled=False)
    render_mode = "rgb_array" if args.video else None
    env = DogEnv(rand_config=rand_cfg, render_mode=render_mode)

    print(f"设备: {device} | ckpt: {args.ckpt} | DR: 关（标称环境）")
    if args.video:
        Path("logs").mkdir(exist_ok=True)
        run_offscreen(actor, obs_rms, env, device, args.steps,
                      Path("logs/dog_walk"))
    else:
        run_interactive(actor, obs_rms, env, device, args.steps)
    env.close()


if __name__ == "__main__":
    main()
