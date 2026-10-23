"""加载 JAX/Flax checkpoint，渲染 MJX 策略在标称环境（关 DR）下的行走视频

与 scripts/render_policy.py（旧 PyTorch 版）对应，但全 JAX/Flax + MJX。

用法（在服务器项目根目录执行）：
    python scripts/render_jax.py --video --steps 1000
    python scripts/render_jax.py --video --steps 1000 --ckpt checkpoints/dog_cmd_mjx_teacher.pkl
    python scripts/render_jax.py --video --steps 1000 --ckpt checkpoints/dog_cmd_mjx_teacher.pkl --out logs/u37.mp4
    # 本地交互式（需要显示器 + pygame）
    python scripts/render_jax.py -i --ckpt checkpoints/dog_cmd_mjx_teacher.pkl

视频生成后 scp 回本地：
    scp ubuntu@<server>:/home/ubuntu/Sxy_bigdog/Trae_dog/logs/dog_walk.mp4 .

渲染原理（与训练解耦）：
  1. 创建 n_envs=1 的 DogEnvMJX，关 DR、关噪声
  2. 加载 pickle checkpoint（params + obs_rms）
  3. 每步：obs_rms 归一化 → Actor.apply 取 μ（确定性）→ env.step
  4. MJX state 用 mjx.get_data 转回 mujoco.MjData，Renderer 每帧 render
  5. matplotlib FuncAnimation 合成 mp4（无 ffmpeg 则 gif）

交互式模式：
  pygame 窗口 + 键盘回调。Renderer 离屏 render → RGB numpy array →
  pygame.surfarray.make_surface → 贴到窗口。完全不依赖独立 GL context，不会黑屏。
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
# 无头服务器没有 DISPLAY：默认 glfw 需要 X11，改走 EGL 离屏渲染
# （若 EGL 不可用再改 osmesa：MUJOCO_GL=osmesa）
os.environ.setdefault("MUJOCO_GL", "egl")

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


# ---------------------------------------------------------------------------
# 指令调度与可视化辅助
# ---------------------------------------------------------------------------

def parse_cmd_schedule(s: str):
    """解析 "step:vx,vy,yaw; step:vx,vy,yaw; ..." 为 [(step, (vx,vy,yaw)), ...]"""
    if not s:
        return []
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        t_str, cmd_str = part.split(":")
        t = int(t_str.strip())
        vx, vy, yaw = [float(x) for x in cmd_str.split(",")]
        out.append((t, (vx, vy, yaw)))
    out.sort(key=lambda x: x[0])
    return out


def get_cmd_at_step(schedule, step, default=(0.0, 0.0, 0.0)):
    """根据调度表返回当前指令；无匹配则返回 default"""
    cmd = default
    for t, c in schedule:
        if step >= t:
            cmd = c
        else:
            break
    return cmd


def _arrow_geom_type():
    """mjGEOM_ARROW 在 MuJoCo 3.0+ 可用，旧版本 fallback 到 CYLINDER"""
    try:
        return mujoco.mjtGeom.mjGEOM_ARROW
    except AttributeError:
        return mujoco.mjtGeom.mjGEOM_CYLINDER


def add_cmd_arrow(renderer, base_pos, vx, vy, yaw):
    """向 renderer.scene 追加 3D 箭头 geom，表示速度指令方向（狗头顶上方）"""
    scn = renderer.scene
    if scn.ngeom >= scn.maxgeom - 1:
        return

    speed = np.sqrt(vx**2 + vy**2)
    # 箭头长度：0.15 ~ 0.5m，对应速度 0 ~ 1.5 m/s
    arrow_len = 0.15 + min(speed / 1.5, 1.0) * 0.35

    g = scn.geoms[scn.ngeom]
    pos = np.array([base_pos[0], base_pos[1], base_pos[2] + 0.5])

    if speed > 0.01:
        # 方向：水平指向 (vx, vy)
        dir_vec = np.array([vx, vy, 0.0])
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)

        # 构造旋转矩阵：局部 +z 指向 dir_vec
        z_axis = dir_vec
        x_axis = np.cross([0, 0, 1], z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1.0, 0, 0])
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        mat = np.array([x_axis, y_axis, z_axis]).T  # (3,3)

        g.type = _arrow_geom_type()
        g.pos[:] = pos
        g.mat[:] = mat
        g.size[0] = 0.03          # width / radius
        g.size[1] = arrow_len     # length
        g.size[2] = 0.0
        g.rgba[:] = [0.0, 1.0, 0.0, 1.0]   # 绿色
    else:
        # 零速：灰色小圆点
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.pos[:] = pos
        g.mat[:] = np.eye(3)
        g.size[0] = 0.05
        g.size[1] = 0.0
        g.size[2] = 0.0
        g.rgba[:] = [0.5, 0.5, 0.5, 0.8]

    # 清空可能干扰的字段
    g.dataid = -1
    g.objtype = 0
    g.objid = -1
    g.category = 0
    g.texid = -1
    g.emission = 0.0
    g.specular = 0.5
    g.shininess = 0.5
    g.reflectance = 0.0
    g.label = b""
    scn.ngeom += 1


def draw_cmd_hud(frame, cmd):
    """用 PIL 在帧左上角画指令 HUD（速度向量圆盘 + 数值）"""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")

    vx, vy, yaw = cmd

    # 半透明黑底面板
    hud_x, hud_y = 10, 10
    hud_w, hud_h = 160, 120
    draw.rectangle(
        [hud_x, hud_y, hud_x + hud_w, hud_y + hud_h],
        fill=(0, 0, 0, 180),
        outline=(0, 255, 0, 255),
        width=1,
    )

    # 标题
    draw.text((hud_x + 5, hud_y + 5), "CMD", fill=(0, 255, 0))

    # 速度向量小圆盘
    cx, cy = hud_x + 80, hud_y + 55
    r = 30
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(100, 100, 100), width=1)
    draw.line([cx - r, cy, cx + r, cy], fill=(80, 80, 80), width=1)
    draw.line([cx, cy - r, cx, cy + r], fill=(80, 80, 80), width=1)

    speed = np.sqrt(vx**2 + vy**2)
    if speed > 0.01:
        max_s = 1.5
        scale = min(speed / max_s, 1.0) * (r - 3)
        angle = np.arctan2(vy, vx)
        ex = cx + scale * np.cos(angle)
        ey = cy - scale * np.sin(angle)  # PIL y 向下
        draw.line([cx, cy, ex, ey], fill=(0, 255, 0), width=2)
        draw.ellipse([ex - 2, ey - 2, ex + 2, ey + 2], fill=(0, 255, 0))

    # 数值
    draw.text(
        (hud_x + 5, hud_y + 95),
        f"vx:{vx:+.2f} vy:{vy:+.2f} yaw:{yaw:+.2f}",
        fill=(255, 255, 255),
    )

    return np.array(img)


def load_ckpt(ckpt_path: str):
    """返回 (actor_params, obs_rms)；obs_rms 是 RunningMeanStd NamedTuple（或 None）"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    return ck["params"], ck.get("obs_rms")


def make_actor_apply(actor, stochastic: bool = False):
    """JIT 前向。stochastic=False 返回确定性 μ；True 返回 μ+σ·ε（训练同款采样）。
    两个分支统一签名 (params, obs, key) -> (act, std)。"""

    if stochastic:
        @jax.jit
        def apply(actor_params, obs, key):
            mu, std = actor.apply(actor_params, obs)
            raw = mu + std * jax.random.normal(key, mu.shape)
            return raw, std
        return apply
    else:
        @jax.jit
        def apply(actor_params, obs, key):
            mu, std = actor.apply(actor_params, obs)
            return mu, std
        return apply


def run_interactive(env, actor_apply, params, obs_rms_dict, seed: int = 0):
    """交互式键盘控制（需要本地显示器 + pygame）

    按键：
      W/S  前进/后退 (vx)
      A/D  左移/右移 (vy)
      Q/E  左转/右转 (yaw_rate)
      X    指令归零
      R    重置环境
      Esc  退出
    """
    import pygame
    from pygame import surfarray

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

    # --- pygame 窗口 ---
    pygame.init()
    WIDTH, HEIGHT = 640, 480
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dog Interactive")
    clock = pygame.time.Clock()

    # --- mujoco Renderer（离屏，完全不碰窗口）---
    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.2
    cam.elevation = -15
    cam.azimuth = 90

    running = True
    while running:
        # --- 键盘 ---
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_w:     cmd["vx"] += cmd_step
                elif ev.key == pygame.K_s:   cmd["vx"] -= cmd_step
                elif ev.key == pygame.K_a:   cmd["vy"] -= cmd_step
                elif ev.key == pygame.K_d:   cmd["vy"] += cmd_step
                elif ev.key == pygame.K_q:   cmd["yaw"] += yaw_step
                elif ev.key == pygame.K_e:   cmd["yaw"] -= yaw_step
                elif ev.key == pygame.K_x:   cmd.update(vx=0.0, vy=0.0, yaw=0.0)
                elif ev.key == pygame.K_r:
                    rng, k2 = jax.random.split(rng)
                    state, obs = env.reset(k2)
                    print("  重置")
                elif ev.key == pygame.K_ESCAPE:
                    running = False
                clamp_cmd()

        # --- Actor step ---
        # 强制 env 内部指令 = 键盘指令（残差步态基准由 state.command 驱动）
        cmd_arr = jnp.array([[cmd["vx"], cmd["vy"], cmd["yaw"]]], dtype=jnp.float32)
        state = state._replace(command=cmd_arr)
        priv = env.get_privileged_obs(state)
        teacher_obs = jnp.concatenate([obs, priv], axis=-1)
        # 覆盖指令位：仅 obs[58:61]；priv(38 维) 不含指令副本，
        # 误写 96:99 会覆盖 质量乘子/base_x/base_y（2026-10-22 实锤 bug）
        teacher_obs = teacher_obs.at[0, 58:61].set(cmd_arr[0])
        obs_n = (teacher_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        rng, k_act = jax.random.split(rng)
        act, _std = actor_apply(params["actor"], obs_n, k_act)
        action = jnp.clip(act, -1.0, 1.0)

        state, obs, reward, term, trunc, info = single_step(state, action)
        if bool(term[0]) or bool(trunc[0]):
            rng, k2 = jax.random.split(rng)
            state, obs = env.reset(k2)

        # --- MJX → MjData → 离屏 render ---
        mj_data = mjx.get_data(mj_model, state.dx)[0]
        cam.lookat[:] = [mj_data.qpos[0], mj_data.qpos[1], 0.25]
        renderer.update_scene(mj_data, camera=cam)
        rgb = renderer.render()  # numpy (H, W, 3) uint8

        # --- numpy → pygame surface → 贴窗口 ---
        # pygame.surfarray 期望 (W, H, 3) + 颜色顺序兼容（pygame 用 BGR）
        # transpose (H,W,3) → (W,H,3)；flip 最后一维做 RGB→BGR
        rgb_w = np.transpose(rgb, (1, 0, 2))  # (W, H, 3)
        rgb_bgr = rgb_w[:, :, ::-1].copy()     # RGB → BGR，复制成连续内存
        surf = surfarray.make_surface(rgb_bgr)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(50)  # ~50Hz

    try:
        renderer.close()
    except Exception:
        pass
    pygame.quit()
    print("  退出")


def run_offscreen(env, actor_apply, params, obs_rms_dict,
                  n_steps: int, out_path: Path, seed: int = 0,
                  cmd_schedule=None, stochastic: bool = False):
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
        # 教师输入是 99 维（61 标准 + 38 特权），不能用 obs(61) 的 shape
        # 兜底，否则下面的归一化会 broadcast 失败。
        n_teacher = env.OBS_DIM + env.PRIV_DIM
        obs_mean = jnp.zeros(n_teacher)
        obs_var = jnp.ones(n_teacher)
        print("  ⚠️ checkpoint 缺 obs_rms，未做观测归一化（教师策略输出不可信）")

    # --- 渲染循环 ---
    frames = []
    total_reward = 0.0
    step = 0
    done = False
    current_cmd = (0.0, 0.0, 0.0)

    for step in range(1, n_steps + 1):
        # 指令调度
        if cmd_schedule:
            current_cmd = get_cmd_at_step(cmd_schedule, step)

        # 强制 env 内部指令 = 调度指令（残差步态的基准由它驱动；
        # 训练时 obs 指令与内部 command 天然一致，渲染必须手动对齐）
        cmd_arr = jnp.array([current_cmd], dtype=jnp.float32)
        state = state._replace(command=cmd_arr)

        # 拼特权 → 99 维 → 归一化 → Actor
        priv = env.get_privileged_obs(state)
        teacher_obs = jnp.concatenate([obs, priv], axis=-1)

        # 覆盖指令位：仅 obs[58:61]；priv 无指令副本，勿动 96:99
        # （那是 质量乘子/base_x/base_y，写错会导致 OOD 漂移/摔倒）
        teacher_obs = teacher_obs.at[0, 58:61].set(cmd_arr[0])

        obs_n = (teacher_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        rng, k_act = jax.random.split(rng)
        act, std = actor_apply(params["actor"], obs_n, k_act)
        action = jnp.clip(act, -1.0, 1.0)

        if step % 100 == 0 or step == 1:
            print(f"  step {step}: |mu|mean={float(jnp.abs(act).mean()):.3f} "
                  f"mu_max={float(jnp.abs(act).max()):.3f} "
                  f"sigma={float(std.mean()):.3f}")

        state, obs, reward, term, trunc, info = single_step(state, action)
        total_reward += float(reward[0])

        # MJX state → mujoco data → render
        mj_data = mjx.get_data(mj_model, state.dx)[0]
        cam.lookat[:] = [mj_data.qpos[0], mj_data.qpos[1], 0.25]
        renderer.update_scene(mj_data, camera=cam)

        # 3D 箭头：狗头顶显示当前指令方向
        add_cmd_arrow(renderer, mj_data.qpos[0:3], *current_cmd)

        rgb = renderer.render()

        # 2D HUD：左上角数值面板
        rgb = draw_cmd_hud(rgb, current_cmd)
        frames.append(rgb)

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
                    help="键盘控制模式（WASD/Q/E 改指令，需本地 pygame + 显示器）")
    ap.add_argument("--video", action="store_true", default=False,
                    help="离屏视频模式（默认）")
    ap.add_argument("--cmds", type=str, default="",
                    help='指令调度 "step:vx,vy,yaw; step:vx,vy,yaw; ..."，'
                         '例：--cmds "0:0,0,0; 200:1.0,0,0; 500:0,0.4,0"')
    ap.add_argument("--stochastic", action="store_true",
                    help="用 μ+σ·ε 采样动作（训练 rollout 同款），对照确定性 μ")
    ap.add_argument("--gait_mode", type=str, default="sine",
                    choices=["sine", "ik"],
                    help="基准步态实现，必须与训练该 checkpoint 时一致")
    args = ap.parse_args()

    if not args.interactive and not args.video:
        args.video = True  # 默认视频模式

    cmd_schedule = parse_cmd_schedule(args.cmds)
    if cmd_schedule:
        print(f"指令调度: {cmd_schedule}")

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
    # 2026-10-22 实锤：use_residual=True 时基准步态吃 env 内部 state.command，
    # 与注入 obs[58:61] 的验收指令无关 → 必须 (a) 关内部重采样 (b) 每步强制
    # state.command=调度指令 (c) 关 max_steps 截断（验收要连续跑 1600 步）
    env_cfg = EnvConfig(
        dr_enabled=False,                    # 标称环境，关 DR
        max_steps=10**9,                     # 关 episode 截断
        cmd_resample_steps=(10**9, 10**9),   # 关内部指令重采样
        gait_mode=args.gait_mode,             # 必须与训练时一致
    )
    env = DogEnvMJX(n_envs=1, config=env_cfg)
    print(f"env: n_envs=1, DR=关, gait={env.cfg.gait_mode}")

    # --- Actor ---
    actor = make_teacher_actor(action_dim=12)
    actor_apply = make_actor_apply(actor, stochastic=args.stochastic)
    print(f"actor: {'随机采样 μ+σ·ε（训练同款）' if args.stochastic else '确定性 μ'}")

    # --- 输出路径 ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        print("模式: 交互式键盘控制（-i）")
        # 交互式需要 pygame
        try:
            import pygame  # noqa: F401
        except ImportError:
            print("❌ 缺 pygame，装一下：pip install pygame")
            sys.exit(1)
        run_interactive(env, actor_apply, params, obs_rms_dict, args.seed)
    else:
        print("模式: 离屏视频（--video）")
        run_offscreen(env, actor_apply, params, obs_rms_dict,
                      args.steps, out_path, args.seed,
                      cmd_schedule=cmd_schedule, stochastic=args.stochastic)


if __name__ == "__main__":
    main()
