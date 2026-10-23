"""训练入口：两阶段师生训练（PPO 训教师 → 蒸馏学生）

阶段 1：PPO 训教师（99 维输入 = 61 标准观测 + 38 特权信息）
阶段 2：冻结教师，蒸馏学生（61 维输入拟合教师动作+价值）

用法（本地冒烟，CPU）：
    python train_jax.py --steps 10000 --n_envs 16

用法（4090 服务器，正式训练）：
    # 先设置 GPU 显存（公用服务器限 40%）
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
    # tmux 内运行
    python -u train_jax.py --steps 20000000 --n_envs 2048 | tee logs/run.log

输出：
    checkpoints/dog_cmd_mjx_teacher.pkl   教师 Flax checkpoint
    checkpoints/dog_cmd_mjx_student.pkl   学生 Flax checkpoint
    checkpoints/dog_cmd_mjx_student.npz  学生部署包（numpy，NUC 推理）
    logs/dog_cmd_mjx_curve.png           学习曲线
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# 必须在 import jax 之后、创建任何数组之前设置
# 公用 4090 服务器限 40% 显存（避免影响他人训练任务）
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

from envs.dog_env_mjx import DogEnvMJX, EnvConfig
from algos.ppo_jax import PPOConfig, train as train_teacher
from algos.distill_jax import DistillConfig, train_student
from algos.networks_jax import make_student


def log(msg):
    """带时间戳的日志，强制 flush（后台运行 + tee 也能实时看到）"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


def save_flax_checkpoint(ts, obs_rms, path):
    """保存 Flax train_state + obs_rms（续训用）"""
    CKPT_DIR.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": ts.params, "obs_rms": obs_rms}, f)
    log(f"  checkpoint 已保存：{path}")


def save_student_npz(ts, obs_rms, path):
    """保存学生网络为 numpy .npz（部署用，NUC 端 numpy 推理）

    只存 Actor 的 mu head 权重（推理用确定性动作，不需要 std）。
    obs_rms 的 mean/var 也存进去，部署时做观测归一化。
    """
    CKPT_DIR.mkdir(exist_ok=True)
    p = ts.params["params"]
    save_dict = {
        "actor_w1": np.asarray(p["Dense_0"]["kernel"]),
        "actor_b1": np.asarray(p["Dense_0"]["bias"]),
        "actor_w2": np.asarray(p["Dense_1"]["kernel"]),
        "actor_b2": np.asarray(p["Dense_1"]["bias"]),
        "actor_w3": np.asarray(p["Dense_2"]["kernel"]),
        "actor_b3": np.asarray(p["Dense_2"]["bias"]),
        "obs_mean": np.asarray(obs_rms.mean),
        "obs_var": np.asarray(obs_rms.var),
    }
    np.savez(path, **save_dict)
    log(f"  部署包已保存：{path}")


def plot_curves(teacher_metrics, student_metrics):
    """绘制学习曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LOG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(teacher_metrics["update"], teacher_metrics["reward"])
    axes[0, 0].set_title("Teacher: Episode Reward")
    axes[0, 0].set_xlabel("update")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(teacher_metrics["update"], teacher_metrics["entropy"],
                    label="entropy")
    axes[0, 1].plot(teacher_metrics["update"], teacher_metrics["kl"],
                    label="kl")
    axes[0, 1].set_title("Teacher: Entropy & KL")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(student_metrics["update"],
                    student_metrics["action_loss"])
    axes[1, 0].set_title("Student: Action Distill Loss")
    axes[1, 0].set_xlabel("update")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(student_metrics["update"],
                    student_metrics["value_loss"])
    axes[1, 1].set_title("Student: Value Distill Loss")
    axes[1, 1].set_xlabel("update")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    out = LOG_DIR / "dog_cmd_mjx_curve.png"
    fig.savefig(out, dpi=110)
    log(f"学习曲线已保存：{out}")


def main():
    parser = argparse.ArgumentParser(description="两阶段师生训练")
    parser.add_argument("--steps", type=int, default=20_000_000,
                        help="总训练步数（默认 20M）")
    parser.add_argument("--n_envs", type=int, default=None,
                        help="并行环境数（默认：GPU=2048, CPU=16）")
    parser.add_argument("--rollout_len", type=int, default=24,
                        help="每个 env 每轮收集的步数")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="PPO 每 update 的 epoch 数（legged_gym 默认 5）")
    parser.add_argument("--n_minibatches", type=int, default=4,
                        help="每个 epoch 的 minibatch 数（legged_gym 默认 4）")
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="固定学习率（v19 回 PureJaxRL 标准 2.5e-4：μ 头增益"
                             " 38×→1× 后等效操作点归位；v17 的 1e-5 是在错误"
                             "增益下压每步 KL 的代偿，与增益同属一个耦合变量）")
    parser.add_argument("--distill_steps", type=int, default=None,
                        help="蒸馏总步数（默认=教师 1/4）")
    parser.add_argument("--no_residual", action="store_true",
                        help="禁用残差模式（纯 PPO 从零学走路）")
    parser.add_argument("--gait_mode", type=str, default=None,
                        choices=["sine", "ik"],
                        help="基准步态实现：sine=v24 正弦（仅前进稳定）；"
                             "ik=v25 IK 全向步进。缺省=EnvConfig 默认 sine")
    parser.add_argument("--no_dr", action="store_true",
                        help="禁用域随机化")
    parser.add_argument("--dr_pools", type=int, default=256,
                        help="模型级 DR 池数 K：1=整批共享（最快，无 DR 多样度）；"
                             "K∈{128,256,512} 仍走 mjx 快路径（~1.2x），恢复 K 份 "
                             "DR 多样度。须整除 n_envs，否则回退 1（关 DR 时恒为 1）")
    parser.add_argument("--mem", type=float, default=None,
                        help="XLA 显存占比 0.0-1.0（覆盖环境变量）")
    parser.add_argument("--resume", action="store_true",
                        help="从 checkpoints/dog_cmd_mjx_teacher.pkl 续训"
                             "（Adam 动量不保存，前 1-2 update 步长偏保守）")
    parser.add_argument("--ent_coef", type=float, default=None,
                        help="熵系数（默认用 PPOConfig 的 0.003；"
                             "σ 噪声墙停滞时降为 0.001 加速 σ 收缩）")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="【已失效】mega-jit 路径不读取 config.target_kl，"
                             "仅 target_kl_mb 生效；保留仅为兼容旧命令")
    parser.add_argument("--target_kl_mb", type=float, default=None,
                        help="per-mb 累积 KL 冻结阈值（默认 PPOConfig 的 0.1；"
                             "v12 判决：actor 每 update 烧满 0.1 预算即被冻结"
                             "（16/16 update 触发、11/16 在 epoch 0），放宽至 0.2）")
    # --- 指令课程（2026-10-23 新增）：缺省 = EnvConfig 的全程范围 ---
    parser.add_argument("--cmd_vx", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="vx 指令范围（默认 -0.5 1.5）。kp=150 下 sine 基准在 "
                             "fwd>=1.0 会摔，建议第一轮用 0 0.5")
    parser.add_argument("--cmd_vy", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="vy 指令范围（默认 -0.5 0.5）")
    parser.add_argument("--cmd_yaw", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"), help="yaw_rate 指令范围（默认 -2.0 2.0）")
    parser.add_argument("--zero_cmd_prob", type=float, default=None,
                        help="零指令概率（默认 0.17）")
    parser.add_argument("--track_sigma2", type=float, default=None,
                        help="线速度跟踪高斯分母（默认 0.25 → σ=0.5 m/s）。"
                             "调小=逼策略精确跟速；注意会拉低 track 均值，"
                             "相对放大 energy/rp 等惩罚的权重")
    parser.add_argument("--yaw_sigma2", type=float, default=None,
                        help="yaw 跟踪高斯分母（默认 0.25 → σ=0.5 rad/s）。"
                             "注意：高斯在零点附近是平的，靠收紧 σ 压不住小幅持续漂移")
    parser.add_argument("--yaw_lin_scale", type=float, default=None,
                        help=">0 时 yaw 奖励改成线性：w_yaw*clip(1-|e|/scale,0,1)。"
                             "建议 v26 用 1.0（覆盖 ±2 rad/s 指令量程），"
                             "对 0.06 rad/s 的航向漂移也能给出恒定惩罚")
    args = parser.parse_args()

    # --mem 覆盖环境变量
    if args.mem is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem)

    # --- 设备检测 ---
    devices = jax.devices()
    is_gpu = any(d.platform == "gpu" for d in devices)
    log(f"=== 两阶段师生训练 ===")
    log(f"总步数: {args.steps:,}")
    log(f"jax devices: {devices}")
    log(f"backend: {'GPU' if is_gpu else 'CPU'}")
    if not is_gpu:
        log("⚠️ CPU 模式：JIT 编译极慢，建议在 4090 服务器上运行")

    # --- 环境数 ---
    if args.n_envs is not None:
        n_envs = args.n_envs
    elif is_gpu:
        n_envs = 2048
    else:
        n_envs = 16
    log(f"n_envs = {n_envs}")

    # --- PPO 参数自适应 ---
    if args.steps <= 10000:
        # 小步数冒烟：用更小的 PPO 参数
        rollout_len = min(args.rollout_len, 8)
        n_epochs = min(args.n_epochs, 2)
        n_minibatches = min(args.n_minibatches, 4)
        log(f"冒烟模式：rollout={rollout_len}, epochs={n_epochs}, "
            f"mb={n_minibatches}")
    else:
        rollout_len = args.rollout_len
        n_epochs = args.n_epochs
        n_minibatches = args.n_minibatches

    # --- 环境 ---
    # 模型级 DR 池数 K：关 DR 时恒为 1（否则 build_model 会走批化慢路径）；
    # K>1 时须整除 n_envs，否则回退 1
    dr_pools = 1 if args.no_dr else args.dr_pools
    if dr_pools > 1 and n_envs % dr_pools != 0:
        log(f"⚠️ dr_pools={dr_pools} 不整除 n_envs={n_envs}，回退 dr_pools=1")
        dr_pools = 1
    # 指令范围（课程）：不传则用 EnvConfig 默认（全程）
    cmd_kwargs = {}
    if args.cmd_vx is not None:
        cmd_kwargs["cmd_vx"] = tuple(args.cmd_vx)
    if args.cmd_vy is not None:
        cmd_kwargs["cmd_vy"] = tuple(args.cmd_vy)
    if args.cmd_yaw is not None:
        cmd_kwargs["cmd_yaw"] = tuple(args.cmd_yaw)
    if args.zero_cmd_prob is not None:
        cmd_kwargs["zero_cmd_prob"] = float(args.zero_cmd_prob)
    if args.track_sigma2 is not None:
        cmd_kwargs["track_sigma2"] = float(args.track_sigma2)
    if args.yaw_sigma2 is not None:
        cmd_kwargs["yaw_sigma2"] = float(args.yaw_sigma2)
    if args.yaw_lin_scale is not None:
        cmd_kwargs["yaw_lin_scale"] = float(args.yaw_lin_scale)

    env_cfg = EnvConfig(
        dr_enabled=not args.no_dr,
        use_residual=not args.no_residual,
        dr_pools=dr_pools,
        gait_mode=("ik" if args.gait_mode == "ik" else "sine"),
        **cmd_kwargs,
    )
    log(f"env: residual={env_cfg.use_residual}, dr={env_cfg.dr_enabled}, "
        f"dr_pools={env_cfg.dr_pools}, gait_mode={env_cfg.gait_mode}")
    log(f"指令范围: vx={env_cfg.cmd_vx} vy={env_cfg.cmd_vy} "
        f"yaw={env_cfg.cmd_yaw} zero_prob={env_cfg.zero_cmd_prob}")
    log(f"跟踪 σ²: lin={env_cfg.track_sigma2} (σ={env_cfg.track_sigma2 ** 0.5:.2f} m/s) "
        f"yaw={env_cfg.yaw_sigma2} (σ={env_cfg.yaw_sigma2 ** 0.5:.2f} rad/s)")
    env = DogEnvMJX(n_envs=n_envs, config=env_cfg)

    rng = jax.random.PRNGKey(42)

    # --- checkpoint 回调 ---
    def teacher_save_fn(ts, obs_rms, update):
        path = CKPT_DIR / "dog_cmd_mjx_teacher.pkl"
        save_flax_checkpoint(ts, obs_rms, path)

    def student_save_fn(ts, _, update):
        path = CKPT_DIR / "dog_cmd_mjx_student.pkl"
        save_flax_checkpoint(ts, None, path)

    # --- --resume：加载教师 checkpoint（params + obs_rms）---
    init_params, init_obs_rms = None, None
    if args.resume:
        ckpt_path = CKPT_DIR / "dog_cmd_mjx_teacher.pkl"
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        init_params, init_obs_rms = ck["params"], ck["obs_rms"]
        log(f"--resume：已加载 {ckpt_path}")

    # ========== 阶段 1：PPO 训教师 ==========
    log("\n--- 阶段 1：PPO 训教师 ---")
    save_interval = max(1, args.steps // (n_envs * rollout_len * 10))
    cfg_kwargs = {}
    if args.ent_coef is not None:
        cfg_kwargs["entropy_coef"] = args.ent_coef
        log(f"熵系数覆盖：{args.ent_coef}")
    if args.target_kl is not None:
        log(f"⚠️ --target_kl 已失效（epoch 闸在 mega-jit 路径中不存在），忽略")
    if args.target_kl_mb is not None:
        cfg_kwargs["target_kl_mb"] = args.target_kl_mb
        log(f"per-mb 累积 KL 冻结阈值覆盖：{args.target_kl_mb}")
    ppo_cfg = PPOConfig(
        n_envs=n_envs,
        total_steps=args.steps,
        rollout_len=rollout_len,
        n_epochs=n_epochs,
        n_minibatches=n_minibatches,
        lr=args.lr,
        save_interval=save_interval,
        log_interval=1,
        **cfg_kwargs,
    )
    log(f"PPO: save_interval={save_interval} updates, fixed lr={args.lr:.1e}")
    rng, train_rng = jax.random.split(rng)
    teacher_ts, obs_rms, teacher_metrics = train_teacher(
        train_rng, env, ppo_cfg, log_fn=log, save_fn=teacher_save_fn,
        init_params=init_params, init_obs_rms=init_obs_rms)
    save_flax_checkpoint(teacher_ts, obs_rms,
                         CKPT_DIR / "dog_cmd_mjx_teacher.pkl")

    # ========== 阶段 2：蒸馏学生 ==========
    log("\n--- 阶段 2：蒸馏学生 ---")
    distill_total = args.distill_steps or (args.steps // 4)
    if args.steps <= 10000:
        d_rollout = 8
        d_epochs = 2
        d_mb = 4
    else:
        d_rollout = args.rollout_len
        d_epochs = args.n_epochs
        d_mb = args.n_minibatches
    distill_cfg = DistillConfig(
        n_envs=n_envs,
        total_steps=distill_total,
        rollout_len=d_rollout,
        n_epochs=d_epochs,
        n_minibatches=d_mb,
        lr=args.lr,
        save_interval=max(1, distill_total // (n_envs * d_rollout * 10)),
        log_interval=1,
    )
    rng, distill_rng = jax.random.split(rng)
    # 蒸馏的观测归一化必须与教师训练（PPO rollout 内 obs_rms 归一化）以及
    # 部署端（npz 存 obs_mean/var）三方一致：教师用完整 99 维统计量，学生
    # 用前 61 维切片。传入教师训出的 obs_rms，见 distill_jax 模块 docstring。
    log(f"  蒸馏输入归一化：教师 99 维 obs_rms / 学生前 61 维切片")
    student_ts, student_metrics = train_student(
        distill_rng, teacher_ts, env, obs_rms, distill_cfg,
        log_fn=log, save_fn=student_save_fn)

    # 学生用 61 维观测的 obs_rms（从教师 99 维的前 61 维截取）
    student_obs_rms = type(obs_rms)(
        mean=obs_rms.mean[:61],
        var=obs_rms.var[:61],
        count=obs_rms.count,
    )
    save_flax_checkpoint(student_ts, student_obs_rms,
                         CKPT_DIR / "dog_cmd_mjx_student.pkl")
    save_student_npz(student_ts, student_obs_rms,
                     CKPT_DIR / "dog_cmd_mjx_student.npz")

    # ========== 绘图 ==========
    plot_curves(teacher_metrics, student_metrics)
    log("\n训练完成 ✅")


if __name__ == "__main__":
    main()
