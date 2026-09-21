"""训练入口：DogEnv（域随机化）+ 自实现 PPO（多环境并行采样）

用法：
    python train.py                  # 单环境 1M 步 sanity check
    python train.py 20480            # 冒烟：约 10 批
    python train.py 20000000 8       # 8 路并行，20M 步（Step 5 长训）

参数：
    argv[1] total_steps   总环境步数（默认 1,000,000）
    argv[2] n_envs        并行环境数（默认 1；>1 用 AsyncVectorEnv 多进程）

每批规模：
    n_envs=1：2048 步/批；n_envs>1：每环境 512 步（如 8 env → 4096 转移/批）
    必须能被 minibatch(64) 整除（已自检）。

产物：
    checkpoints/dog_ppo_best.pt    评估 mean 新高即存（含 obs_rms 统计量）
    checkpoints/dog_ppo_final.pt   训练结束权重
    logs/dog_ppo_curve.png         学习曲线（训练窗 vs 评估 + V 探针）

说明：
    - DR 默认开启，评估也走 DR（10 局不同 seed → eval std 非零，更真实）
    - lr 衰减按 anneal_steps（默认=total_steps）；长短实验如需公平对比可显式指定
"""

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from envs.dog_env import DogEnv, RandConfig
from algos.networks import PolicyNetwork, ValueNetwork
from algos.ppo import (PPOConfig, collect_rollout_vec, compute_gae_vec,
                       update_ppo, evaluate, set_learning_rate)
from utils.normalizer import RunningMeanStd


def make_dog_env(seed: int, rand_cfg: RandConfig = None):
    """模块级环境工厂（AsyncVectorEnv spawn 要求 thunk 可 pickle，
    闭包在 Windows 上不可 pickle，故放模块级）。"""
    if rand_cfg is None:
        rand_cfg = RandConfig()
    env = DogEnv(rand_config=rand_cfg)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_vector_env(n_envs: int, base_seed: int, rand_cfg: RandConfig):
    """n_envs=1 用 SyncVectorEnv；>1 用 AsyncVectorEnv。

    autoreset 用 SAME_STEP（旧版语义：done 当步返回新局初态、终态放
    infos['final_obs']），PPO 的 done/reset 记账最直接。老版本 gymnasium
    不支持 autoreset_mode 参数时静默回退。
    """
    from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode
    thunks = [partial(make_dog_env, seed=base_seed + i, rand_cfg=rand_cfg)
              for i in range(n_envs)]
    cls = SyncVectorEnv if n_envs == 1 else AsyncVectorEnv
    try:
        return cls(thunks, autoreset_mode=AutoresetMode.SAME_STEP)
    except (TypeError, AttributeError):
        return cls(thunks)


def train(total_steps: int = None, n_envs: int = 1):
    cfg = PPOConfig()
    if total_steps is not None:
        cfg.total_steps = total_steps
    cfg.n_envs = n_envs
    # 多环境时每环境 512 步/批（batch=512×N）；单环境保持 2048
    cfg.steps_per_env = 2048 if n_envs == 1 else 512
    batch_total = cfg.steps_per_batch
    assert batch_total % cfg.minibatch == 0, \
        f"batch {batch_total} 不能被 minibatch {cfg.minibatch} 整除"

    # --- 可复现性：numpy / torch 种子（各 env 种子在工厂内单独设） ---
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dr_cfg = RandConfig()
    venv = make_vector_env(cfg.n_envs, cfg.seed, dr_cfg)
    obs, _ = venv.reset(seed=[cfg.seed + i for i in range(cfg.n_envs)])

    state_dim = obs.shape[1]              # 58
    action_dim = 12
    assert state_dim == 58

    actor = PolicyNetwork(state_dim, action_dim).to(cfg.device)
    critic = ValueNetwork(state_dim).to(cfg.device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    print(f"设备: {cfg.device} | 目标 {cfg.total_steps:,} 步 | "
          f"{cfg.n_envs} 环境并行（{batch_total} 转移/批）")
    print(f"Actor 参数 {sum(p.numel() for p in actor.parameters()):,} | "
          f"Critic 参数 {sum(p.numel() for p in critic.parameters()):,}")
    print(f"策略 50Hz（frame_skip=10）| 单局 1000 step = 20s | 域随机化: 开")
    print()

    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # obs 在线归一化（随 checkpoint 一起保存，部署时必须用同一统计量）
    obs_rms = RunningMeanStd(shape=(state_dim,))

    probe_states = None          # V 探针：第 1 批冻结 256 个【归一化】状态
    best_score = -float("inf")

    history = {"step": [], "train_mean": [], "eval_mean": [],
               "eval_std": [], "v_probe": []}
    window_scores = []
    total_steps_done = 0
    batch_no = 0
    train_start = time.time()

    # lr 衰减时长（与 total_steps 解耦；默认两者相同）
    anneal_denom = cfg.anneal_steps or cfg.total_steps
    # 评估批次间隔（batch 数；不整除时四舍五入）
    eval_every = max(1, round(cfg.eval_interval / batch_total))

    while total_steps_done < cfg.total_steps:
        # 学习率线性衰减：起点 cfg.lr，终点 0
        if cfg.anneal_lr:
            frac = max(0.0, 1.0 - total_steps_done / anneal_denom)
            set_learning_rate(actor_opt, cfg.lr * frac)
            set_learning_rate(critic_opt, cfg.lr * frac)

        batch_t0 = time.time()
        buf, info = collect_rollout_vec(venv, actor, critic, obs, cfg, obs_rms)
        obs = info["next_obs"]
        advantages, returns = compute_gae_vec(buf, critic, cfg)
        stats = update_ppo(actor, critic, actor_opt, critic_opt,
                           buf, advantages, returns, cfg)
        batch_dt = time.time() - batch_t0

        if probe_states is None:
            flat_obs = torch.as_tensor(buf["obs"].reshape(-1, state_dim),
                                       dtype=torch.float32, device=cfg.device)
            probe_states = flat_obs[:256]

        total_steps_done += batch_total
        batch_no += 1
        window_scores.extend(info["ep_scores"])

        do_eval = batch_no % eval_every == 0
        is_last = total_steps_done >= cfg.total_steps
        if do_eval or is_last:
            eval_factory = partial(make_dog_env, seed=0, rand_cfg=dr_cfg)
            ev = evaluate(actor, eval_factory, cfg, obs_rms=obs_rms)
            with torch.no_grad():
                v_probe = critic(probe_states).mean().item()
            train_mean = float(np.mean(window_scores)) if window_scores else float("nan")
            train_std = float(np.std(window_scores)) if window_scores else float("nan")
            sigma = actor.log_std.exp().detach().cpu().numpy()
            elapsed = time.time() - train_start
            sps = total_steps_done / elapsed

            print(
                f"[{total_steps_done:>8,}步] "
                f"训练窗 {train_mean:7.1f}±{train_std:5.1f} | "
                f"评估 {ev['mean']:7.1f}±{ev['std']:5.1f} "
                f"[{ev['min']:7.1f},{ev['max']:7.1f}]"
            )
            stop_flag = "KL停" if stats["kl_stop"] else "    "
            print(
                f"           clip {stats['clip_frac']*100:4.1f}% "
                f"kl {stats['approx_kl']:.4f} H {stats['entropy']:5.2f} "
                f"V探 {v_probe:7.1f} vloss {stats['critic_loss']:8.1f} | "
                f"{sps:,.0f} step/s 批{batch_dt:.1f}s {stop_flag}"
                f"({stats['n_updates']}/{stats['max_updates']})"
            )
            print(f"           σ = {np.array2string(sigma, precision=3, separator=', ')}")

            history["step"].append(total_steps_done)
            history["train_mean"].append(train_mean)
            history["eval_mean"].append(ev["mean"])
            history["eval_std"].append(ev["std"])
            history["v_probe"].append(v_probe)
            window_scores.clear()

            if ev["mean"] > best_score:
                best_score = ev["mean"]
                torch.save({"actor": actor.state_dict(),
                            "obs_rms": obs_rms.state_dict()},
                           "checkpoints/dog_ppo_best.pt")

    venv.close()
    torch.save({"actor": actor.state_dict(),
                "obs_rms": obs_rms.state_dict()},
               "checkpoints/dog_ppo_final.pt")
    print(f"\n训练结束，最佳评估均值 = {best_score:.1f}，checkpoint 已存 checkpoints/")

    plot_history(history)
    return actor, history


def plot_history(history):
    """训练窗 vs 确定性评估双曲线 + V 探针"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array(history["step"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps, history["train_mean"], label="train window (noisy)", alpha=0.6)
    ax1.plot(steps, history["eval_mean"], label="eval (deterministic mu)",
             color="red", linewidth=2)
    ax1.fill_between(steps,
                     np.array(history["eval_mean"]) - np.array(history["eval_std"]),
                     np.array(history["eval_mean"]) + np.array(history["eval_std"]),
                     color="red", alpha=0.15)
    ax1.axhline(500, color="orange", linewidth=0.8, linestyle="--",
                label="zero-policy baseline ≈500")
    ax1.set_ylabel("episode return")
    ax1.set_title("PPO DogEnv (58-dim obs, 50Hz, position control, domain rand)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps, history["v_probe"], color="green")
    ax2.set_ylabel("V on fixed probe states")
    ax2.set_xlabel("env steps")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/dog_ppo_curve.png", dpi=120)
    print("学习曲线已保存: logs/dog_ppo_curve.png")


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else None
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    train(total, n)
