"""训练入口：DogEnv + 自实现 PPO（Step 3）

用法：
    python train.py                # 1M 步 sanity check（PPOConfig 默认）
    python train.py 20480          # 冒烟：10 批 + 1 次评估，验证管线
    python train.py 20000000       # 指定总步数（Step 5 长训）

产物：
    checkpoints/dog_ppo_best.pt    评估 mean 新高即存
    checkpoints/dog_ppo_final.pt   训练结束权重
    logs/dog_ppo_curve.png         学习曲线（训练窗 vs 评估 + V 探针）

说明：Step 3.1 已提前接入 obs 归一化（RunningMeanStd，原 Step 4 内容），
域随机化仍在 Step 4。1M 步验证 reward 曲线与护栏指标。
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from envs.dog_env import DogEnv
from algos.networks import PolicyNetwork, ValueNetwork
from algos.ppo import (PPOConfig, collect_rollout, compute_gae, update_ppo,
                       evaluate, set_learning_rate)
from utils.normalizer import RunningMeanStd


def train(total_steps: int = None):
    cfg = PPOConfig()
    if total_steps is not None:
        cfg.total_steps = total_steps

    # --- 可复现性：env / numpy / torch 三处种子 ---
    env = DogEnv()
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    state_dim = env.observation_space.shape[0]   # 58
    action_dim = env.action_space.shape[0]       # 12
    assert state_dim == 58 and action_dim == 12

    actor = PolicyNetwork(state_dim, action_dim).to(cfg.device)
    critic = ValueNetwork(state_dim).to(cfg.device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    print(f"设备: {cfg.device} | 目标 {cfg.total_steps:,} 步")
    print(f"Actor 参数 {sum(p.numel() for p in actor.parameters()):,} | "
          f"Critic 参数 {sum(p.numel() for p in critic.parameters()):,}")
    print(f"策略 50Hz（frame_skip=10）| 单局 1000 step = 20s")
    print()

    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # obs 在线归一化（随 checkpoint 一起保存，部署时必须用同一统计量）
    obs_rms = RunningMeanStd(shape=(state_dim,))

    probe_states = None          # V 探针：第 1 批冻结 256 个【归一化】状态
    best_score = -float("inf")
    obs, _ = env.reset(seed=cfg.seed)

    history = {"step": [], "train_mean": [], "eval_mean": [],
               "eval_std": [], "v_probe": []}
    window_scores = []
    total_steps_done = 0
    batch_no = 0
    train_start = time.time()

    while total_steps_done < cfg.total_steps:
        # 学习率线性衰减：起点 cfg.lr，终点 0（按剩余步数比例）
        if cfg.anneal_lr:
            frac = 1.0 - total_steps_done / cfg.total_steps
            set_learning_rate(actor_opt, cfg.lr * frac)
            set_learning_rate(critic_opt, cfg.lr * frac)

        batch_t0 = time.time()
        buf, info = collect_rollout(env, actor, critic, obs, cfg, obs_rms)
        obs = info["next_obs"]
        advantages, returns = compute_gae(buf, critic, cfg)
        stats = update_ppo(actor, critic, actor_opt, critic_opt,
                           buf, advantages, returns, cfg)
        batch_dt = time.time() - batch_t0

        if probe_states is None:
            probe_states = torch.as_tensor(
                np.asarray(buf["obs"])[:256], dtype=torch.float32, device=cfg.device)

        total_steps_done += cfg.steps_per_batch
        batch_no += 1
        window_scores.extend(info["ep_scores"])

        do_eval = batch_no % (cfg.eval_interval // cfg.steps_per_batch) == 0
        is_last = total_steps_done >= cfg.total_steps
        if do_eval or is_last:
            ev = evaluate(actor, DogEnv, cfg, obs_rms=obs_rms)
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
                f"({stats['n_updates']}/128)"
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

    env.close()
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
    ax1.set_title("PPO DogEnv (58-dim obs, 50Hz, position control)")
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
    train(total)
