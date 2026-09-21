"""PPO 算法（Step 3.1 稳定性修复版）

相对 Step 3（1M 实验：clip 60~75%、KL 频繁爆炸到 1~13、σ 0.6→1.4）：
    1. lr 3e-4 → 2e-4，并线性衰减到 0（anneal_lr）
    2. entropy_beta 0.01 → 0.003（遏制 σ 持续膨胀）
    3. 新增 KL 早停：minibatch 级 approx-KL > target_kl(0.02) 立即结束
       本批剩余 epoch —— 治"策略灾难性跳变"的主药
    4. 新增梯度裁剪 max_grad_norm=0.5（CleanRL 默认，防爆步）
    5. 采样/评估全链路接入 RunningMeanStd obs 归一化（utils/normalizer.py）

数学骨架（双掩码 GAE / clip 目标 / adv 标准化 / raw 动作重算 logp）不变。

包含：
    PPOConfig        —— 全部超参
    collect_rollout  —— 归一化后采样（归一化 obs 入 buffer）
    compute_gae      —— 双掩码 GAE（buf obs 已是归一化版）
    update_ppo       —— clip + KL 早停 + 梯度裁剪，返回护栏监控
    evaluate         —— 确定性 μ 评估（冻结归一化统计量，10 局）
    set_learning_rate —— 线性衰减辅助
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import device as torch_device


@dataclass
class PPOConfig:
    # rollout / 更新节奏
    steps_per_batch: int = 2048
    epochs: int = 4
    minibatch: int = 64
    # 优化（Step 3.1：lr 3e-4→2e-4，entropy 0.01→0.003）
    lr: float = 2e-4
    anneal_lr: bool = True          # 线性衰减到 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_beta: float = 0.003
    target_kl: float = 0.02         # KL 早停阈值（None 关闭）
    max_grad_norm: float = 0.5      # 梯度范数裁剪（None 关闭）
    # 规模
    total_steps: int = 1_000_000
    eval_interval: int = 20_480  # 每 10 批评估一次
    max_episode_steps: int = 1000
    seed: int = 42
    device: torch_device = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# rollout 收集
#   三条铁律：
#   ① 原始 obs 先 obs_rms.update，再 normalize 后送网络
#   ② next_obs 用【同批统计量】归一化，保证 V(s)/V(s') 尺度成对一致
#   ③ buffer 存归一化 obs 和 raw 采样动作；clip(-1,1) 只在送 env 前
# ============================================================
def collect_rollout(env, actor, critic, obs, cfg: PPOConfig, obs_rms=None):
    buf = {k: [] for k in
           ("obs", "next_obs", "actions", "log_probs", "rewards", "values",
            "terminated", "truncated")}
    ep_scores = []          # 本批内跑完的整局回报
    cur_return = 0.0        # 当前未完成局的累计回报
    entropy_sum = 0.0       # 本批熵累计（返回均值做监控）

    with torch.no_grad():
        for _ in range(cfg.steps_per_batch):
            # 仅训练路径更新统计量；obs_rms=None 时退化为不归一化
            if obs_rms is not None:
                obs_rms.update(obs)
                obs_in = obs_rms.normalize(obs)
            else:
                obs_in = obs
            state_t = torch.as_tensor(obs_in, dtype=torch.float32, device=cfg.device)

            dist = actor(state_t)
            action_raw = dist.sample()                          # 可能越界
            log_prob = dist.log_prob(action_raw).sum(-1)        # 12 维求和
            value = critic(state_t)
            entropy_sum += dist.entropy().sum(-1).item()

            # clip 只发生在此处：raw 入 buffer，clip 版给环境
            action_env = action_raw.cpu().numpy().clip(-1.0, 1.0)
            next_obs, reward, term, trunc, _ = env.step(action_env)
            cur_return += reward

            # next_obs 用当前（更新后）统计量归一化；done 时为终止/截断态
            next_in = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs

            buf["obs"].append(obs_in)
            buf["next_obs"].append(next_in)
            buf["actions"].append(action_raw.cpu().numpy())
            buf["log_probs"].append(log_prob.item())
            buf["rewards"].append(reward)
            buf["values"].append(value.item())
            buf["terminated"].append(term)
            buf["truncated"].append(trunc)

            obs = next_obs          # 续接始终是【原始】obs，下轮再归一化
            if term or trunc:
                ep_scores.append(cur_return)
                cur_return = 0.0
                obs, _ = env.reset()  # 续接 obs 换成新局初态

    info = {
        "entropy_mean": entropy_sum / cfg.steps_per_batch,
        "ep_scores": ep_scores,
        "next_obs": obs,
    }
    return buf, info


# ============================================================
# GAE（双掩码）
#   boot  = 1 - terminated：坠毁未来没有了 → V(s') 不自举；
#                           时间截断轨迹仍在 → 照常自举
#   chain = 1 - done      ：本局结束，A 链绝不传到下一局
#   buf["obs"]/["next_obs"] 已是归一化版，critic 直接读
# ============================================================
def compute_gae(buf, critic, cfg: PPOConfig):
    obs_t = torch.as_tensor(np.asarray(buf["obs"]),
                            dtype=torch.float32, device=cfg.device)
    next_obs_t = torch.as_tensor(np.asarray(buf["next_obs"]),
                                 dtype=torch.float32, device=cfg.device)
    with torch.no_grad():
        values = critic(obs_t).cpu().numpy()
        next_values = critic(next_obs_t).cpu().numpy()

    T = len(buf["rewards"])
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        done = buf["terminated"][t] or buf["truncated"][t]
        boot = 1.0 - float(buf["terminated"][t])
        chain = 1.0 - float(done)
        delta = buf["rewards"][t] + cfg.gamma * next_values[t] * boot - values[t]
        last_adv = delta + cfg.gamma * cfg.gae_lambda * chain * last_adv
        advantages[t] = last_adv

    returns = advantages + values  # 原始尺度 λ-return
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


# ============================================================
# PPO clip 更新（4 epoch × minibatch）+ KL 早停 + 梯度裁剪
#   L_actor  = -mean(min(r·A, clip(r)·A)) - β·H
#   L_critic = MSE(V_new, returns)
#
# KL 早停：每个 minibatch 更新后，若【本批所有更新的累计平均 approx-KL】
#   超过 target_kl 立即跳出（不用单 minibatch 值——64 样本噪声大易误触，
#   同 sb3 的整批均值语义），监控里 kl_stop=True。
# ============================================================
def update_ppo(actor, critic, actor_opt, critic_opt, buf,
               advantages, returns, cfg: PPOConfig):
    obs_all = torch.as_tensor(np.asarray(buf["obs"]),
                              dtype=torch.float32, device=cfg.device)
    act_all = torch.as_tensor(np.asarray(buf["actions"]),
                              dtype=torch.float32, device=cfg.device)
    logp_old_all = torch.as_tensor(np.asarray(buf["log_probs"]),
                                   dtype=torch.float32, device=cfg.device)
    adv_all = torch.as_tensor(advantages, dtype=torch.float32, device=cfg.device)
    ret_all = torch.as_tensor(returns, dtype=torch.float32, device=cfg.device)

    T = len(buf["rewards"])
    idx = np.arange(T)
    clip_fracs, entropies, approx_kls = [], [], []
    n_updates = 0
    kl_stop = False

    for _epoch in range(cfg.epochs):
        if kl_stop:
            break
        np.random.shuffle(idx)
        for start in range(0, T, cfg.minibatch):
            mb = torch.as_tensor(idx[start:start + cfg.minibatch], device=cfg.device)

            dist_new = actor(obs_all[mb])
            logp_new = dist_new.log_prob(act_all[mb]).sum(-1)  # 12 维求和
            ratio = torch.exp(logp_new - logp_old_all[mb])

            adv_mb = adv_all[mb]
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps,
                                1.0 + cfg.clip_eps) * adv_mb
            entropy = dist_new.entropy().sum(-1).mean()
            actor_loss = -torch.min(surr1, surr2).mean() \
                         - cfg.entropy_beta * entropy

            values_new = critic(obs_all[mb])
            critic_loss = F.mse_loss(values_new, ret_all[mb])

            actor_opt.zero_grad()
            actor_loss.backward()  # μ 网络与 log_std 同此一步
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()
            critic_opt.zero_grad()
            critic_loss.backward()
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_opt.step()

            with torch.no_grad():
                mb_kl = (logp_old_all[mb] - logp_new).mean().clamp_min(0.0).item()
                clip_fracs.append(
                    ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item())
                entropies.append(entropy.item())
                approx_kls.append(mb_kl)
            n_updates += 1

            # KL 早停：本批累计平均 KL 超阈值，策略已跑远，剩余 epoch 放弃
            if cfg.target_kl is not None \
                    and float(np.mean(approx_kls)) > cfg.target_kl:
                kl_stop = True
                break

    return {
        "clip_frac": float(np.mean(clip_fracs)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
        "critic_loss": float(critic_loss.item()),
        "kl_stop": kl_stop,
        "n_updates": n_updates,
    }


# ============================================================
# 确定性评估：只取 μ（高斯最高点），10 局不同 reset seed
#   obs_rms 只 normalize、不 update（冻结训练统计量）
# ============================================================
def evaluate(actor, make_env, cfg: PPOConfig, n_episodes: int = 10,
             obs_rms=None):
    env = make_env()
    scores = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=10000 + ep)
            total = 0.0
            for _ in range(cfg.max_episode_steps):
                obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
                state_t = torch.as_tensor(obs_in, dtype=torch.float32, device=cfg.device)
                mu = actor(state_t).mean
                action = mu.cpu().numpy().clip(-1.0, 1.0)
                obs, reward, term, trunc, _ = env.step(action)
                total += reward
                if term or trunc:
                    break
            scores.append(total)
    env.close()
    scores = np.asarray(scores)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


def set_learning_rate(optimizer, lr_now: float) -> None:
    """统一设置一个 optimizer 所有参数组的学习率"""
    for group in optimizer.param_groups:
        group["lr"] = lr_now
