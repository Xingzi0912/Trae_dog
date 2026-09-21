"""PPO 算法（Step 3，从 14_ppo_mujoco.py 拆出并配置化）

包含：
    PPOConfig   —— 全部超参（agents.md 参数清单，entropy_beta=0.01）
    collect_rollout —— 用当前策略采样（raw 动作入 buffer，clip 只发给 env）
    compute_gae     —— 双掩码 GAE（term 不 bootstrap / trunc 要 bootstrap）
    update_ppo      —— clip 目标 + 多 epoch minibatch，返回护栏监控
    evaluate        —— 确定性 μ 评估（make_env 工厂，默认 DogEnv）

与 14 号脚本的差异：
    1. 全局常量 → PPOConfig 数据类，train.py 集中配置
    2. evaluate 不再 gym.make(ENV_ID)，改为接收 make_env 工厂（DogEnv）
    3. ENTROPY_BETA 0.0 → 0.01（agents.md：局部最优多，加探索）
    GAE/clip/多 epoch/adv 标准化/logp 重算等数学骨架原封不动。
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
    # 优化
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_beta: float = 0.01  # agents.md 相对 14 号（0.0）的改动
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
#   两条铁律（同 14 号）：
#   ① log_prob 对 12 维求和：联合高斯密度 = 各维之积 → 对数之和
#   ② buffer 存 raw 采样动作；clip(-1,1) 只在送 env.step 前
# ============================================================
def collect_rollout(env, actor, critic, obs, cfg: PPOConfig):
    buf = {k: [] for k in
           ("obs", "next_obs", "actions", "log_probs", "rewards", "values",
            "terminated", "truncated")}
    ep_scores = []          # 本批内跑完的整局回报
    cur_return = 0.0        # 当前未完成局的累计回报
    entropy_sum = 0.0       # 本批熵累计（返回均值做监控）

    with torch.no_grad():
        for _ in range(cfg.steps_per_batch):
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device)

            dist = actor(state_t)
            action_raw = dist.sample()                          # 可能越界
            log_prob = dist.log_prob(action_raw).sum(-1)        # 12 维求和
            value = critic(state_t)
            entropy_sum += dist.entropy().sum(-1).item()

            # clip 只发生在此处：raw 入 buffer，clip 版给环境
            action_env = action_raw.cpu().numpy().clip(-1.0, 1.0)
            next_obs, reward, term, trunc, _ = env.step(action_env)
            cur_return += reward

            buf["obs"].append(obs)
            buf["next_obs"].append(next_obs)  # done 时是终止/截断态（reset 前）
            buf["actions"].append(action_raw.cpu().numpy())
            buf["log_probs"].append(log_prob.item())
            buf["rewards"].append(reward)
            buf["values"].append(value.item())
            buf["terminated"].append(term)
            buf["truncated"].append(trunc)

            obs = next_obs
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
#   铁律① returns = A + V 在标准化之前
#   铁律② 整批标准化一次，minibatch 只切片
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
# PPO clip 更新（4 epoch × minibatch）
#   L_actor  = -mean(min(r·A, clip(r)·A)) - β·H
#   L_critic = MSE(V_new, returns)
#   r = exp(logp_new - logp_old)，logp 在【raw 动作】点上重算
#
# 监控：clip_frac（健康 5%~30%）/ approx_kl（跑远警报）/ entropy
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

    for _epoch in range(cfg.epochs):
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
            actor_opt.step()
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            with torch.no_grad():
                clip_fracs.append(
                    ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item())
                entropies.append(entropy.item())
                approx_kls.append(
                    (logp_old_all[mb] - logp_new).mean().clamp_min(0.0).item())

    return {
        "clip_frac": float(np.mean(clip_fracs)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
        "critic_loss": float(critic_loss.item()),
    }


# ============================================================
# 确定性评估：只取 μ（高斯最高点），固定种子跨轮可比
# make_env: 返回新 DogEnv 的可调用对象（train.py 传入，避免与 gym 耦合）
# ============================================================
def evaluate(actor, make_env, cfg: PPOConfig, n_episodes: int = 10):
    env = make_env()
    scores = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=10000 + ep)
            total = 0.0
            for _ in range(cfg.max_episode_steps):
                state_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device)
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
