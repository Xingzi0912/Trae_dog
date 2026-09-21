"""PPO 算法（Step 4：多环境并行采样版）

相对 Step 3.1（单环境版）：
    1. 采样输入从单个 env 改为 gymnasium 向量环境（Sync/AsyncVectorEnv），
       一批数据 = steps_per_env × n_envs 条转移；N 个环境的 done/reset/bootstrap
       完全独立，episode 记账按 env 分开
    2. GAE 改为按 env 分开的反向链（同 CleanRL 向量版）：
       坠毁 → 不自举 V(s')；截断 → 照常自举；done → A 链不传下一局
    3. KL 早停/梯度裁剪/lr 衰减接入点语义不变，分母改为动态（总 minibatch 数）
    4. 新增 anneal_steps：lr 衰减时长与 total_steps 解耦
       （200k 验证与 1M 跑在同样步数处 lr 一致，实验可公平对比）

数学骨架（双掩码 GAE / clip 目标 / adv 全局标准化 / raw 动作重算 logp）不变。

包含：
    PPOConfig           —— 全部超参（含 n_envs / steps_per_env / anneal_steps）
    collect_rollout_vec —— 多环境归一化采样（buffer 形状 T×N）
    compute_gae_vec     —— 逐 env GAE（T×N）
    update_ppo          —— clip + KL 早停 + 梯度裁剪，返回护栏监控
    evaluate            —— 确定性 μ 评估（冻结归一化统计量，10 局）
    set_learning_rate   —— 线性衰减辅助
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import device as torch_device


@dataclass
class PPOConfig:
    # rollout / 更新节奏（一批 = steps_per_env × n_envs 条转移）
    n_envs: int = 1
    steps_per_env: int = 2048
    epochs: int = 4
    minibatch: int = 64
    # 优化
    lr: float = 2e-4
    anneal_lr: bool = True          # 线性衰减
    anneal_steps: int = None        # 衰减时长（None = total_steps）
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_beta: float = 0.003
    target_kl: float = 0.02         # KL 早停阈值（None 关闭）
    max_grad_norm: float = 0.5      # 梯度范数裁剪（None 关闭）
    # 规模
    total_steps: int = 1_000_000
    eval_interval: int = 20_480  # 每隔约多少 env step 评估一次
    max_episode_steps: int = 1000
    seed: int = 42
    device: torch_device = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def steps_per_batch(self) -> int:
        """每批总转移数 = T × N"""
        return self.n_envs * self.steps_per_env


# ============================================================
# 多环境 rollout 收集
#   三条铁律不变：
#   ① 原始 obs 先 obs_rms.update，再 normalize 后送网络（N 个一起批量更新）
#   ② done 环境的 final obs 从 infos["final_obs"] 取（SAME_STEP autoreset 语义：
#      step 返回的 obs 对 done env 是【新局初态】，final obs 在 info 里）
#   ③ buffer 存归一化 obs / final obs 和 raw 采样动作；clip(-1,1) 只在送 env 前
# ============================================================
def collect_rollout_vec(venv, actor, critic, obs, cfg: PPOConfig, obs_rms=None):
    T, N = cfg.steps_per_env, cfg.n_envs
    state_dim = obs.shape[1]
    action_dim = actor.mu_head.out_features

    # 预分配（比逐帧 append 快，GAE 直接按 T×N 索引）
    b_obs = np.zeros((T, N, state_dim), dtype=np.float32)
    b_next = np.zeros((T, N, state_dim), dtype=np.float32)
    b_act = np.zeros((T, N, action_dim), dtype=np.float32)
    b_logp = np.zeros((T, N), dtype=np.float32)
    b_rew = np.zeros((T, N), dtype=np.float32)
    b_term = np.zeros((T, N), dtype=np.float32)
    b_trunc = np.zeros((T, N), dtype=np.float32)

    ep_scores = []             # 本批内跑完的整局回报
    cur_return = np.zeros(N)   # 各 env 未完成局的累计回报
    entropy_sum = 0.0          # 本批熵累计（返回均值做监控）

    with torch.no_grad():
        for t in range(T):
            # 仅训练路径更新统计量；obs_rms=None 时退化为不归一化
            if obs_rms is not None:
                obs_rms.update(obs)                      # (N, dim) 批量
                obs_in = obs_rms.normalize(obs)
            else:
                obs_in = obs
            state_t = torch.as_tensor(obs_in, dtype=torch.float32, device=cfg.device)

            dist = actor(state_t)
            action_raw = dist.sample()                          # 可能越界
            log_prob = dist.log_prob(action_raw).sum(-1)        # (N,)，12 维求和
            value = critic(state_t)
            entropy_sum += dist.entropy().sum(-1).sum().item()

            # clip 只发生在此处：raw 入 buffer，clip 版给环境
            action_env = action_raw.cpu().numpy().clip(-1.0, 1.0)
            next_obs, reward, term, trunc, infos = venv.step(action_env)
            term = np.asarray(term, dtype=bool)
            trunc = np.asarray(trunc, dtype=bool)
            done = term | trunc

            # SAME_STEP：done env 返回的 next_obs 是新局初态；
            # 真正的 s'（终态）在 infos["final_obs"]，取出来做 bootstrap
            final_obs = next_obs.copy()
            if done.any():
                fo = infos.get("final_obs", None)
                if fo is not None:
                    for i in np.where(done)[0]:
                        if fo[i] is not None:
                            final_obs[i] = fo[i]

            cur_return += reward
            if done.any():
                for i in np.where(done)[0]:
                    ep_scores.append(float(cur_return[i]))
                    cur_return[i] = 0.0

            b_obs[t] = obs_in
            b_next[t] = obs_rms.normalize(final_obs) if obs_rms is not None else final_obs
            b_act[t] = action_raw.cpu().numpy()
            b_logp[t] = log_prob.cpu().numpy()
            b_rew[t] = reward
            b_term[t] = term.astype(np.float32)
            b_trunc[t] = trunc.astype(np.float32)

            obs = next_obs   # 续接：done env 已是新局初态（原始 obs）

    buf = {
        "obs": b_obs, "next_obs": b_next, "actions": b_act,
        "log_probs": b_logp, "rewards": b_rew,
        "terminated": b_term, "truncated": b_trunc,
    }
    info = {
        "entropy_mean": entropy_sum / (T * N),
        "ep_scores": ep_scores,
        "next_obs": obs,
    }
    return buf, info


# ============================================================
# GAE（逐 env 的双掩码反向链，同 CleanRL 向量版）
#   boot  = 1 - terminated：坠毁未来没有了 → V(s') 不自举；
#                           时间截断轨迹仍在 → 照常自举
#   chain = 1 - done      ：本局结束，A 链绝不传到下一局
#   buf["obs"]/["next_obs"] 已是归一化版，critic 直接读
# ============================================================
def compute_gae_vec(buf, critic, cfg: PPOConfig):
    T, N = buf["obs"].shape[:2]
    obs_flat = torch.as_tensor(buf["obs"].reshape(T * N, -1),
                               dtype=torch.float32, device=cfg.device)
    next_flat = torch.as_tensor(buf["next_obs"].reshape(T * N, -1),
                                dtype=torch.float32, device=cfg.device)
    with torch.no_grad():
        values = critic(obs_flat).view(T, N).cpu().numpy()
        next_values = critic(next_flat).view(T, N).cpu().numpy()

    advantages = np.zeros((T, N), dtype=np.float32)
    last_adv = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        term = buf["terminated"][t]
        done = term + buf["truncated"][t]
        boot = 1.0 - term
        chain = 1.0 - done
        delta = buf["rewards"][t] + cfg.gamma * next_values[t] * boot - values[t]
        last_adv = delta + cfg.gamma * cfg.gae_lambda * chain * last_adv
        advantages[t] = last_adv

    returns = advantages + values  # 原始尺度 λ-return
    # 全局（跨所有 env）标准化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


# ============================================================
# PPO clip 更新（4 epoch × minibatch）+ KL 早停 + 梯度裁剪
#   L_actor  = -mean(min(r·A, clip(r)·A)) - β·H
#   L_critic = MSE(V_new, returns)
#
# KL 早停：本批所有更新的累计平均 approx-KL 超过 target_kl 立即跳出
#   （不用单 minibatch 值——小样本噪声大易误触，同 sb3 整批均值语义）。
# ============================================================
def update_ppo(actor, critic, actor_opt, critic_opt, buf,
               advantages, returns, cfg: PPOConfig):
    n_total = buf["obs"].shape[0] * buf["obs"].shape[1]
    obs_all = torch.as_tensor(buf["obs"].reshape(n_total, -1),
                              dtype=torch.float32, device=cfg.device)
    act_all = torch.as_tensor(buf["actions"].reshape(n_total, -1),
                              dtype=torch.float32, device=cfg.device)
    logp_old_all = torch.as_tensor(buf["log_probs"].reshape(n_total),
                                   dtype=torch.float32, device=cfg.device)
    adv_all = torch.as_tensor(advantages.reshape(n_total),
                              dtype=torch.float32, device=cfg.device)
    ret_all = torch.as_tensor(returns.reshape(n_total),
                              dtype=torch.float32, device=cfg.device)

    idx = np.arange(n_total)
    clip_fracs, entropies, approx_kls = [], [], []
    n_updates = 0
    kl_stop = False
    # 本批理论总 minibatch 数（日志分母；KL 停时实际 n_updates < 它）
    max_updates = cfg.epochs * (n_total // cfg.minibatch)

    for _epoch in range(cfg.epochs):
        if kl_stop:
            break
        np.random.shuffle(idx)
        for start in range(0, n_total, cfg.minibatch):
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
        "max_updates": max_updates,
    }


# ============================================================
# 确定性评估：只取 μ（高斯最高点），10 局不同 reset seed
#   obs_rms 只 normalize、不 update（冻结训练统计量）
#   DR 开启时 10 局初态/动力学不同，std 有意义（不再是 ±0.0）
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
