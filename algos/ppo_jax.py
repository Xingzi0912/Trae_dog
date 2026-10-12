"""PPO 训练（JAX/Flax 版，决策 6 + 师生架构阶段 1：训教师）

教师网络输入 99 维 = 61 标准观测 + 38 特权信息。
学生蒸馏见 distill_jax.py。

PPO 配方（沿用旧 PyTorch 版验证过的超参，决策 6）：
    - GAE (gamma=0.99, lambda=0.95)
    - clip ratio 0.2，value clip 0.2
    - 熵系数 0.003（防止 σ 膨胀）
    - KL 早停 0.02（防止策略灾难性跳跃）
    - value loss coef 0.5
    - lr 线性衰减（anneal_steps 独立于 total_steps）
    - obs_rms 在线归一化（IMU 量级 ~10 vs 关节角 ~1）

关键修复（2026-10-12）：
    1. log_prob 一致性：rollout 和 loss 统一用 gaussian_log_prob（含常数项），
       旧版 rollout 缺 -0.5*action_dim*log(2π)，导致 KL 常年 ~10.5 假早停
    2. JIT 编译：ppo_grad / actor_forward 全部 jit，避免每步重新 trace
    3. KL 早停除数修正：按实际完成的 minibatch 数计平均
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from envs.dog_env_mjx import DogEnvMJX, EnvConfig, tree_where_done
from algos.networks_jax import make_teacher


@dataclass(frozen=True)
class PPOConfig:
    # 优化
    lr: float = 3e-4
    anneal_steps: int = 20_000_000
    max_grad_norm: float = 1.0
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    target_kl: float = 0.02
    # 训练
    n_envs: int = 2048
    rollout_len: int = 24
    n_epochs: int = 10
    n_minibatches: int = 32
    total_steps: int = 20_000_000
    # 保存与日志
    save_interval: int = 50     # 每 N 个 update 存 checkpoint
    log_interval: int = 1       # 每 N 个 update 打印
    save_dir: str = "checkpoints"


class Rollout(NamedTuple):
    obs: jax.Array          # (T, N, 99)
    action: jax.Array       # (T, N, 12)
    log_prob: jax.Array     # (T, N)
    value: jax.Array        # (T, N)
    reward: jax.Array       # (T, N)
    done: jax.Array         # (T, N) bool


class RunningMeanStd(NamedTuple):
    """在线观测归一化（Welford 算法）"""
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @staticmethod
    def create(shape):
        return RunningMeanStd(
            mean=jnp.zeros(shape),
            var=jnp.ones(shape),
            count=jnp.float32(1e-4),
        )

    def update(self, batch):
        batch = batch.reshape(-1, batch.shape[-1])
        batch_mean = jnp.mean(batch, axis=0)
        batch_var = jnp.var(batch, axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / new_count
        new_mean = self.mean + delta * batch_count / new_count
        new_var = M2 / new_count
        return RunningMeanStd(mean=new_mean, var=new_var, count=new_count)

    def normalize(self, x):
        return (x - self.mean) / jnp.sqrt(self.var + 1e-8)


# ---------------------------------------------------------------------------
# 纯函数（可 JIT）
# ---------------------------------------------------------------------------
def make_teacher_obs(obs_std, priv):
    """拼接标准观测和特权信息 → 教师观测 99 维"""
    return jnp.concatenate([obs_std, priv], axis=-1)


def gaussian_log_prob(action, mu, std):
    """对角高斯 log_prob（含常数项，rollout 与 loss 统一用此函数）

    旧版 rollout 缺 -0.5*action_dim*log(2π)，导致 ratio 恒含 e^{10.56} 偏置，
    KL 假信号 ~10.5 触发早停，策略无法学习。
    """
    action_dim = action.shape[-1]
    return (-0.5 * (((action - mu) / std) ** 2).sum(-1)
            - jnp.log(std).sum(-1)
            - 0.5 * action_dim * jnp.log(2 * jnp.pi))


def make_ppo_loss(clip_eps, entropy_coef, value_coef):
    """构造 PPO loss 闭包（超参固化，便于 JIT）"""
    def ppo_loss(params, batch):
        model = make_teacher()
        mu, std, value = model.apply(params, batch["obs"])
        log_prob = gaussian_log_prob(batch["action"], mu, std)
        ratio = jnp.exp(log_prob - batch["old_log_prob"])
        adv = batch["advantage"]
        surr1 = ratio * adv
        surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        policy_loss = -jnp.minimum(surr1, surr2).mean()
        value_pred_clipped = batch["old_value"] + \
            jnp.clip(value - batch["old_value"], -clip_eps, clip_eps)
        value_loss_unclipped = (value - batch["return_val"]) ** 2
        value_loss_clipped = (value_pred_clipped - batch["return_val"]) ** 2
        value_loss = 0.5 * jnp.maximum(value_loss_unclipped,
                                        value_loss_clipped).mean()
        entropy = 0.5 * (jnp.log(2 * jnp.pi * std**2) + 1).sum(-1).mean()
        kl = (batch["old_log_prob"] - log_prob).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        return loss, {"policy_loss": policy_loss, "value_loss": value_loss,
                      "entropy": entropy, "kl": kl}
    return ppo_loss


def compute_gae(rewards, values, dones, last_value,
                gamma=0.99, gae_lambda=0.95):
    """GAE（向量化，按 env 分离，逆序 scan + reverse=True）

    rewards, values, dones: (T, N)
    last_value: (N,) bootstrap value
    返回: advantages (T, N), returns (T, N)
    """
    T, N = rewards.shape

    def scan_fn(carry, t):
        last_gae, last_val = carry
        mask = 1.0 - dones[t].astype(jnp.float32)
        delta = rewards[t] + gamma * last_val * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        return (last_gae, values[t]), last_gae

    _, advantages = jax.lax.scan(
        scan_fn, (jnp.zeros(N), last_value),
        None, length=T, reverse=True)
    returns = advantages + values
    return advantages, returns


def create_train_state(rng, tx):
    model = make_teacher()
    dummy = jnp.zeros((1, 99))
    params = model.init(rng, dummy)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def train(rng, env, config: PPOConfig,
          log_fn: Callable[[str], None] = print,
          save_fn: Callable | None = None):
    """PPO 训练主循环

    Args:
        rng: JAX 随机键
        env: DogEnvMJX 实例
        config: PPOConfig
        log_fn: 日志回调（默认 print）
        save_fn: 可选 checkpoint 回调 (ts, obs_rms, update) -> None
    返回: (ts, obs_rms, metrics)
    """
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr),
    )
    rng, init_rng = jax.random.split(rng)
    ts = create_train_state(init_rng, tx)
    obs_rms = RunningMeanStd.create((99,))

    # JIT'd 核心函数（只编译一次，后续 update 复用）
    ppo_loss = make_ppo_loss(config.clip_eps, config.entropy_coef,
                             config.value_coef)
    ppo_grad = jax.jit(jax.value_and_grad(ppo_loss, has_aux=True))
    actor_forward = jax.jit(
        lambda params, obs: make_teacher().apply(params, obs))

    # reset
    rng, reset_rng = jax.random.split(rng)
    state, obs = env.reset(reset_rng)
    priv = env.get_privileged_obs(state)
    teacher_obs = make_teacher_obs(obs, priv)

    n_updates = config.total_steps // (config.n_envs * config.rollout_len)
    batch_size = config.n_envs * config.rollout_len
    log_fn(f"  PPO: n_updates={n_updates}, batch_size={batch_size}, "
           f"mb_size={batch_size // config.n_minibatches}")

    metrics = {"update": [], "reward": [], "entropy": [], "kl": [],
               "value_loss": [], "policy_loss": []}

    for update in range(n_updates):
        t0 = time.time()
        rollout_obs, rollout_act, rollout_lp, rollout_val = [], [], [], []
        rollout_rew, rollout_done = [], []

        # --- rollout 收集（Python 循环，每步走 JIT'd env.step + actor_forward）---
        for t in range(config.rollout_len):
            mu, std, value = actor_forward(ts.params, teacher_obs)
            rng, act_rng = jax.random.split(rng)
            action = mu + std * jax.random.normal(act_rng, mu.shape)
            log_prob = gaussian_log_prob(action, mu, std)
            action = jnp.clip(action, -1.0, 1.0)

            state, obs, reward, term, trunc, info = env.step(state, action)
            done = jnp.logical_or(term, trunc)
            priv = info["privileged_obs"]
            teacher_obs = make_teacher_obs(obs, priv)

            rollout_obs.append(teacher_obs)
            rollout_act.append(action)
            rollout_lp.append(log_prob)
            rollout_val.append(value)
            rollout_rew.append(reward)
            rollout_done.append(done)

            # auto-reset（done 的 env 用 fresh 替换）
            rng, reset_rng = jax.random.split(rng)
            fresh_state, fresh_obs = env.reset(reset_rng)
            state = tree_where_done(done, fresh_state, state)
            obs = tree_where_done(done, fresh_obs, obs)
            priv = env.get_privileged_obs(state)
            teacher_obs = make_teacher_obs(obs, priv)

        # --- GAE ---
        _, _, last_value = actor_forward(ts.params, teacher_obs)
        obs_arr = jnp.stack(rollout_obs)
        act_arr = jnp.stack(rollout_act)
        lp_arr = jnp.stack(rollout_lp)
        val_arr = jnp.stack(rollout_val)
        rew_arr = jnp.stack(rollout_rew)
        done_arr = jnp.stack(rollout_done)

        advantages, returns = compute_gae(
            rew_arr, val_arr, done_arr, last_value,
            config.gamma, config.gae_lambda)

        obs_rms = obs_rms.update(obs_arr)

        # --- PPO update ---
        batch = {
            "obs": obs_arr.reshape(batch_size, -1),
            "action": act_arr.reshape(batch_size, -1),
            "old_log_prob": lp_arr.reshape(batch_size),
            "advantage": advantages.reshape(batch_size),
            "return_val": returns.reshape(batch_size),
            "old_value": val_arr.reshape(batch_size),
        }
        batch["advantage"] = (batch["advantage"] - batch["advantage"].mean()) \
                             / (batch["advantage"].std() + 1e-8)

        last_info = None
        for epoch in range(config.n_epochs):
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, batch_size)
            mb_size = batch_size // config.n_minibatches
            kl_sum = 0.0
            n_mb_done = 0
            for i in range(config.n_minibatches):
                mb_idx = perm[i * mb_size:(i + 1) * mb_size]
                mb = jax.tree_util.tree_map(lambda x: x[mb_idx], batch)
                (loss, info_grad), grads = ppo_grad(ts.params, mb)
                ts = ts.apply_gradients(grads=grads)
                kl_sum += float(info_grad["kl"])
                last_info = info_grad
                n_mb_done += 1
            avg_kl = kl_sum / n_mb_done
            if avg_kl > config.target_kl:
                log_fn(f"    KL 早停 @epoch {epoch}, kl={avg_kl:.4f}")
                break

        # --- metrics ---
        dt = time.time() - t0
        sps = batch_size / dt if dt > 0 else 0
        metrics["update"].append(update)
        metrics["reward"].append(float(rew_arr.mean()))
        metrics["entropy"].append(float(last_info["entropy"]))
        metrics["kl"].append(avg_kl)
        metrics["value_loss"].append(float(last_info["value_loss"]))
        metrics["policy_loss"].append(float(last_info["policy_loss"]))

        if update % config.log_interval == 0:
            log_fn(f"  update {update}/{n_updates}, "
                   f"reward={float(rew_arr.mean()):.3f}, "
                   f"entropy={float(last_info['entropy']):.3f}, "
                   f"kl={avg_kl:.4f}, "
                   f"dt={dt:.1f}s, sps={sps:.0f}")

        if save_fn is not None and (update + 1) % config.save_interval == 0:
            save_fn(ts, obs_rms, update)

    return ts, obs_rms, metrics
