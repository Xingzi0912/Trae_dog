"""PPO 训练（JAX/Flax 版，决策 6 + 师生架构阶段 1：训教师）

教师网络输入 99 维 = 61 标准观测 + 38 特权信息。
学生蒸馏见 distill_jax.py。

PPO 配方（2026-10-14 对齐 rsl_rl/legged_gym 成熟配置，Go1/Go2/ANYmal 事实标准）：
    - GAE (gamma=0.99, lambda=0.95)
    - clip ratio 0.2，value clip 0.2
    - 熵系数 0.003（比前人 0.01 更保守，防 σ 膨胀）
    - KL 早停 0.01（epoch 级，actor-only 兜底）
    - 自适应 lr（起点 1e-3，KL>2*target 减半 / <target/2 回升，替代线性退火）
    - value loss coef 0.5
    - obs_rms 在线归一化（IMU 量级 ~10 vs 关节角 ~1）

关键修复（2026-10-12）：
    1. log_prob 一致性：rollout 和 loss 统一用 gaussian_log_prob（含常数项），
       旧版 rollout 缺 -0.5*action_dim*log(2π)，导致 KL 常年 ~10.5 假早停
    2. JIT 编译：ppo_grad / actor_forward 全部 jit，避免每步重新 trace
    3. KL 早停除数修正：按实际完成的 minibatch 数计平均





    
关键修复（2026-10-13，mega-jit 重构）：
    旧版"scan rollout + Python 端 epoch/minibatch 循环"在 4090 服务器上
    每个 update 触发一次完整重编译（dt≈19~26 分钟，remat 警告字节数逐字节
    相同），且每 minibatch float(kl) 造成 host-device 同步。现把
    rollout + GAE + n_epochs×n_minibatches 包进单个 jit（Brax 风格），
    整个训练只编译一次；KL 早停改为 epoch mask（超阈值后 lr 置零，参数
    冻结，语义与 Python break 等价）。2026-10-14 起改用 host 侧自适应 lr。
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
from algos.networks_jax import make_teacher_actor, make_teacher_critic


@dataclass(frozen=True)
class PPOConfig:
    # 优化 —— 2026-10-14 对齐 rsl_rl/legged_gym 成熟配置（Go1/Go2/ANYmal 事实标准）
    # 固定 lr（v4 实测定标）：1e-4×80步/update 12 update 自激；~5e-6 被 KL
    # floor 冻死；adaptive lr 因 KL 对 lr 噪声主导（单样本 kl_mean 驱动乘性
    # 调整，减半/回升增益不对称）退化为几何随机游走净向下，弃用。
    # 5e-5 = 平衡点，每 update μ 漂移~0.04/dim，epoch 早停兜底单 update 爆炸。
    lr: float = 5e-5
    lr_min: float = 1e-6
    lr_max: float = 1e-2
    anneal_steps: int = 0       # 已弃用（保留字段兼容旧 checkpoint 脚本）
    max_grad_norm: float = 1.0
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.003   # 前人 0.01；取更低值防 σ 膨胀（CPU 版教训）
    value_coef: float = 0.5
    target_kl: float = 0.01     # legged_gym 默认；epoch 早停兜底（actor-only）
    # 训练
    n_envs: int = 2048
    rollout_len: int = 24
    n_epochs: int = 5           # legged_gym 默认
    n_minibatches: int = 4      # legged_gym 默认
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
    """构造 PPO loss 闭包（超参固化，便于 JIT）

    教师为独立双网：params["actor"] 走 Actor(mu,std)，params["critic"] 走
    Critic(value)。两子树参数完全隔离。
    """
    actor, critic = make_teacher_actor(), make_teacher_critic()

    def ppo_loss(params, batch):
        mu, std = actor.apply(params["actor"], batch["obs"])
        value = critic.apply(params["critic"], batch["obs"])
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
        jnp.arange(T), reverse=True)
    returns = advantages + values
    return advantages, returns


def _to_strong(x):
    """float 叶子统一转强类型（weak_type=False）float32。

    jnp.zeros/ones/full 以及 Flax/optax 的初始化器都可能产出 weak_type=True 的
    数组；只要第一次 optimizer update 把它们变成强类型，jit 的缓存键（含
    avals.weak_type）就变了 → 整个 mega-jit 再编译一次（实测 128 env 下
    update 0/1 各 18.2s/17.3s，之后才落到 6.1s）。把初始 params/opt_state 强类型
    化后，jit_update 全程只编译一次。非 float 叶子（如 adam 的 int32 count）原样保留。
    """
    if jnp.issubdtype(x.dtype, jnp.floating):
        return jnp.asarray(np.asarray(x, dtype=np.float32))
    return x


def create_train_state(rng, tx):
    actor, critic = make_teacher_actor(), make_teacher_critic()
    # dummy 也要强类型：jnp.zeros 是弱类型，而 Flax 以输入 dtype 决定参数 dtype
    dummy = jnp.array(np.zeros((1, 99), dtype=np.float32))
    rng, rng_c = jax.random.split(rng)
    params = {"actor": actor.init(rng, dummy),
              "critic": critic.init(rng_c, dummy)}
    ts = TrainState.create(apply_fn=actor.apply, params=params, tx=tx)
    return ts.replace(
        params=jax.tree_util.tree_map(_to_strong, ts.params),
        opt_state=jax.tree_util.tree_map(_to_strong, ts.opt_state))


def make_update(env, config: PPOConfig, tx):
    """构造单次 JIT 编译的完整 PPO update（Brax 风格 mega-jit）

    rollout + GAE + n_epochs×n_minibatches 全部在一次 jit 内完成：
    - 整个训练只编译一次，杜绝 per-update 重编译（2026-10-13 实测旧结构
      每个 update 重编译 scan_rollout，dt≈19~26 分钟）
    - 消除 Python 循环的全部 host-device 同步（旧版每 minibatch
      float(kl) 同步一次，32 次/update）
    - KL 早停 = epoch mask：某 epoch 平均 KL 超 target_kl 后，后续 epoch
      **仅 actor** 冻结（梯度置 0）；critic 在全部 epoch 继续更新——
      value 拟合不受策略 KL 约束（2026-10-14，修复 vloss 横在 O(10) 的根因）

    jit_update 参数（签名跨 update 完全稳定，不会重编译）：
        state, teacher_obs, rng, params, opt_state,
        obs_mean, obs_var, obs_count, lr_t, dr_model
    模型级 DR（dr_model）：由 train() 每个 update 重采样的**标量** DRParams，
    在 jit 内经 env.build_model 构建成**非批化**单模型、以 in_axes=None 广播给
    整批 env——模型一旦按 env 批化，mjx.step 快路径即失效（慢 31 倍，见
    dog_env_mjx.py 文件头）。这里必须走参数而非闭包常量，否则每次重采样都会让
    常量变化 → mega-jit 重编译。
    返回：(state, teacher_obs, rng, params, opt_state,
           obs_mean, obs_var, obs_count) 8 元 carry, metrics
        metrics: reward 标量；kl/entropy/value_loss/policy_loss/active
                 为 (n_epochs,) 逐 epoch 数组。active[e]=1 表示 epoch e
                 实际参与了更新（0 = 已被 KL mask 冻结）。
    """
    actor, critic = make_teacher_actor(), make_teacher_critic()
    ppo_loss = make_ppo_loss(config.clip_eps, config.entropy_coef,
                             config.value_coef)
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    rollout_len = config.rollout_len
    batch_size = config.n_envs * config.rollout_len
    n_mb = config.n_minibatches
    mb_size = batch_size // n_mb
    target_kl = config.target_kl

    @jax.jit
    def jit_update(state, teacher_obs, rng, params, opt_state,
                   obs_mean, obs_var, obs_count, lr_t, dr_model):
        # 模型级 DR 在 vmap 外层注入：K=1 得非批化单模型、K>1 得按池批化 (K,...)
        # 模型（均保住 mjx.step 快路径，见 envs/dog_env_mjx.py 文件头「关键差异 6」）
        mx_model = env.build_model(dr_model)

        # ---------- 1. rollout（obs_rms 本批内固定）----------
        def scan_step(carry, _):
            st, tobs, k = carry
            # normalize（rollout 期间固定，结束后才更新）
            tobs_n = (tobs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
            mu, std = actor.apply(params["actor"], tobs_n)
            value = critic.apply(params["critic"], tobs_n)
            # 先 clip 再算 log_prob（与 ppo_loss 一致）
            k, k_act, k_reset = jax.random.split(k, 3)
            action = mu + std * jax.random.normal(k_act, mu.shape)
            action = jnp.clip(action, -1.0, 1.0)
            log_prob = gaussian_log_prob(action, mu, std)
            # env step + auto-reset
            st, obs, rew, term, trunc, info = env.step(st, action, mx_model)
            done = jnp.logical_or(term, trunc)
            tobs_new = make_teacher_obs(obs, info["privileged_obs"])
            fresh_st, fresh_obs = env.reset(k_reset, dr_model)
            fresh_tobs = make_teacher_obs(fresh_obs,
                                          env.get_privileged_obs(fresh_st))
            st = tree_where_done(done, fresh_st, st)
            tobs_out = tree_where_done(done, fresh_tobs, tobs_new)
            out = (tobs_n, action, log_prob, value, rew, done, tobs_new)
            return (st, tobs_out, k), out

        (state, teacher_obs, rng), ro = jax.lax.scan(
            scan_step, (state, teacher_obs, rng), None, length=rollout_len)
        obs_arr, act_arr, lp_arr, val_arr, rew_arr, done_arr, raw_arr = ro

        # ---------- 2. GAE ----------
        tobs_last_n = (teacher_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        last_value = critic.apply(params["critic"], tobs_last_n)
        adv, ret = compute_gae(rew_arr, val_arr, done_arr, last_value,
                               config.gamma, config.gae_lambda)

        # ---------- 3. obs_rms 更新（本批原始观测，供下一 update 用）----------
        flat = raw_arr.reshape(-1, raw_arr.shape[-1])
        b_mean = jnp.mean(flat, axis=0)
        b_var = jnp.var(flat, axis=0)
        b_cnt = flat.shape[0]
        delta = b_mean - obs_mean
        obs_count = obs_count + b_cnt
        m2 = (obs_var * (obs_count - b_cnt) + b_var * b_cnt
              + delta ** 2 * (obs_count - b_cnt) * b_cnt / obs_count)
        obs_mean = obs_mean + delta * b_cnt / obs_count
        obs_var = m2 / obs_count

        # ---------- 4. PPO epochs（scan + KL mask）----------
        batch = {
            "obs": obs_arr.reshape(batch_size, -1),
            "action": act_arr.reshape(batch_size, -1),
            "old_log_prob": lp_arr.reshape(batch_size),
            "advantage": adv.reshape(batch_size),
            "return_val": ret.reshape(batch_size),
            "old_value": val_arr.reshape(batch_size),
        }
        a = batch["advantage"]
        batch["advantage"] = (a - a.mean()) / (a.std() + 1e-8)

        def epoch_fn(carry, epoch_rng):
            params, opt_state, active = carry
            a_in = active
            lr_eff = lr_t  # critic 恒定学习率（不随 KL mask 冻结）
            lr_actor = lr_t * active  # KL mask：active=0 时仅 actor 冻结
            perm = jax.random.permutation(epoch_rng, batch_size)
            mbs = jax.tree_util.tree_map(
                lambda x: x[perm].reshape(n_mb, mb_size, *x.shape[1:]),
                batch)

            def mb_fn(c, mb):
                p, os_ = c
                (_, info), grads = grad_fn(p, mb)
                upd, os_ = tx.update(grads, os_, p)
                # tx 内 adam lr=1.0，在此按 lr 缩放：KL mask 只作用于
                # actor 子树，critic 全程学习（2026-10-14 vloss 修复核心）
                upd = {
                    "actor": jax.tree_util.tree_map(
                        lambda u: u * lr_actor, upd["actor"]),
                    "critic": jax.tree_util.tree_map(
                        lambda u: u * lr_eff, upd["critic"]),
                }
                p = optax.apply_updates(p, upd)
                return (p, os_), info

            (params, opt_state), infos = jax.lax.scan(
                mb_fn, (params, opt_state), mbs)
            kl_e = infos["kl"].mean()
            active = active * (kl_e <= target_kl).astype(jnp.float32)
            m_out = {"kl": kl_e, "entropy": infos["entropy"].mean(),
                     "value_loss": infos["value_loss"].mean(),
                     "policy_loss": infos["policy_loss"].mean(),
                     "active": a_in}
            return (params, opt_state, active), m_out

        rng, k_epoch = jax.random.split(rng)
        epoch_rngs = jax.random.split(k_epoch, config.n_epochs)
        (params, opt_state, _), m_ep = jax.lax.scan(
            epoch_fn, (params, opt_state, jnp.float32(1.0)), epoch_rngs)

        metrics = {"reward": rew_arr.mean(), **m_ep}
        return (state, teacher_obs, rng, params, opt_state,
                obs_mean, obs_var, obs_count), metrics

    return jit_update


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def train(rng, env, config: PPOConfig,
          log_fn: Callable[[str], None] = print,
          save_fn: Callable | None = None):
    """PPO 训练主循环（mega-jit 版）

    单 update = 一次 jit_update 调用（rollout+GAE+epochs 全在 GPU 完成）。
    自适应 lr（rsl_rl 标准）：每 update 按活跃 epoch 的 KL 均值调整，
    KL>2*target 减半、KL<target/2 回升 1.25 倍，替代旧线性退火。

    Args:
        rng: JAX 随机键
        env: DogEnvMJX 实例
        config: PPOConfig
        log_fn: 日志回调（默认 print）
        save_fn: 可选 checkpoint 回调 (ts, obs_rms, update) -> None
    返回: (ts, obs_rms, metrics)
    """
    # adam 用 lr=1.0，真实 lr 在 jit_update 内由 lr_t 缩放（数学上等价，
    # 同时支持 lr 衰减与 KL epoch mask）
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(1.0),
    )
    rng, init_rng = jax.random.split(rng)
    ts = create_train_state(init_rng, tx)
    obs_rms = RunningMeanStd.create((99,))

    actor_forward = jax.jit(
        lambda params, obs: make_teacher_actor().apply(params["actor"], obs))

    # reset（先采样本 rollout 的 K 份模型级 DR；K=1 即整批 env 共享一份）
    rng, reset_rng, k_dr = jax.random.split(rng, 3)
    dr_model = env.resample_dr(k_dr)
    state, obs = env.reset(reset_rng, dr_model)
    priv = env.get_privileged_obs(state)
    teacher_obs = make_teacher_obs(obs, priv)

    n_updates = config.total_steps // (config.n_envs * config.rollout_len)
    batch_size = config.n_envs * config.rollout_len
    log_fn(f"  PPO: n_updates={n_updates}, batch_size={batch_size}, "
           f"mb_size={batch_size // config.n_minibatches}")

    metrics = {"update": [], "reward": [], "entropy": [], "kl": [],
               "value_loss": [], "policy_loss": []}

    # --- obs_rms warmup（避免第一次 rollout 用 mean=0/var=1 等于没归一化）---
    # teacher_obs 含 IMU(~10) / 关节角(~1) / 特权信息，尺度差异大；
    # 若不 warmup，第一次梯度更新会让网络输出剧变 → 第二次 rollout KL 偏高
    log_fn("  obs_rms warmup...")
    mx_model = env.build_model(dr_model)
    for _ in range(config.rollout_len):
        teacher_obs_norm = obs_rms.normalize(teacher_obs)
        mu, std = actor_forward(ts.params, teacher_obs_norm)
        rng, act_rng = jax.random.split(rng)
        action = mu + std * jax.random.normal(act_rng, mu.shape)
        action = jnp.clip(action, -1.0, 1.0)
        state, obs, _, _, _, info = env.step(state, action, mx_model)
        priv = info["privileged_obs"]
        teacher_obs = make_teacher_obs(obs, priv)
        obs_rms = obs_rms.update(teacher_obs)
    # warmup 后 reset，避免初始 transient 状态污染第一个真实 rollout
    rng, reset_rng = jax.random.split(rng)
    state, obs = env.reset(reset_rng, dr_model)
    priv = env.get_privileged_obs(state)
    teacher_obs = make_teacher_obs(obs, priv)
    log_fn(f"  obs_rms warmup done: mean|obs|={float(jnp.abs(obs_rms.mean).mean()):.3f}, "
           f"mean|var|={float(obs_rms.var.mean()):.3f}")

    # mega-jit：rollout+GAE+epochs 单次编译，整个训练只编译一次
    jit_update = make_update(env, config, tx)
    log_fn("  jit_update 编译中（一次性，会较慢）...")

    params, opt_state = ts.params, ts.opt_state
    lr_cur = float(config.lr)   # 固定 lr（adaptive 因 KL 噪声主导弃用，见 PPOConfig）
    for update in range(n_updates):
        t0 = time.time()
        # 每个 update 重采样 K 份模型级 DR（跨 update 覆盖 DR 分布；
        # 显式传参而非闭包常量，保证 mega-jit 不重编译）
        rng, k_dr = jax.random.split(rng)
        dr_model = env.sample_model_dr(k_dr)

        lr_t = jnp.float32(lr_cur)
        carry_out, m = jit_update(
            state, teacher_obs, rng, params, opt_state,
            obs_rms.mean, obs_rms.var, obs_rms.count, lr_t, dr_model)
        (state, teacher_obs, rng, params, opt_state,
         o_mean, o_var, o_count) = carry_out
        obs_rms = RunningMeanStd(mean=o_mean, var=o_var, count=o_count)
        ts = ts.replace(params=params, opt_state=opt_state)
        m = jax.device_get(m)   # 每 update 仅这一次 host 同步

        dt = time.time() - t0
        sps = batch_size / dt if dt > 0 else 0

        # 逐 epoch 指标：取最后一个活跃 epoch（被 mask 冻结的 epoch 参数
        # 不变，其 kl 恒等于冻结点值；活跃序列末尾即触发早停的那个 epoch）
        n_active = max(1, int(m["active"].sum()))
        kl = float(m["kl"][n_active - 1])
        ent = float(m["entropy"][n_active - 1])
        vloss = float(m["value_loss"][n_active - 1])
        ploss = float(m["policy_loss"][n_active - 1])
        if n_active < config.n_epochs:
            log_fn(f"    KL 早停 @epoch {n_active - 1}, kl={kl:.4f}")

        # KL 监控（仅记录，不再驱动 lr 调整；epoch 早停兜底单 update 爆炸）

        metrics["update"].append(update)
        metrics["reward"].append(float(m["reward"]))
        metrics["entropy"].append(ent)
        metrics["kl"].append(kl)
        metrics["value_loss"].append(vloss)
        metrics["policy_loss"].append(ploss)

        if update % config.log_interval == 0:
            log_fn(f"  update {update}/{n_updates}, "
                   f"reward={float(m['reward']):.3f}, "
                   f"entropy={ent:.3f}, kl={kl:.4f}, "
                   f"vloss={vloss:.3f}, ploss={ploss:.4f}, "
                   f"lr={lr_cur:.2e}, dt={dt:.1f}s, sps={sps:.0f}")

        if save_fn is not None and (update + 1) % config.save_interval == 0:
            save_fn(ts, obs_rms, update)

    return ts, obs_rms, metrics
