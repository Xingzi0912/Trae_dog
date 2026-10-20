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
    value_clip_eps: float = 1.0  # v9 判决：与 ratio clip 解耦。0.2 时 critic
    # 每 update 最多向目标移 0.2，初始 2.5 偏差关不上→E[adv] 常驻 +3（v8 实测
    # adv_mean +2.49→+3.00）→ 81% 样本 A>0 → ratio clip 单侧饱和 → 梯度死 → 停滞
    entropy_coef: float = 0.003   # 前人 0.01；取更低值防 σ 膨胀（CPU 版教训）
    value_coef: float = 0.5
    target_kl: float = 0.01     # legged_gym 默认；epoch 早停兜底（actor-only）
    target_kl_mb: float = 0.1   # 单 mb 步 KL 上限（基础值，kl_scale_std=True 时
    # 会按当前 σ / 初始 σ 缩放；v12 判决：固定 0.2 阈值在 σ 缓慢收缩期间
    # 让 actor 累积偏离翻倍，u43-102 KL 从 0.16 爆到 5.3 杀翻训练）
    kl_scale_std: bool = True   # True：mb_kl_limit = target_kl_mb × (cur_σ / init_σ)
    # σ 退火（2026-10-16 诊断后新增）：
    #   adam 每步只把 log_std 移动 ≈lr(5e-5)，实测 27 个 update 只从 0.607 到
    #   0.603 ⇒ 整个 406-update run 里学习型 σ 最多降到 ~0.57。而 σ=0.6 对
    #   residual_scale=0.2 意味着 rollout 的动作基本是采样噪声（r_energy=-0.232
    #   ≈ 12·w_energy·σ²=0.218、r_jerk=-0.068 ≈ i.i.d. 预测 -0.087 ⇒ μ 的 RMS
    #   只有 ~0.3），μ 的学习信号被噪声埋掉；且 actor 被 KL 闸门冻结时
    #   log_std 一起冻（它在 actor 参数里）。
    #   改为 host 侧每 update 线性重写 log_std 均值：σ_start → σ_end，逐维
    #   离散度保留（整体平移），σ 不再依赖梯度。
    std_schedule: bool = True
    # v15：σ 钉 0.5，拆除退火定时器。依据：v13/v14 两个 run 都在 σ≈0.5
    # 见顶（v13 u24 σ=0.519、v14 u30 σ=0.499），σ 是唯一在两个 run 间
    # 严格同步的变量；钉 0.5 后 kl_scale_std 阈值恒定
    # 0.1×(0.5/0.607)≈0.082。若中途 kl 再加速则 v16 = n_minibatches 4→32。
    std_start: float = 0.5
    std_end: float = 0.5
    # 退火覆盖的 update 数（之后保持 std_end）；0 或 <0 表示铺满整个 run
    std_anneal_updates: int = 120
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


def make_ppo_loss(clip_eps, entropy_coef, value_coef, value_clip_eps):
    """构造 PPO loss 闭包（超参固化，便于 JIT）

    教师为独立双网：params["actor"] 走 Actor(mu,std)，params["critic"] 走
    Critic(value)。两子树参数完全隔离。
    """
    actor, critic = make_teacher_actor(), make_teacher_critic()

    def ppo_loss(params, batch):
        mu, std = actor.apply(params["actor"], batch["obs"])
        value = critic.apply(params["critic"], batch["obs"])
        log_prob = gaussian_log_prob(batch["action"], mu, std)
        # -- 数值防护 1：clip exp 输入，防 overflow/underflow 成 inf/NaN --
        # v12 u103 NaN 源头：exp(Δ) 当 Δ>30 时 overflow 成 inf，ratio clip 对
        # inf 无效 → policy_loss=inf → gradient=NaN → params 污染
        log_ratio = jnp.clip(log_prob - batch["old_log_prob"], -20.0, 20.0)
        ratio = jnp.exp(log_ratio)
        adv = batch["advantage"]
        # --- per-minibatch advantage 归一化 ---
        # purejaxrl#L227；作用：把 adv 从原始量级（实测 ≈ ±8）拉回 ≈ N(0,1)，
        # 这样同样的 lr/clip_eps/KL 闸门看到的是"每步相对变化"而非绝对量，
        # 有效步长缩 ~adv_std 倍（上次 adv_std≈6.8 → ~7×），单 mb KL 从 0.2~0.43
        # 回到 0.03~0.06 → KL 闸门不再第一步就触发（ep 从 0/4 回升），且 critic
        # 拟合 return 不受影响。v14 新增。
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        surr1 = ratio * adv
        surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        policy_loss = -jnp.minimum(surr1, surr2).mean()
        value_pred_clipped = batch["old_value"] + \
            jnp.clip(value - batch["old_value"], -value_clip_eps, value_clip_eps)
        value_loss_unclipped = (value - batch["return_val"]) ** 2
        value_loss_clipped = (value_pred_clipped - batch["return_val"]) ** 2
        value_loss = 0.5 * jnp.maximum(value_loss_unclipped,
                                        value_loss_clipped).mean()
        entropy = 0.5 * (jnp.log(2 * jnp.pi * std**2) + 1).sum(-1).mean()
        kl = (batch["old_log_prob"] - log_prob).mean()
        # -- 探针：ratio clip / value clip 实际生效比例 --
        # ratio_clip_pct 高 = 策略在大幅移动；>80% + ploss 尖峰 → 自激
        ratio_clip_pct = jnp.float32(
            ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).mean())
        value_clip_pct = jnp.float32(
            (jnp.abs(value - batch["old_value"]) > value_clip_eps).mean())
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        return loss, {"policy_loss": policy_loss, "value_loss": value_loss,
                      "entropy": entropy, "kl": kl,
                      "n_finite": jnp.float32(1.0),
                      "ratio_clip_pct": ratio_clip_pct,
                      "value_clip_pct": value_clip_pct}
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
    """构造 PPO update 的三个独立 JIT 函数 + scan_step 优化

    拆成 rollout / gae_rms / ppo 三段：避免 mega-jit 下局部改动触发整体重编译。
    scan_step 去掉了每步完整 env.reset + get_privileged_obs（原每步对
    全部 2048 env 完整初始化一遍，24 步 = 24 倍浪费），fresh 模板在
    rollout 开始前只算一次。

    2026-10-15：拆分（原为计时诊断）+ scan_step 性能修复；
    2026-10-16：耗时统计已从日志中移除（瓶颈已确认在物理求解器）。
    """
    actor, critic = make_teacher_actor(), make_teacher_critic()
    ppo_loss = make_ppo_loss(config.clip_eps, config.entropy_coef,
                             config.value_coef, config.value_clip_eps)
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    rollout_len = config.rollout_len
    batch_size = config.n_envs * config.rollout_len
    n_mb = config.n_minibatches
    mb_size = batch_size // n_mb

    # ------------------------------------------------------------------
    # Part 1: JIT rollout（scan，fresh_st/fresh_tobs 作为参数传入）
    # ------------------------------------------------------------------
    @jax.jit
    def jit_rollout(state, teacher_obs, rng, params,
                    obs_mean, obs_var, mx_model,
                    fresh_st, fresh_tobs):
        """rollout_len 步 scan，每步 step 后用 tree_where_done 从预计算的
        fresh 模板替换 done 的 env（不再每步完整 env.reset）。"""

        def scan_step(carry, _):
            st, tobs, k, act_clip_cnt = carry
            tobs_n = (tobs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
            mu, std = actor.apply(params["actor"], tobs_n)
            value = critic.apply(params["critic"], tobs_n)
            k, k_act = jax.random.split(k)
            action = mu + std * jax.random.normal(k_act, mu.shape)
            n_clipped = jnp.sum(jnp.abs(action) > 0.999)
            action = jnp.clip(action, -1.0, 1.0)
            log_prob = gaussian_log_prob(action, mu, std)
            st, obs, rew, term, trunc, info = env.step(st, action, mx_model)
            done = jnp.logical_or(term, trunc)
            tobs_new = make_teacher_obs(obs, info["privileged_obs"])
            # 奖励分项：7 项各自对 N 个 env 求均值 → (7,)，供日志归因
            # （顺序 track/yaw/upright/energy/jerk/vz/rp，sum 应等于 reward）
            rterms = jnp.mean(jnp.stack(
                [info["r_track"], info["r_yaw"], info["r_upright"],
                 info["r_energy"], info["r_jerk"], info["r_vz"], info["r_rp"]],
                axis=-1), axis=0)
            # ← 关键优化：从 carry 取预计算的 fresh，不再每步完整 reset 2048 env
            st = tree_where_done(done, fresh_st, st)
            tobs_out = tree_where_done(done, fresh_tobs, tobs_new)
            out = (tobs_n, action, log_prob, value, rew, done, tobs_new, rterms)
            return (st, tobs_out, k, act_clip_cnt + n_clipped), out

        carry_init = (state, teacher_obs, rng, jnp.float32(0))
        carry_end, ro = jax.lax.scan(scan_step, carry_init, None,
                                     length=rollout_len)
        (state, teacher_obs, rng, act_clip_total) = carry_end
        (obs_arr, act_arr, lp_arr, val_arr, rew_arr, done_arr, raw_arr,
         rterms_arr) = ro
        n_action_elts = rollout_len * config.n_envs * act_arr.shape[-1]
        action_clip_pct = act_clip_total / jnp.float32(n_action_elts)
        alive_pct = 1.0 - done_arr.mean()
        return state, teacher_obs, rng, (
            obs_arr, act_arr, lp_arr, val_arr, rew_arr, done_arr, raw_arr,
            rterms_arr, action_clip_pct, alive_pct)

    # ------------------------------------------------------------------
    # Part 2: JIT GAE + obs_rms 更新
    # ------------------------------------------------------------------
    @jax.jit
    def jit_gae_rms(mx_model, teacher_obs, raw_arr, obs_arr, act_arr,
                    lp_arr, val_arr, rew_arr, done_arr,
                    params, obs_mean, obs_var, obs_count):
        tobs_last_n = (teacher_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        last_value = critic.apply(params["critic"], tobs_last_n)
        adv, ret = compute_gae(rew_arr, val_arr, done_arr, last_value,
                               config.gamma, config.gae_lambda)

        flat = raw_arr.reshape(-1, raw_arr.shape[-1])
        b_mean = jnp.mean(flat, axis=0)
        b_var = jnp.var(flat, axis=0)
        b_cnt = flat.shape[0]
        delta = b_mean - obs_mean
        new_count = obs_count + b_cnt
        m2 = (obs_var * obs_count + b_var * b_cnt
              + delta ** 2 * obs_count * b_cnt / new_count)
        obs_mean_new = obs_mean + delta * b_cnt / new_count
        obs_var_new = m2 / new_count
        return adv, ret, obs_mean_new, obs_var_new, new_count

    # ------------------------------------------------------------------
    # Part 3: JIT PPO epochs（scan + mb KL mask + epoch 早停）
    # ------------------------------------------------------------------
    @jax.jit
    def jit_ppo(obs_arr, act_arr, lp_arr, val_arr,
                adv, ret, params, opt_state,
                lr_t, mb_kl_limit, epoch_rngs):
        """epoch_rngs: (n_epochs, 2) JAX 随机键"""
        batch = {
            "obs": obs_arr.reshape(batch_size, -1),
            "action": act_arr.reshape(batch_size, -1),
            "old_log_prob": lp_arr.reshape(batch_size),
            "advantage": adv.reshape(batch_size),
            "return_val": ret.reshape(batch_size),
            "old_value": val_arr.reshape(batch_size),
        }

        def epoch_fn(carry, epoch_rng):
            p, os_, active = carry
            a_in = active
            perm = jax.random.permutation(epoch_rng, batch_size)
            mbs = jax.tree_util.tree_map(
                lambda x: x[perm].reshape(n_mb, mb_size, *x.shape[1:]), batch)

            def mb_fn(c, mb):
                pp, oos, act = c
                (_, info), grads = grad_fn(pp, mb)
                # 每个 leaf 先 .all() 归约成标量，再 jnp.all 汇总——不能用
                # Python 内置 all()（会对 tracer 做 bool()，trace 期即报
                # TracerBoolConversionError）。各 leaf 归约后同形 ()，可 stack。
                is_finite = jnp.all(jnp.stack([
                    jnp.isfinite(g).all()
                    for g in jax.tree_util.tree_leaves(grads)]))
                new_upd, new_oos = tx.update(grads, oos, pp)
                # 非有限时不仅要丢弃该 mb 的 update，还必须**回滚 opt_state**：
                # 脏梯度会让 adam 的 m/v（以及 clip_by_global_norm 的 NaN scale）
                # 永久变 NaN，而 gate 只挡住当步，下一步 is_finite 恢复真值后照
                # 样把 NaN update 落到 params 上（v12 u103 僵死的老路）。整棵
                # opt_state 用 where 逐叶回滚，params 与动量都停在旧值。
                oos = jax.tree_util.tree_map(
                    lambda n, o: jnp.where(is_finite, n, o), new_oos, oos)
                # v18：check-then-apply。pre-step KL 超限的 mb 直接不施加、
                # 并冻结 actor 剩余步——违规步永不落地（v17 及以前为
                # apply-then-check：超限步已落到 params 上才冻结后续，
                # 分布被推移 → 下一 mb pre-KL 更高 → 连锁超限的自激回路，
                # 即"稳定但饿死"的成因，2026-10-20 定位）。
                kl_ok = info["kl"] <= mb_kl_limit
                fire = act * kl_ok * is_finite
                upd = {
                    "actor": jax.tree_util.tree_map(
                        lambda u: u * (lr_t * fire), new_upd["actor"]),
                    "critic": jax.tree_util.tree_map(
                        lambda u: u * lr_t * is_finite, new_upd["critic"]),
                }
                pp = optax.apply_updates(pp, upd)
                act = act * jnp.where(kl_ok, 1.0, 0.0)
                info_f = {**info,
                          "finite_count": is_finite,
                          "kl_overshoot": jnp.where(kl_ok, 0.0, 1.0)}
                return (pp, oos, act), info_f

            (p, os_, act_final), infos = jax.lax.scan(
                mb_fn, (p, os_, active), mbs)
            kl_e = infos["kl"].mean()
            active = act_final
            m_out = {"kl": kl_e,
                     "entropy": infos["entropy"].mean(),
                     "value_loss": infos["value_loss"].mean(),
                     "policy_loss": infos["policy_loss"].mean(),
                     "active": a_in,
                     "finite_count": infos["finite_count"].mean(),
                     "ratio_clip_pct": infos["ratio_clip_pct"].mean(),
                     "value_clip_pct": infos["value_clip_pct"].mean(),
                     "kl_overshoot_pct": infos["kl_overshoot"].mean(),
                     "kl_max": infos["kl"].max()}
            return (p, os_, active), m_out

        (params, opt_state, _), m_ep = jax.lax.scan(
            epoch_fn, (params, opt_state, jnp.float32(1.0)), epoch_rngs)
        return params, opt_state, m_ep

    return jit_rollout, jit_gae_rms, jit_ppo, actor, critic


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def train(rng, env, config: PPOConfig,
          log_fn: Callable[[str], None] = print,
          save_fn: Callable | None = None,
          init_params=None, init_obs_rms=None):
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
        init_params: 可选已训练参数 dict（--resume 续训，据此重建 TrainState；
                 Adam 动量不保存在 checkpoint，从零重热，前 1-2 update
                 步长偏保守属正常）
        init_obs_rms: 与 init_params 配套的已收敛 obs_rms
    返回: (ts, obs_rms, metrics)
    """
    # adam 用 lr=1.0，真实 lr 在 jit_update 内由 lr_t 缩放（数学上等价，
    # 同时支持 lr 衰减与 KL epoch mask）
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(1.0),
    )
    if init_params is not None:
        actor = make_teacher_actor()
        ts = TrainState.create(apply_fn=actor.apply, params=init_params, tx=tx)
        # 与 create_train_state 同样的强类型处理（weak_type 变化会改 jit 缓存键
        # → mega-jit 多编译一次，见 create_train_state docstring）
        ts = ts.replace(
            params=jax.tree_util.tree_map(_to_strong, ts.params),
            opt_state=jax.tree_util.tree_map(_to_strong, ts.opt_state))
        obs_rms = init_obs_rms
        log_fn("  --resume：已从 checkpoint 恢复 params + obs_rms")
    else:
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
               "value_loss": [], "policy_loss": [],
               "adv_mean": [], "adv_std": []}

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

    # --- 三 JIT 函数（拆 mega-jit，避免局部重编译）---
    jit_rollout, jit_gae_rms, jit_ppo, actor, critic = make_update(
        env, config, tx)
    log_fn("  三 JIT 函数编译中（jit_rollout / jit_gae_rms / jit_ppo，"
           "各一次性，各较慢）...")

    params, opt_state = ts.params, ts.opt_state
    lr_cur = float(config.lr)

    # -- 捕获初始 log_std（kl_scale_std 缩放基准）--
    init_log_std = float(jax.device_get(
        params["actor"]["params"]["log_std"].mean()))
    log_fn(f"  初始 σ：{jnp.exp(jnp.float32(init_log_std)):.3f}")

    # σ 退火：host 侧重写 actor 的 log_std（整体平移，保留逐维离散度）。
    # 副作用：params 由 FrozenDict 变普通 dict，pytree 类型变一次 ⇒ 三个 jit
    # 各多编译一次，此后结构稳定。σ 因此不再受 KL 冻结影响（原来 log_std 在
    # actor 参数里，被 act 掩码一起冻住）。
    def _with_log_std(p, log_std):
        return {**p, "actor": {**p["actor"],
                               "params": {**p["actor"]["params"],
                                          "log_std": log_std}}}

    anneal_n = (config.std_anneal_updates
                if config.std_anneal_updates > 0 else n_updates)
    # 续训时 checkpoint 的 σ 可能已低于 std_start：退火起点取当前 σ，保证单调不升
    std_start_eff = min(config.std_start,
                        float(jnp.exp(jnp.float32(init_log_std))))
    if config.std_schedule:
        log_fn(f"  σ 退火：{std_start_eff:.3f} → {config.std_end:.3f}，"
               f"在 {anneal_n} 个 update 内线性，之后保持（σ 不再依赖梯度）")

    for update in range(n_updates):
        # ---- host 侧开销：DR 重采样 + fresh 模板预算 + KL 阈值 ----
        rng, k_dr = jax.random.split(rng)
        dr_model = env.sample_model_dr(k_dr)
        mx_model = env.build_model(dr_model)

        if config.std_schedule:
            std_t = std_start_eff + (config.std_end - std_start_eff) * min(
                1.0, update / max(1, anneal_n - 1))
            ls_old = params["actor"]["params"]["log_std"]
            params = _with_log_std(
                params, ls_old + (jnp.log(jnp.float32(std_t)) - ls_old.mean()))

        cur_log_std = float(jax.device_get(
            params["actor"]["params"]["log_std"].mean()))
        if config.kl_scale_std:
            mb_kl_limit_dyn = config.target_kl_mb * (
                jnp.exp(jnp.float32(cur_log_std)) /
                jnp.exp(jnp.float32(init_log_std)))
        else:
            mb_kl_limit_dyn = jnp.float32(config.target_kl_mb)
        lr_t = jnp.float32(lr_cur)

        # 预算 fresh 模板（一次 env.reset，scan 内部用 tree_where_done 替换）
        # 注意：所有 done env 都用同一份 fresh 初始状态模板（近似不影响训练）
        rng, k_fresh = jax.random.split(rng)
        fresh_st, fresh_obs = env.reset(k_fresh, dr_model)
        fresh_tobs = make_teacher_obs(
            fresh_obs, env.get_privileged_obs(fresh_st))

        rng, k_epoch = jax.random.split(rng)
        epoch_rngs = jax.random.split(k_epoch, config.n_epochs)

        # ==============================================================
        # [阶段 1] rollout
        # ==============================================================
        state, teacher_obs, rng, ro = jit_rollout(
            state, teacher_obs, rng, params,
            obs_rms.mean, obs_rms.var, mx_model,
            fresh_st, fresh_tobs)
        obs_arr, act_arr, lp_arr, val_arr, rew_arr, done_arr, raw_arr, \
            rterms_arr, action_clip_pct, alive_pct = ro

        # ==============================================================
        # [阶段 2] GAE + RMS
        # ==============================================================
        adv, ret, o_mean, o_var, o_count = jit_gae_rms(
            mx_model, teacher_obs, raw_arr, obs_arr, act_arr,
            lp_arr, val_arr, rew_arr, done_arr,
            params, obs_rms.mean, obs_rms.var, obs_rms.count)

        # ==============================================================
        # [阶段 3] PPO epochs
        # ==============================================================
        params, opt_state, m_ep = jit_ppo(
            obs_arr, act_arr, lp_arr, val_arr,
            adv, ret, params, opt_state,
            lr_t, mb_kl_limit_dyn, epoch_rngs)
        # host 同步（最终 metrics 汇总在后面再 device_get）
        m_ep_arr = jax.device_get(m_ep)
        adv_mean = float(jax.device_get(adv.mean()))
        adv_std = float(jax.device_get(adv.std()))
        vmean = float(jax.device_get(val_arr.mean()))
        vmin = float(jax.device_get(val_arr.min()))
        vmax = float(jax.device_get(val_arr.max()))

        obs_rms = RunningMeanStd(mean=o_mean, var=o_var, count=o_count)
        ts = ts.replace(params=params, opt_state=opt_state)

        # host 侧组装 metrics（reward 从 rew_arr 取，adv 从上面算好的）
        m = {
            "reward": float(jax.device_get(rew_arr.mean())),
            "adv_mean": adv_mean, "adv_std": adv_std,
            "adv_p99": float(jax.device_get(
                jnp.percentile(adv.reshape(-1), 99))),
            "adv_tail_pct": float(jax.device_get(
                (jnp.abs(adv.reshape(-1)) > 2 * adv_std).mean())),
            "vmean": vmean, "vmin": vmin, "vmax": vmax,
            "action_clip_pct": float(jax.device_get(action_clip_pct)),
            "alive_pct": float(jax.device_get(alive_pct)),
            # 奖励分项：整条 rollout 的 T×N 均值，(7,) 顺序
            # track/yaw/upright/energy/jerk/vz/rp
            "rterms": jax.device_get(rterms_arr.mean(axis=0)),
            **m_ep_arr,
        }

        # -- NaN 检查：定位是哪个指标 + 崩溃现场快照 --
        probe = {"reward": m["reward"], "kl": m["kl"], "entropy": m["entropy"],
                 "value_loss": m["value_loss"], "policy_loss": m["policy_loss"]}
        p_arr = {k: jax.device_get(jnp.asarray(v, jnp.float32))
                 for k, v in probe.items()}
        bad = [k for k, v in p_arr.items()
               if not bool(jnp.all(jnp.isfinite(v)))]
        if bad:
            snap = {k: (float(jnp.nanmean(v))
                        if bool(jnp.any(jnp.isfinite(v))) else float("nan"))
                    for k, v in p_arr.items()}
            raise RuntimeError(
                f"update {update}: NaN/inf 指标={bad}; 各指标均值={snap}; "
                f"adv_p99={m['adv_p99']:.3f} vmean={m['vmean']:.3f} "
                f"vmin={m['vmin']:.1f} vmax={m['vmax']:.1f} "
                f"kl_max={float(jnp.nanmax(jnp.asarray(m['kl_max'], jnp.float32))):.3f}")

        # -- 逐 epoch 指标 --
        n_active = max(1, int(m["active"].sum()))
        idx = n_active - 1
        kl = float(m["kl"][idx])
        ent = float(m["entropy"][idx])
        vloss = float(m["value_loss"][idx])
        ploss = float(m["policy_loss"][idx])
        ratio_clip = float(m["ratio_clip_pct"][idx])
        value_clip = float(m["value_clip_pct"][idx])
        kl_overshoot = float(m["kl_overshoot_pct"][idx])
        kl_max = float(m["kl_max"][idx])
        cur_std = float(jnp.exp(jnp.float32(cur_log_std)))
        # KL 冻结事件可见性：kl_overshoot_pct 非零 ⟺ 该 update 里 actor 被
        # KL 上限打断过。每次 update 至多一个 epoch 会超限（超限的那个 mb
        # 仍会施加，随后 act 置 0→后续 mb 的 kl=0），所以该 epoch 的占比恒为
        # 1/n_minibatches（4 个 mb 时即 25%），0% = 全程没触发。
        # idx = 最后一个"起始时未冻结"的 epoch = 冻结发生的 epoch，故
        # ep=0/4 表示 epoch 0 就被冻（actor 只拿到 1/5 梯度步，v12 最坏情形），
        # ep=4/4 表示冻结最晚发生或没发生。
        # 非有限梯度事件可见性：finite_count 是 epoch 均值，<1 即有 mb 被打回。
        # 该事件只回滚当步 opt_state + update（见 jit_ppo.mb_fn），params 保持
        # 干净，所以仅告警不中断训练；持续出现才是防护失效的信号。
        finite_min = float(m["finite_count"].min())
        if finite_min < 1.0:
            log_fn(f"  ⚠️ update {update}: minibatch 梯度非有限"
                   f"（已丢弃该步 update 与 opt_state，params 未污染）"
                   f" finite_min={finite_min:.3f}")

        metrics["update"].append(update)
        metrics["reward"].append(m["reward"])
        metrics["entropy"].append(ent)
        metrics["kl"].append(kl)
        metrics["value_loss"].append(vloss)
        metrics["policy_loss"].append(ploss)
        metrics["adv_mean"].append(adv_mean)
        metrics["adv_std"].append(adv_std)

        if update % config.log_interval == 0:
            log_fn(
                f"  update {update}/{n_updates}, "
                f"reward={m['reward']:.3f}, "
                f"entropy={ent:.3f}, kl={kl:.4f}, "
                f"vloss={vloss:.3f}, ploss={ploss:.4f}, "
                f"adv={adv_mean:+.3f}±{adv_std:.3f}, "
                f"σ={cur_std:.3f}, kl_lim={float(mb_kl_limit_dyn):.3f}, "
                f"kl_ovr={kl_overshoot*100:.0f}%, ep={idx}/{config.n_epochs - 1}, "
                f"clip[R={ratio_clip*100:.0f}% V={value_clip*100:.0f}% A={float(m['action_clip_pct'])*100:.1f}%], "
                f"finite={finite_min*100:.0f}%, "
                f"alive={float(m['alive_pct'])*100:.1f}%")
            # —— 奖励归因：7 项分项均值（sum 应与上面的 reward 相等）——
            rt = [float(v) for v in m["rterms"]]
            log_fn(f"      └ 奖励分项 track={rt[0]:.3f} yaw={rt[1]:.3f} "
                   f"upright={rt[2]:.3f} energy={rt[3]:.3f} "
                   f"jerk={rt[4]:.3f} vz={rt[5]:.3f} rp={rt[6]:.3f} "
                   f"| sum={sum(rt):.3f}")

        if save_fn is not None and (update + 1) % config.save_interval == 0:
            save_fn(ts, obs_rms, update)

    return ts, obs_rms, metrics
