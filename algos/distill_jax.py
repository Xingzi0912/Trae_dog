"""学生网络蒸馏（阶段 2：冻结教师，监督学习拟合学生）

两阶段训练的第二阶段：
    1. 教师网络已由 ppo_jax.train() 训收敛
    2. 冻结教师权重，用教师的动作均值 μ 和价值 V 监督学生
    3. 学生输入只有 61 维标准观测（实机可获取）

蒸馏损失：
    L = α · MSE(μ_student, μ_teacher) + β · MSE(V_student, V_teacher)
    α=1.0, β=0.5（动作拟合为主，价值为辅）

数据来源：
    - 用训练好的教师在环境中跑 rollout，收集 (obs_61, μ_teacher, V_teacher)
    - 或者直接复用 PPO 训练时存的 rollout 数据

关键修复（2026-10-12）：
    1. JIT 编译 distill_grad + teacher/student forward
    2. 统一用 env.get_privileged_obs（旧版 collect_distill_data 内联了
       jax.vmap(env._get_privileged_obs) 与 env.get_privileged_obs 重复）
    3. 加 log_fn / save_fn 回调，与 ppo_jax 接口一致

关键修复（2026-10-16，归一化一致性）：
    旧版蒸馏全程喂**原始**观测：教师是按 obs_rms 归一化观测训练的
    （ppo_jax.jit_rollout 内 (tobs-mean)/sqrt(var)），这里却用原始 obs 求
    教师 μ/V → 目标在分布外；学生也被拟合成"吃原始观测"的策略，而部署端
    npz 存了 obs_mean/obs_var 且推理前归一化 → 三段尺度互不一致。
    现 collect_distill_data 接收教师 obs_rms：教师用完整 99 维，学生用前
    OBS_DIM(61) 维切片（= 教师标准观测部分，与 train_jax 存进 npz 的一致）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from envs.dog_env_mjx import DogEnvMJX, EnvConfig, tree_where_done
from algos.networks_jax import make_teacher_actor, make_teacher_critic, make_student


@dataclass(frozen=True)
class DistillConfig:
    lr: float = 3e-4
    n_envs: int = 2048
    rollout_len: int = 24
    n_epochs: int = 10
    n_minibatches: int = 32
    total_steps: int = 5_000_000
    alpha: float = 1.0   # 动作蒸馏权重
    beta: float = 0.5    # 价值蒸馏权重
    save_interval: int = 50
    log_interval: int = 1


def make_distill_loss(alpha, beta):
    """构造蒸馏 loss 闭包（超参固化，便于 JIT）"""
    def distill_loss(student_params, batch):
        student = make_student()
        mu_s, _, v_s = student.apply(student_params, batch["obs"])
        mu_t = batch["mu_teacher"]
        v_t = batch["v_teacher"]
        action_loss = jnp.mean((mu_s - mu_t) ** 2)
        value_loss = jnp.mean((v_s - v_t) ** 2)
        loss = alpha * action_loss + beta * value_loss
        return loss, {"action_loss": action_loss, "value_loss": value_loss}
    return distill_loss


def _normalize(x, mean, var):
    """obs_rms 归一化：与 ppo_jax.jit_rollout 及部署端 npz（obs_mean/obs_var）同式"""
    return (x - mean) / jnp.sqrt(var + 1e-8)


def collect_distill_data(rng, teacher_ts, env, n_steps,
                         teacher_forward, obs_rms):
    """用教师网络跑 rollout，收集蒸馏数据

    Args:
        obs_rms: 教师训练收敛后的 99 维观测归一化统计量（train_teacher 返回）。
            **必须传入**：教师按归一化观测训练，喂原始观测会让 μ/V 目标跑出
            分布；学生侧取前 env.OBS_DIM 维切片，与部署 npz 的 obs_mean/var 同源。

    返回 dict: obs (n_steps*n_envs, 61)（已按 obs_rms[:61] 归一化）,
              mu_teacher, v_teacher
    """
    obs_list, mu_list, v_list = [], [], []
    rng, reset_rng, k_dr = jax.random.split(rng, 3)
    # 模型级 DR 在 vmap 外层注入（K=1 整批共享一份；K>1 按池共享，见 env 文件头）
    dr_model = env.resample_dr(k_dr)
    mx_model = env.build_model(dr_model)
    state, obs = env.reset(reset_rng, dr_model)
    priv = env.get_privileged_obs(state)
    teacher_obs = jnp.concatenate([obs, priv], axis=-1)

    for _ in range(n_steps):
        # 教师侧用完整 99 维统计量归一化（与 PPO rollout 内的输入一致）
        mu, std, value = teacher_forward(
            teacher_ts.params,
            _normalize(teacher_obs, obs_rms.mean, obs_rms.var))
        obs_list.append(obs)
        mu_list.append(mu)
        v_list.append(value)

        action = jnp.clip(mu, -1.0, 1.0)  # 蒸馏阶段用确定性动作
        state, obs, reward, term, trunc, info = env.step(state, action, mx_model)
        done = jnp.logical_or(term, trunc)
        priv = info["privileged_obs"]
        teacher_obs = jnp.concatenate([obs, priv], axis=-1)

        # auto-reset
        rng, reset_rng = jax.random.split(rng)
        fresh_state, fresh_obs = env.reset(reset_rng, dr_model)
        state = tree_where_done(done, fresh_state, state)
        obs = tree_where_done(done, fresh_obs, obs)
        priv = env.get_privileged_obs(state)
        teacher_obs = jnp.concatenate([obs, priv], axis=-1)

    d = env.OBS_DIM
    return {
        # 学生输入 = 教师观测的前 61 维（标准观测），按同一份 obs_rms 的前 61 维归一化
        "obs": _normalize(jnp.concatenate(obs_list, axis=0),
                          obs_rms.mean[:d], obs_rms.var[:d]),
        "mu_teacher": jnp.concatenate(mu_list, axis=0),
        "v_teacher": jnp.concatenate(v_list, axis=0),
    }


def train_student(rng, teacher_ts, env, obs_rms, config: DistillConfig,
                  log_fn: Callable[[str], None] = print,
                  save_fn: Callable | None = None):
    """蒸馏训练学生网络

    Args:
        obs_rms: 教师训练收敛后的 99 维 / 61 维观测归一化统计量（train_teacher
            返回的 RunningMeanStd，99 维）。数据采集与学生输入都按它归一化，
            与 PPO rollout、部署端 npz 三方对齐（见模块 docstring 关键修复）。
    """
    tx = optax.adam(config.lr)
    student = make_student()
    rng, init_rng = jax.random.split(rng)
    dummy = jnp.zeros((1, 61))
    student_params = student.init(init_rng, dummy)
    ts = TrainState.create(apply_fn=student.apply,
                           params=student_params, tx=tx)

    # JIT'd 核心函数
    distill_loss = make_distill_loss(config.alpha, config.beta)
    distill_grad = jax.jit(jax.value_and_grad(distill_loss, has_aux=True))
    teacher_forward = jax.jit(
        lambda params, obs: (
            *make_teacher_actor().apply(params["actor"], obs),
            make_teacher_critic().apply(params["critic"], obs)))
    student_forward = jax.jit(
        lambda params, obs: make_student().apply(params, obs))

    n_updates = config.total_steps // (config.n_envs * config.rollout_len)
    batch_size = config.n_envs * config.rollout_len
    log_fn(f"  Distill: n_updates={n_updates}, batch_size={batch_size}")

    metrics = {"update": [], "action_loss": [], "value_loss": []}

    for update in range(n_updates):
        t0 = time.time()
        # 收集蒸馏数据
        rng, data_rng = jax.random.split(rng)
        data = collect_distill_data(data_rng, teacher_ts, env,
                                    config.rollout_len, teacher_forward,
                                    obs_rms)

        # 训练学生
        total_samples = data["obs"].shape[0]
        last_info = None
        for epoch in range(config.n_epochs):
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, total_samples)
            mb_size = total_samples // config.n_minibatches
            for i in range(config.n_minibatches):
                mb_idx = perm[i * mb_size:(i + 1) * mb_size]
                mb = jax.tree_util.tree_map(lambda x: x[mb_idx], data)
                (loss, info), grads = distill_grad(ts.params, mb)
                ts = ts.apply_gradients(grads=grads)
                last_info = info

        dt = time.time() - t0
        metrics["update"].append(update)
        metrics["action_loss"].append(float(last_info["action_loss"]))
        metrics["value_loss"].append(float(last_info["value_loss"]))

        if update % config.log_interval == 0:
            log_fn(f"  distill {update}/{n_updates}, "
                   f"action_loss={float(last_info['action_loss']):.5f}, "
                   f"value_loss={float(last_info['value_loss']):.5f}, "
                   f"dt={dt:.1f}s")

        if save_fn is not None and (update + 1) % config.save_interval == 0:
            save_fn(ts, None, update)

    return ts, metrics
