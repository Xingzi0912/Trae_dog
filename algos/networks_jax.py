"""Actor-Critic 网络（JAX/Flax 版，决策 6 + 师生架构）

结构沿用 PyTorch 版验证过的配方：
    - Actor / Critic 独立双网（不共享 trunk）
    - 对角高斯策略，log_std 状态无关（全局可学习参数，初始 -0.5 → σ≈0.61）
    - μ 头无激活（输出全体实数，环境端 clip）
    - V 头线性无界

师生架构：
    - Teacher: obs_dim = 61（标准）+ 38（特权）= 99
    - Student: obs_dim = 61（实机可获取）
    - 两者结构完全相同，仅输入维度不同

checkpoint 双存：
    - Flax 原生（train_state，续训用）
    - .npz（部署用，MLP 权重 + obs_rms，NUC 端 numpy 推理）
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


class Actor(nn.Module):
    """对角高斯策略网络（独立参数，PPO 教师用）

    2026-10-14 从 ActorCritic 拆出：KL 早停只 mask actor 梯度、critic 全速
    学习（value 拟合不受策略 KL 约束），需要 params 树按 {"actor","critic"}
    分子树才能在 jit 内逐子树 mask。
    """
    obs_dim: int
    action_dim: int
    hidden: int = 512
    init_log_std: float = -0.5

    @nn.compact
    def __call__(self, obs):
        a = nn.Dense(self.hidden)(obs)
        a = nn.relu(a)
        a = nn.Dense(self.hidden)(a)
        a = nn.relu(a)
        # v19：μ 头对齐 PureJaxRL 演员头 orthogonal(0.01)。Flax 默认
        # lecun_normal 行范数≈1，μ 敏感度实测高 38×；KL/步 ∝ (lr×增益)²，
        # v15 爆炸（等效 PJRL lr 1.9e-3）、v17/v18 饥饿（等效 3.8e-4）、
        # clip[A] 起始 11.9% 与 μ 外漂 NaN，同此一因。
        mu = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01))(a)

        # 全局 log_std（状态无关）
        # 必须用**强类型**常量初始化：jnp.full / jnp.zeros / jnp.ones 产出的是弱类型
        # 数组（weak_type=True），而第一次 optimizer update 之后它变成强类型。jit 的
        # 缓存键含 avals（含 weak_type），于是第二个 update 会重编译一次（实测 64 env
        # 下 update 0/1 各 17.3s/16.5s，之后才降到 5s）。用 numpy float32 → jnp.array
        # 得到 weak_type=False 的常量，params（及其 opt_state 的 mu/nu）从第一帧起就
        # 稳定，jit_update 只编译一次。
        log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.array(
                np.full(shape, self.init_log_std, dtype=np.float32)),
            (self.action_dim,))
        std = jnp.exp(log_std)
        return mu, std


class Critic(nn.Module):
    """价值网络（独立参数，与 Actor 完全隔离）"""
    obs_dim: int
    hidden: int = 512

    @nn.compact
    def __call__(self, obs):
        v = nn.Dense(self.hidden)(obs)
        v = nn.relu(v)
        v = nn.Dense(self.hidden)(v)
        v = nn.relu(v)
        return nn.Dense(1)(v).squeeze(-1)   # (batch,)


class ActorCritic(nn.Module):
    """对角高斯 Actor + V Critic（单 Module 双独立 trunk，学生/蒸馏用）"""
    obs_dim: int
    action_dim: int
    hidden: int = 512
    init_log_std: float = -0.5

    @nn.compact
    def __call__(self, obs):
        # Actor trunk
        a = nn.Dense(self.hidden)(obs)
        a = nn.relu(a)
        a = nn.Dense(self.hidden)(a)
        a = nn.relu(a)
        mu = nn.Dense(self.action_dim)(a)

        # 全局 log_std（状态无关，强类型初始化原因见 Actor）
        log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.array(
                np.full(shape, self.init_log_std, dtype=np.float32)),
            (self.action_dim,))
        std = jnp.exp(log_std)

        # Critic trunk（独立）
        v = nn.Dense(self.hidden)(obs)
        v = nn.relu(v)
        v = nn.Dense(self.hidden)(v)
        v = nn.relu(v)
        value = nn.Dense(1)(v).squeeze(-1)  # (batch,)

        return mu, std, value


def make_teacher_actor(action_dim: int = 12) -> Actor:
    """教师 Actor：99 维输入（61 标准 + 38 特权）"""
    return Actor(obs_dim=99, action_dim=action_dim)


def make_teacher_critic() -> Critic:
    """教师 Critic：99 维输入"""
    return Critic(obs_dim=99)


def make_teacher(action_dim: int = 12) -> ActorCritic:
    """教师合一网络（旧接口；smoke 脚本用。正式 PPO 走 make_teacher_actor/critic）"""
    return ActorCritic(obs_dim=99, action_dim=action_dim)


def make_student(action_dim: int = 12) -> ActorCritic:
    """学生网络：61 维输入（实机可获取）"""
    return ActorCritic(obs_dim=61, action_dim=action_dim)
