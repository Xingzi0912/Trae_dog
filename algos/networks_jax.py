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
import flax.linen as nn


class ActorCritic(nn.Module):
    """对角高斯 Actor + V Critic，共享结构（教师/学生通用）"""
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

        # 全局 log_std（状态无关）
        log_std = self.param(
            "log_std", lambda rng, shape: jnp.full(shape, self.init_log_std),
            (self.action_dim,))
        std = jnp.exp(log_std)

        # Critic trunk（独立）
        v = nn.Dense(self.hidden)(obs)
        v = nn.relu(v)
        v = nn.Dense(self.hidden)(v)
        v = nn.relu(v)
        value = nn.Dense(1)(v).squeeze(-1)  # (batch,)

        return mu, std, value


def make_teacher(action_dim: int = 12) -> ActorCritic:
    """教师网络：99 维输入（61 标准 + 38 特权）"""
    return ActorCritic(obs_dim=99, action_dim=action_dim)


def make_student(action_dim: int = 12) -> ActorCritic:
    """学生网络：61 维输入（实机可获取）"""
    return ActorCritic(obs_dim=61, action_dim=action_dim)
