"""Actor-Critic 网络（Step 3，从 14_ppo_mujoco.py 拆出）

与 14 号脚本的唯一结构差异：
    hidden 256 → 512（agents.md：58 维观测 + 机器狗动态更复杂）
    state_dim/action_dim 由外部传入（DogEnv = 58 / 12）。

配方沿用 14 号已验证结论：
    - Actor / Critic 独立双网（不共享 trunk，PPO 论文/CleanRL/rsl_rl 默认）
    - 对角高斯策略，log_std 为状态无关的全局 nn.Parameter（初始 -0.5 → σ≈0.61）
    - μ 头无激活（输出全体实数，环境端 clip）
    - V 头线性无界，squeeze(-1) 与 advantage 形状对齐
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """Gaussian 策略：state → 512 → ReLU → 512 → ReLU → μ(action_dim)，σ 全局可学习"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, x) -> Normal:
        """返回对角高斯 N(μ(s), diag(σ²))，σ 状态无关"""
        mu = self.mu_head(self.trunk(x))
        std = self.log_std.exp().expand_as(mu)  # (action_dim,) → (batch, action_dim)
        return Normal(mu, std)


class ValueNetwork(nn.Module):
    """状态价值：state → 512 → ReLU → 512 → ReLU → 1（线性输出，无界）"""

    def __init__(self, state_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,1) → (batch,)
