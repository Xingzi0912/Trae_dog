"""基准步态 + 残差模式验证

验证：
1. gait 函数输出形状 (N, 12)，零指令时小幅摆动
2. 残差模式下零策略（action=0）= 纯基准步态，狗应该在走
3. 对比旧模式（action=0=站立）和残差模式的关节运动

用法：python -m scripts.test_gait
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 直接 `python3 scripts/xxx.py` 时 sys.path[0] 是 scripts/，仓库根不在路径里，
# 需手动补入才能 import envs/algos（`python -m scripts.xxx` 则自带）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from envs.dog_env_mjx import DogEnvMJX, EnvConfig
from envs.gait import gait, Q_DEFAULT


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def test_gait_fn():
    """测试 gait 函数"""
    log("=== 测试 gait 函数 ===")
    # 零指令
    cmd_zero = jnp.zeros((1, 3))
    g0 = gait(jnp.array([0]), cmd_zero)
    log(f"  零指令 step 0: gait={np.asarray(g0[0])}")
    log(f"  零指令 step 0: max swing={np.abs(np.asarray(g0[0])).max():.4f} rad")

    # 有 vx 指令
    cmd_vx = jnp.array([[1.0, 0.0, 0.0]])
    g1 = gait(jnp.array([0]), cmd_vx)
    log(f"  vx=1.0 step 0: gait={np.asarray(g1[0])}")
    log(f"  vx=1.0 step 0: max swing={np.abs(np.asarray(g1[0])).max():.4f} rad")

    # 不同时间步
    phases = jnp.arange(0, 50, 10)
    g_seq = gait(phases, jnp.tile(cmd_vx, (5, 1)))
    log(f"  不同步 thigh 摆动: FL_thigh={np.asarray(g_seq[:, 1])}")
    assert g0.shape == (1, 12)
    assert g_seq.shape == (5, 12)
    log("  gait 函数 ✅")


def test_residual_env():
    """测试残差模式下零策略行走"""
    log("\n=== 测试残差模式 env（零策略=基准步态行走）===")
    cfg = EnvConfig(
        dr_enabled=False,
        cmd_vx=(0.5, 1.0),  # 给一个前进指令
        cmd_vy=(0.0, 0.0),
        cmd_yaw=(0.0, 0.0),
        zero_cmd_prob=0.0,
        use_residual=True,
    )
    env = DogEnvMJX(n_envs=1, config=cfg)
    state, obs = env.reset(jax.random.PRNGKey(0))
    actions = jnp.zeros((1, 12))  # 零残差

    q_list = []
    z_list = []
    for i in range(50):
        state, obs, rew, term, trunc, info = env.step(state, actions)
        q = np.asarray(obs[0, 10:22])
        z = float(np.asarray(obs[0, 0]))
        q_list.append(q)
        z_list.append(z)
        if (i + 1) % 10 == 0:
            log(f"  step {i+1}: base_z={z:.4f}, "
                f"FL_thigh={q[1]:.3f}, FL_calf={q[2]:.3f}")

    q_arr = np.array(q_list)
    thigh_range = q_arr[:, 1].max() - q_arr[:, 1].min()
    log(f"  50 步 FL_thigh 摆动范围: {np.degrees(thigh_range):.1f}°")
    log(f"  base_z 范围: [{min(z_list):.4f}, {max(z_list):.4f}]")
    assert thigh_range > 0.05, "大腿应该在摆动"
    assert min(z_list) > 0.2, "不应该摔倒"
    log("  残差模式行走 ✅")


if __name__ == "__main__":
    test_gait_fn()
    test_residual_env()
    log("\n全部步态测试通过 ✅")
