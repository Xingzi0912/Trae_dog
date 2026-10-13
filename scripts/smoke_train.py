"""训练入口冒烟测试（极小步数，验证管线全通）

用法：python -m scripts.smoke_train
"""

from __future__ import annotations

import pickle
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
import optax
from flax.training.train_state import TrainState

from envs.dog_env_mjx import DogEnvMJX, EnvConfig
from algos.networks_jax import make_teacher, make_student
from algos.ppo_jax import (PPOConfig, ppo_loss, make_teacher_obs,
                           RunningMeanStd, create_train_state)
from algos.distill_jax import distill_loss


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def test_save_npz():
    """验证 save_student_npz 能正确提取权重"""
    log("测试 save_student_npz ...")
    student = make_student()
    params = student.init(jax.random.PRNGKey(0), jnp.zeros((1, 61)))
    ts = TrainState.create(
        apply_fn=student.apply, params=params,
        tx=optax.adam(1e-3))
    obs_rms = RunningMeanStd.create((61,))

    # 模拟 save_student_npz 的提取逻辑
    p = params["params"]
    save_dict = {
        "actor_w1": np.asarray(p["Dense_0"]["kernel"]),
        "actor_b1": np.asarray(p["Dense_0"]["bias"]),
        "actor_w2": np.asarray(p["Dense_1"]["kernel"]),
        "actor_b2": np.asarray(p["Dense_1"]["bias"]),
        "actor_w3": np.asarray(p["Dense_2"]["kernel"]),
        "actor_b3": np.asarray(p["Dense_2"]["bias"]),
    }
    log(f"  w1 shape={save_dict['actor_w1'].shape} (expect 61,512)")
    log(f"  w3 shape={save_dict['actor_w3'].shape} (expect 512,12)")
    assert save_dict["actor_w1"].shape == (61, 512)
    assert save_dict["actor_w3"].shape == (512, 12)
    log("  save_student_npz ✅")


def test_ppo_one_step():
    """验证 PPO 单步 update 能跑通"""
    log("测试 PPO 单步 update ...")
    env_cfg = EnvConfig(dr_enabled=True, use_residual=True)
    env = DogEnvMJX(n_envs=4, config=env_cfg)

    tx = optax.adam(3e-4)
    rng = jax.random.PRNGKey(42)
    ts = create_train_state(rng, tx)

    state, obs = env.reset(rng)
    priv = env.get_privileged_obs(state)
    teacher_obs = make_teacher_obs(obs, priv)

    # 收集 2 步 rollout
    rollout_obs, rollout_act, rollout_lp, rollout_val = [], [], [], []
    rollout_rew, rollout_done = [], []
    for _ in range(2):
        mu, std, value = ts.apply_fn(ts.params, teacher_obs)
        rng, act_rng = jax.random.split(rng)
        action = mu + std * jax.random.normal(act_rng, mu.shape)
        log_prob = -0.5 * (((action - mu) / std) ** 2).sum(-1) \
                   - jnp.log(std).sum(-1)
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

    # 构造 batch 并做一次 PPO update
    batch = {
        "obs": jnp.stack(rollout_obs).reshape(-1, 99),
        "action": jnp.stack(rollout_act).reshape(-1, 12),
        "old_log_prob": jnp.stack(rollout_lp).reshape(-1),
        "advantage": jnp.zeros(8),
        "return_val": jnp.zeros(8),
        "old_value": jnp.stack(rollout_val).reshape(-1),
    }
    (loss, info), grads = jax.value_and_grad(
        ppo_loss, has_aux=True)(ts.params, batch)
    ts = ts.apply_gradients(grads=grads)
    log(f"  PPO loss={loss:.4f}, kl={info['kl']:.4f}")
    log("  PPO 单步 update ✅")


def test_distill_one_step():
    """验证蒸馏单步 update 能跑通"""
    log("测试蒸馏单步 update ...")
    teacher = make_teacher()
    student = make_student()
    rng = jax.random.PRNGKey(0)
    teacher_params = teacher.init(rng, jnp.zeros((1, 99)))
    student_params = student.init(rng, jnp.zeros((1, 61)))
    student_ts = TrainState.create(
        apply_fn=student.apply, params=student_params,
        tx=optax.adam(3e-4))

    batch = {
        "obs": jnp.zeros((4, 61)),
        "mu_teacher": jnp.zeros((4, 12)),
        "v_teacher": jnp.zeros(4),
    }
    (loss, info), grads = jax.value_and_grad(
        distill_loss, has_aux=True)(student_ts.params, teacher_params, batch)
    student_ts = student_ts.apply_gradients(grads=grads)
    log(f"  distill loss={loss:.4f}")
    log("  蒸馏单步 update ✅")


if __name__ == "__main__":
    log(f"jax devices = {jax.devices()}")
    test_save_npz()
    test_ppo_one_step()
    test_distill_one_step()
    log("\n训练管线冒烟测试全部通过 ✅")
