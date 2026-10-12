"""JAX 师生架构冒烟测试（阶段 3 验证）

验证：
1. env 能创建、reset、step，privileged_obs 维度正确 (N, 38)
2. 教师网络 (99→12+1) 前向传播
3. 学生网络 (61→12+1) 前向传播
4. PPO 损失能计算、梯度能反传
5. 蒸馏损失能计算

用法：python -m scripts.smoke_jax
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from envs.dog_env_mjx import DogEnvMJX, EnvConfig
from algos.networks_jax import make_teacher, make_student
from algos.ppo_jax import (PPOConfig, ppo_loss, make_teacher_obs,
                           RunningMeanStd)
from algos.distill_jax import distill_loss


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"jax devices = {jax.devices()}")
    n_envs = 4

    # 1. env
    log(f"1. 创建 env (n_envs={n_envs}, dr=False) ...")
    cfg = EnvConfig(dr_enabled=False)
    env = DogEnvMJX(n_envs=n_envs, config=cfg)
    state, obs = env.reset(jax.random.PRNGKey(0))
    priv = env.get_privileged_obs(state)
    log(f"   obs shape={obs.shape} (expect {n_envs},61)")
    log(f"   priv shape={priv.shape} (expect {n_envs},38)")
    assert obs.shape == (n_envs, 61), f"obs shape wrong: {obs.shape}"
    assert priv.shape == (n_envs, 38), f"priv shape wrong: {priv.shape}"

    # 2. teacher forward
    log("2. 教师网络前向传播 ...")
    teacher = make_teacher()
    teacher_params = teacher.init(jax.random.PRNGKey(1), jnp.zeros((1, 99)))
    teacher_obs = make_teacher_obs(obs, priv)
    mu, std, value = teacher.apply(teacher_params, teacher_obs)
    log(f"   mu shape={mu.shape}, std shape={std.shape}, value shape={value.shape}")
    assert mu.shape == (n_envs, 12)
    assert value.shape == (n_envs,)

    # 3. student forward
    log("3. 学生网络前向传播 ...")
    student = make_student()
    student_params = student.init(jax.random.PRNGKey(2), jnp.zeros((1, 61)))
    mu_s, std_s, value_s = student.apply(student_params, obs)
    log(f"   mu shape={mu_s.shape}, value shape={value_s.shape}")
    assert mu_s.shape == (n_envs, 12)

    # 4. PPO loss + grad
    log("4. PPO 损失 + 梯度 ...")
    batch = {
        "obs": teacher_obs,
        "action": jnp.zeros((n_envs, 12)),
        "old_log_prob": jnp.zeros(n_envs),
        "advantage": jnp.zeros(n_envs),
        "return_val": jnp.zeros(n_envs),
        "old_value": jnp.zeros(n_envs),
    }
    (loss, info), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
        teacher_params, batch)
    log(f"   loss={loss:.4f}, kl={info['kl']:.4f}, entropy={info['entropy']:.4f}")

    # 5. distill loss
    log("5. 蒸馏损失 ...")
    distill_batch = {
        "obs": obs,
        "mu_teacher": mu,
        "v_teacher": value,
    }
    (d_loss, d_info), _ = jax.value_and_grad(distill_loss, has_aux=True)(
        student_params, teacher_params, distill_batch)
    log(f"   distill loss={d_loss:.4f}, action_loss={d_info['action_loss']:.4f}, "
        f"value_loss={d_info['value_loss']:.4f}")

    # 6. 跑 1 步 env 验证 step 返回 priv
    log("6. env.step 验证 info['privileged_obs'] ...")
    action = jnp.zeros((n_envs, 12))
    state, obs, rew, term, trunc, info = env.step(state, action)
    log(f"   reward shape={rew.shape}, term shape={term.shape}")
    log(f"   info['privileged_obs'] shape={info['privileged_obs'].shape}")
    assert info["privileged_obs"].shape == (n_envs, 38)

    log("\n全部冒烟测试通过 ✅")


if __name__ == "__main__":
    main()
