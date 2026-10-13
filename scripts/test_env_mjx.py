"""DogEnvMJX 环境测试脚本（阶段 2 验证）

四项检查：
1. 一致性：同 XML 下 CPU 循环 vs MJX 环境，零策略 100 步逐对比观测（对齐阶段1阈值）
2. 语义点检：站立时 g_b=[0,0,-1]、obs[7:10]==CPU qvel[3:6]、1000 步零策略不塌
3. 指令重采样：4 env 跑 300 步，每个 env 的 command 至少变一次
4. DR 冒烟：4 env 开 DR，reset 后 env 间有差异、10 步无 NaN

用法（项目根目录）：python scripts/test_env_mjx.py
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
import mujoco
import numpy as np
from mujoco import mjx

from envs.dog_env_mjx import DogEnvMJX, EnvConfig


def log(msg):
    """带时间戳的打印，立即刷新"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

ROOT = Path(__file__).resolve().parent.parent
ENVS = ROOT / "envs"
SCENE = str(ENVS / "scene_mjx.xml")
Q_DEFAULT = np.array([0.0, 0.9, -1.8] * 4)


# ---------------------------------------------------------------------------
# CPU 侧：与 MJX 完全对齐的观测构造
# ---------------------------------------------------------------------------
def _quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def make_obs_cpu(qpos, qvel, last_action, command):
    """CPU 侧构造与 dog_env_mjx._get_obs 完全一致的 61 维观测"""
    R = _quat_to_mat(qpos[3:7])
    g_b = -R[2, :]                 # projected gravity
    lin_b = R.T @ qvel[0:3]        # 机身系线速度
    ang_b = qvel[3:6]              # 机身系角速度（freejoint 本就机身系）
    return np.concatenate([
        qpos[2:3],                 #  0   base_z
        g_b,                       #  1:4
        lin_b,                     #  4:7
        ang_b,                     #  7:10
        qpos[7:],                  # 10:22
        qvel[6:],                  # 22:34
        last_action,               # 34:46
        Q_DEFAULT,                 # 46:58
        command,                   # 58:61
    ])


def rollout_cpu_zero(n_steps):
    """CPU MuJoCo 零策略 rollout（ctrl=q_default 站立），返回 (n_steps+1, 61) 观测"""
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    obs_list = [make_obs_cpu(data.qpos, data.qvel, np.zeros(12), np.zeros(3))]
    ctrl = Q_DEFAULT.copy()
    for _ in range(n_steps):
        data.ctrl[:] = ctrl
        for _ in range(10):          # frame_skip=10
            mujoco.mj_step(model, data)
        obs_list.append(make_obs_cpu(data.qpos, data.qvel,
                                     np.zeros(12), np.zeros(3)))
    return np.array(obs_list)


def rollout_mjx_zero(n_steps):
    """MJX 环境零策略 rollout（DR 关闭、指令恒 0），返回 (n_steps+1, 61) 观测"""
    cfg = EnvConfig(
        dr_enabled=False,
        cmd_vx=(0.0, 0.0),
        cmd_vy=(0.0, 0.0),
        cmd_yaw=(0.0, 0.0),
        zero_cmd_prob=1.0,
        use_residual=False,  # 一致性测试用旧模式（纯 q_default）
    )
    log("  构建 DogEnvMJX(n_envs=1) ...")
    env = DogEnvMJX(n_envs=1, config=cfg)
    log("  第一次 env.reset()（含 JIT 编译 _reset_batch）...")
    t0 = time.time()
    state, obs = env.reset(jax.random.PRNGKey(0))
    log(f"  reset 完成，耗时 {time.time()-t0:.1f}s")
    obs_list = [np.asarray(obs[0])]
    actions = jnp.zeros((1, 12))
    log(f"  第一次 env.step()（含 JIT 编译 _step_batch）...")
    t0 = time.time()
    state, obs, rew, term, trunc, info = env.step(state, actions)
    log(f"  step 编译完成，耗时 {time.time()-t0:.1f}s")
    obs_list.append(np.asarray(obs[0]))
    for i in range(1, n_steps):
        state, obs, rew, term, trunc, info = env.step(state, actions)
        obs_list.append(np.asarray(obs[0]))
        if (i + 1) % 25 == 0:
            log(f"  step 进度 {i+1}/{n_steps}")
    return np.array(obs_list)


# ---------------------------------------------------------------------------
# 检查项 1：一致性
# ---------------------------------------------------------------------------
def test_consistency():
    print("\n=== 检查项 1：CPU vs MJX 零策略观测一致性（100 策略步 = 2s）===")
    N = 100
    log("  CPU rollout 100 步 ...")
    obs_cpu = rollout_cpu_zero(N)
    log("  CPU rollout 完成")
    log("  MJX rollout 100 步 ...")
    obs_mjx = rollout_mjx_zero(N)
    log("  MJX rollout 完成")
    d = np.abs(obs_cpu - obs_mjx)

    print(f"  base_z     : max={d[:, 0].max():.6f} m")
    print(f"  g_b (1:4)  : max={d[:, 1:4].max():.6f}")
    print(f"  lin_b(4:7) : max={d[:, 4:7].max():.6f}")
    print(f"  ang_b(7:10): max={d[:, 7:10].max():.6f}")
    print(f"  q   (10:22): max={np.degrees(d[:, 10:22].max()):.4f}°")
    print(f"  dq  (22:34): max={d[:, 22:34].max():.6f} rad/s")
    print(f"  last_act   : max={d[:, 34:46].max():.6f}")
    print(f"  q_default  : max={d[:, 46:58].max():.6f}")
    print(f"  cmd        : max={d[:, 58:61].max():.6f}")

    jmax = np.degrees(d[:, 10:22].max())
    zmax = d[:, 0].max()
    ok = jmax < 0.5 and zmax < 0.01
    print(f"  判定：关节 max={jmax:.4f}°(<0.5°), base_z max={zmax:.6f}m(<1cm) "
          f"-> {'✅' if ok else '⚠️ 超阈值，需分析'}")
    return ok


# ---------------------------------------------------------------------------
# 检查项 2：语义点检
# ---------------------------------------------------------------------------
def test_semantics():
    print("\n=== 检查项 2：语义点检 ===")
    cfg = EnvConfig(
        dr_enabled=False,
        cmd_vx=(0.0, 0.0),
        cmd_vy=(0.0, 0.0),
        cmd_yaw=(0.0, 0.0),
        zero_cmd_prob=1.0,
        use_residual=False,  # 语义点检用旧模式（零策略=站立）
    )
    env = DogEnvMJX(n_envs=1, config=cfg)
    state, obs = env.reset(jax.random.PRNGKey(0))
    obs0 = np.asarray(obs[0])

    # 2a: 站立时 g_b ≈ [0, 0, -1]
    g_b = obs0[1:4]
    err_g = np.abs(g_b - np.array([0.0, 0.0, -1.0])).max()
    print(f"  2a. reset 后 g_b = {g_b}, 距 [0,0,-1] 最大误差 = {err_g:.6f}")
    ok_g = err_g < 0.05
    print(f"      -> {'✅' if ok_g else '⚠️'}")

    # 2b: obs[7:10] == CPU qvel[3:6]（freejoint 角速度本就机身系，验证未错乘 R.T）
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    cpu_ang = data.qvel[3:6].copy()
    mjx_ang = obs0[7:10]
    err_ang = np.abs(cpu_ang - mjx_ang).max()
    print(f"  2b. reset 后 obs[7:10]={mjx_ang}, CPU qvel[3:6]={cpu_ang}, 差={err_ang:.6f}")
    ok_ang = err_ang < 1e-5
    print(f"      -> {'✅' if ok_ang else '⚠️'}")

    # 2c: 1000 步零策略不塌
    actions = jnp.zeros((1, 12))
    z_min = float(obs0[0])
    fell = False
    log("  2c. 跑 1000 步零策略（复用已编译的 step，应很快）...")
    for i in range(1000):
        state, obs, rew, term, trunc, info = env.step(state, actions)
        z = float(np.asarray(obs[0, 0]))
        z_min = min(z_min, z)
        if bool(np.asarray(term[0])):
            fell = True
            break
        if (i + 1) % 200 == 0:
            log(f"  2c. 进度 {i+1}/1000, 当前 base_z={z:.4f}")
    print(f"  2c. 1000 步零策略 base_z min = {z_min:.4f} m, 摔倒={'是' if fell else '否'}")
    ok_fall = (not fell) and z_min > 0.25
    print(f"      -> {'✅' if ok_fall else '⚠️'}")

    return ok_g and ok_ang and ok_fall


# ---------------------------------------------------------------------------
# 检查项 3：指令重采样
# ---------------------------------------------------------------------------
def test_command_resample():
    print("\n=== 检查项 3：指令重采样（4 env × 300 步）===")
    cfg = EnvConfig(dr_enabled=False)          # 不开 DR，指令仍按课程重采样
    log("  构建 DogEnvMJX(n_envs=4, dr=False) ...")
    env = DogEnvMJX(n_envs=4, config=cfg)
    log("  reset（n_envs=4 新 shape，需重新编译）...")
    t0 = time.time()
    state, obs = env.reset(jax.random.PRNGKey(0))
    log(f"  reset 完成，耗时 {time.time()-t0:.1f}s")

    cmd_history = [np.asarray(state.command)]  # (4, 3)
    actions = jnp.zeros((4, 12))
    log("  第一次 step（dr=False 新控制流，需编译）...")
    t0 = time.time()
    state, obs, rew, term, trunc, info = env.step(state, actions)
    log(f"  step 编译完成，耗时 {time.time()-t0:.1f}s")
    cmd_history.append(np.asarray(state.command))
    for i in range(1, 300):
        state, obs, rew, term, trunc, info = env.step(state, actions)
        cmd_history.append(np.asarray(state.command))
        if (i + 1) % 100 == 0:
            log(f"  进度 {i+1}/300")
    cmd_history = np.array(cmd_history)        # (301, 4, 3)

    all_ok = True
    for i in range(4):
        # 与第一步 command 比较，看是否变化过
        diff = np.abs(cmd_history[1:, i] - cmd_history[0, i]).sum(axis=1)
        n_changes = int((diff > 1e-6).sum())
        print(f"  env {i}: command 变化步数 = {n_changes}/300")
        if n_changes == 0:
            all_ok = False
            print(f"    ⚠️ env {i} 指令从未变化！")

    print(f"  判定：所有 env 指令均变化过 -> {'✅' if all_ok else '❌'}")
    return all_ok


# ---------------------------------------------------------------------------
# 检查项 4：DR 冒烟
# ---------------------------------------------------------------------------
def test_dr_smoke():
    print("\n=== 检查项 4：DR 冒烟（4 env，开 DR）===")
    log("  构建 DogEnvMJX(n_envs=4, dr=True) ...")
    env = DogEnvMJX(n_envs=4, config=EnvConfig(dr_enabled=True))
    log("  reset（dr=True 新控制流，需编译）...")
    t0 = time.time()
    state, obs = env.reset(jax.random.PRNGKey(0))
    log(f"  reset 完成，耗时 {time.time()-t0:.1f}s")

    # 4a: 模型级 DR 现为「整批共享标量」（2026-10-13 性能重构：模型按 env 批化
    #     会让 mjx.step 慢 31 倍）。env 间差异只来自初态扰动/动作延迟/指令
    dr = state.dr
    fric_std = float(np.asarray(dr.fric).std())
    mass_std = float(np.asarray(dr.mass_scale).std())
    kp_std = float(np.asarray(dr.kp_scale).std())
    print(f"  4a. 模型级 DR 整批共享（std 应为 0）: fric={fric_std:.4f}, "
          f"mass={mass_std:.4f}, kp={kp_std:.4f}")
    print(f"      本 rollout 模型级 DR 标量: "
          f"fric={float(np.asarray(dr.fric)[0]):.3f}, "
          f"mass={float(np.asarray(dr.mass_scale)[0]):.3f}, "
          f"kp={float(np.asarray(dr.kp_scale)[0]):.3f}, "
          f"kv={float(np.asarray(dr.kv_scale)[0]):.3f}")
    obs_np = np.asarray(obs)
    obs_std = float(obs_np.std(axis=0).max())
    print(f"      obs env 间最大 std = {obs_std:.6f}")
    ok_diff = (fric_std < 1e-6 and mass_std < 1e-6 and kp_std < 1e-6
               and obs_std > 1e-6)

    # 4b: 跑 10 步无 NaN
    actions = jnp.zeros((4, 12))
    has_nan = False
    log("  第一次 step（dr=True，含 build_model 外层 DR，需编译）...")
    t0 = time.time()
    state, obs, rew, term, trunc, info = env.step(state, actions)
    log(f"  step 编译完成，耗时 {time.time()-t0:.1f}s")
    for step in range(1, 10):
        state, obs, rew, term, trunc, info = env.step(state, actions)
        obs_np = np.asarray(obs)
        rew_np = np.asarray(rew)
        if np.isnan(obs_np).any() or np.isnan(rew_np).any():
            has_nan = True
            print(f"  4b. ⚠️ step {step} 出现 NaN！")
            break

    print(f"  4b. 10 步无 NaN -> {'✅' if not has_nan else '❌'}")
    ok = ok_diff and not has_nan
    print(f"  判定：env 间有差异且无 NaN -> {'✅' if ok else '❌'}")
    return ok


# ---------------------------------------------------------------------------
def main():
    log(f"jax devices = {jax.devices()}")
    log(f"scene = {SCENE}")

    r1 = test_consistency()
    r2 = test_semantics()
    r3 = test_command_resample()
    r4 = test_dr_smoke()

    print("\n" + "=" * 50)
    print("汇总：")
    print(f"  1. 一致性       : {'✅' if r1 else '❌'}")
    print(f"  2. 语义点检     : {'✅' if r2 else '❌'}")
    print(f"  3. 指令重采样   : {'✅' if r3 else '❌'}")
    print(f"  4. DR 冒烟      : {'✅' if r4 else '❌'}")
    all_ok = all([r1, r2, r3, r4])
    print(f"\n  全部通过 -> {'✅' if all_ok else '❌'}")


if __name__ == "__main__":
    main()
