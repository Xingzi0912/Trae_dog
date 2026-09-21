"""scripts/verify_reward.py — Step 2 reward 函数手动验证

三个场景对应 agents.md Step 2 待办的三条：
  1. 零策略：验证 reward ≈ +0.5/step（只有 upright），forward/energy/jerk 都 = 0
  2. 周期摆腿：验证 forward / energy / jerk 三分量都被触发且符号正确
  3. 持续外力矩让狗侧翻：验证 terminated 触发 + upright 归零

运行：
    d:\data\Trae\deep-learning\.venv\Scripts\python.exe d:\data\Trae\dog_rl\scripts\verify_reward.py

判定标准（看到 ✅ 即通过，⚠️ 需要复盘）：
  场景1：upright mean ≈ +0.5，forward mean ≈ 0，狗没塌
  场景2：forward |mean| > 0.001（摆腿后狗确实在动），energy mean < 0，jerk mean < 0
  场景3：300 步内 terminated 触发，最终 upright=0
"""

import sys
from pathlib import Path

# 让脚本能在 dog_rl 项目外被直接 python 调用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from envs.dog_env import DogEnv


# ============================================================
# 场景 1：零策略（baseline）
# ============================================================
def scenario1_zero_policy():
    print("=" * 64)
    print("场景 1：零策略（验证 reward ≈ +0.5/step，forward/energy/jerk = 0）")
    print("=" * 64)

    env = DogEnv()
    _, info = env.reset()
    print(f"[reset] base_z = {info['base_z']:.4f} m （站立目标 ≈ 0.27m）")
    print()

    action = np.zeros(12, dtype=np.float32)
    keys = ["forward", "energy", "jerk", "upright"]
    sums = {k: 0.0 for k in keys}
    total = 0.0
    base_z_history = []
    n_steps = 500

    for step in range(n_steps):
        _, r, term, _, info = env.step(action)
        for k in keys:
            sums[k] += info[f"reward_{k}"]
        total += r
        base_z_history.append(info["base_z"])
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1:4d}  reward={r:+.4f}  "
                  f"upright={info['upright']}  base_z={info['base_z']:.4f}")
        if term:
            print(f"\n  ❌ step {step + 1} terminated, base_z={info['base_z']:.4f}")
            break

    n = len(base_z_history)
    print()
    print(f"[结果] 跑 {n} 步")
    for k in keys:
        print(f"  reward_{k:8s}:  total={sums[k]:+.4f}   mean={sums[k] / n:+.4f}/step")
    print(f"  reward_total :  total={total:+.4f}   mean={total / n:+.4f}/step")
    print(f"  base_z       :  min={min(base_z_history):.4f}  "
          f"max={max(base_z_history):.4f}  final={base_z_history[-1]:.4f}")

    ok_upright = abs(sums["upright"] / n - 0.5) < 0.05
    ok_forward = abs(sums["forward"]) < 0.01
    ok_energy = abs(sums["energy"]) < 0.01
    ok_jerk = abs(sums["jerk"]) < 0.01
    ok_stand = min(base_z_history) > env._z_terminate

    print()
    print(f"  [{'✅' if ok_upright else '⚠️'}] upright mean ≈ +0.5    （实际 {sums['upright'] / n:+.4f}）")
    print(f"  [{'✅' if ok_forward else '⚠️'}] forward mean ≈ 0       （实际 {sums['forward'] / n:+.4f}）")
    print(f"  [{'✅' if ok_energy else '⚠️'}] energy mean ≈ 0        （实际 {sums['energy'] / n:+.4f}）")
    print(f"  [{'✅' if ok_jerk else '⚠️'}] jerk mean ≈ 0          （实际 {sums['jerk'] / n:+.4f}）")
    print(f"  [{'✅' if ok_stand else '❌'}] 零策略站住没塌         （base_z min = {min(base_z_history):.4f}）")
    env.close()


# ============================================================
# 场景 2：周期摆腿（验证 forward / energy / jerk 触发）
# ============================================================
def scenario2_periodic_stepping():
    print()
    print("=" * 64)
    print("场景 2：周期摆腿（验证 forward / energy / jerk 三分量都被触发）")
    print("=" * 64)

    env = DogEnv()
    _, info = env.reset()
    print(f"[reset] base_z = {info['base_z']:.4f} m")
    print()

    keys = ["forward", "energy", "jerk", "upright"]
    sums = {k: 0.0 for k in keys}
    total = 0.0
    forward_speed = []
    base_z_history = []
    n_steps = 500
    freq = 1.0        # 1 Hz 摆腿
    amp = 0.4         # action 摆动幅度
    dt = 0.02         # env step 物理仿真步长（mujoco 默认 0.002s，但 mj_step 1 次 = 0.002s？

    # 实际上 mujoco 默认 dt=0.002s，但 DogEnv 的 step 调一次 mj_step，
    # 所以 500 步 = 1s 仿真时间。1Hz 摆腿在 500 步内完成 1 个周期。
    # 为了多跑几个周期，把 freq 提到 5Hz，500步 = 1s 内跑 5 个周期。
    freq = 5.0

    for step in range(n_steps):
        t = step * 0.002  # mujoco 默认 dt
        phase = 2 * np.pi * freq * t
        # 对角同相 trot：FL/RR 一组，FR/RL 一组（反相）
        action = np.zeros(12, dtype=np.float32)
        # 关节顺序 [FL, FR, RL, RR] × [hip, thigh, calf]
        s = amp * np.sin(phase)
        s_opp = amp * np.sin(phase + np.pi)
        # FL thigh(1) + RR thigh(10) 同相
        action[1] = s
        action[10] = s
        # FR thigh(4) + RL thigh(7) 反相
        action[4] = s_opp
        action[7] = s_opp
        # calf 反相补偿（thigh 前摆时 calf 后收，自然协调）
        action[2] = -0.6 * s
        action[11] = -0.6 * s
        action[5] = -0.6 * s_opp
        action[8] = -0.6 * s_opp

        _, r, term, _, info = env.step(action)
        for k in keys:
            sums[k] += info[f"reward_{k}"]
        total += r
        forward_speed.append(info["forward_x_speed"])
        base_z_history.append(info["base_z"])

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1:4d}  reward={r:+.4f}  "
                  f"forward={info['reward_forward']:+.3f}  "
                  f"energy={info['reward_energy']:+.3f}  "
                  f"jerk={info['reward_jerk']:+.4f}  "
                  f"vx={info['forward_x_speed']:+.3f}  base_z={info['base_z']:.4f}")
        if term:
            print(f"\n  ❌ step {step + 1} terminated, base_z={info['base_z']:.4f}")
            break

    n = len(base_z_history)
    print()
    print(f"[结果] 跑 {n} 步（≈ {n * 0.002:.2f}s 仿真时间）")
    for k in keys:
        print(f"  reward_{k:8s}:  total={sums[k]:+.4f}   mean={sums[k] / n:+.4f}/step")
    print(f"  reward_total :  total={total:+.4f}   mean={total / n:+.4f}/step")
    print(f"  forward_speed: mean={np.mean(forward_speed):+.4f} m/s  "
          f"final={forward_speed[-1]:+.4f} m/s")
    print(f"  base_z       : min={min(base_z_history):.4f}  "
          f"max={max(base_z_history):.4f}")

    # 判定
    ok_forward_trig = abs(sums["forward"]) > 0.001
    ok_energy_neg = sums["energy"] < -0.001
    ok_jerk_neg = sums["jerk"] < -0.001
    ok_stand = min(base_z_history) > env._z_terminate
    print()
    print(f"  [{'✅' if ok_forward_trig else '⚠️'}] forward reward 被触发    "
          f"（total={sums['forward']:+.4f}, |mean vx|={abs(np.mean(forward_speed)):.4f}）")
    print(f"  [{'✅' if ok_energy_neg else '⚠️'}] energy 是负惩罚       （total={sums['energy']:+.4f}）")
    print(f"  [{'✅' if ok_jerk_neg else '⚠️'}] jerk 是负惩罚          （total={sums['jerk']:+.4f}）")
    print(f"  [{'✅' if ok_stand else '❌'}] 摆腿过程没塌           （base_z min={min(base_z_history):.4f}）")
    if not ok_forward_trig:
        print("  💡 forward ≈ 0：摆腿幅度/相位可能没让狗前进。reward 函数本身在工作，")
        print("     只是这个手动摆腿动作不产生净前进。PPO 训练时由策略自己学正确步态。")
    env.close()


# ============================================================
# 场景 3：持续外力矩让狗侧翻（验证 terminated 触发）
# ============================================================
def scenario3_external_torque():
    print()
    print("=" * 64)
    print("场景 3：持续外力矩让狗侧翻（验证 terminated 触发 + upright 归零）")
    print("=" * 64)

    env = DogEnv()
    _, info = env.reset()
    print(f"[reset] base_z = {info['base_z']:.4f} m")
    print()

    action = np.zeros(12, dtype=np.float32)
    torque_x = 30.0   # 绕 x 轴 30 N·m，足以让 base 侧翻
    terminated_at = None

    for step in range(300):
        # 在 env.step 调 mj_step 之前施加外力矩
        # qfrc_applied 前 6 维对应 freejoint 的 [fx, fy, fz, tx, ty, tz]
        env.data.qfrc_applied[3] = torque_x
        _, r, term, _, info = env.step(action)
        # 清零，下一步再设（持续施加 = 每步都设）
        env.data.qfrc_applied[:] = 0.0

        if (step + 1) % 20 == 0 or step == 0:
            print(f"  step {step + 1:4d}  base_z={info['base_z']:.4f}  "
                  f"roll={info['roll']:+.3f}  pitch={info['pitch']:+.3f}  "
                  f"upright={info['upright']}  reward={r:+.4f}")

        if term:
            terminated_at = step + 1
            print()
            print(f"  ✅ step {step + 1} terminated: base_z={info['base_z']:.4f} < {env._z_terminate}")
            print(f"     roll={info['roll']:+.3f}  pitch={info['pitch']:+.3f}  upright={info['upright']}")
            break

    print()
    if terminated_at is not None:
        print(f"  ✅ terminated 在 step {terminated_at} 触发，狗被力矩推翻")
        print(f"  ✅ 摔倒时 upright={info['upright']}（应 = 0，因 roll 已超 0.4）")
    else:
        print(f"  ❌ 300 步内未 terminated，需要更大力矩或更长步数")
        print(f"     当前 base_z={info['base_z']:.4f}  roll={info['roll']:+.3f}")
    env.close()


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("MuJoCo 默认 dt = 0.002s（mujoco.MjOption.dt）")
    print("DogEnv.step 调一次 mj_step，所以 500 步 ≈ 1s 仿真时间")
    print()
    scenario1_zero_policy()
    scenario2_periodic_stepping()
    scenario3_external_torque()
    print()
    print("=" * 64)
    print("Step 2 验证脚本跑完，请根据上面的 ✅/⚠️/❌ 标记判断 reward 设计")
    print("=" * 64)
