"""FK 数学验证：用 MuJoCo 真值对照 deploy.kin_estimator

不依赖任何实机硬件，在本机直接跑：
    python scripts/verify_fk.py

验证 3 项：
    1. 腿 FK 脚位置  vs MuJoCo foot geom 世界位置反投影  → 期望误差 < 1mm
    2. 估计 base_z   vs data.xpos(base,2)                 → 期望 < 5mm
    3. 估计机身系 v  vs R^T @ qvel[0:3]                    → 期望静态 <0.05、动态 <0.15 m/s

跑两段：零策略站立 200 步（静态）+ 小幅随机动作 300 步（动态，验雅可比）。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.dog_env import DogEnv, RandConfig
from deploy.kin_estimator import KinEstimator, leg_fk, _FOOT_D
import mujoco


def get_foot_positions_b(model, data):
    """MuJoCo 真值：脚球心相对 base 的位置（base frame）"""
    R = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")] \
        .reshape(3, 3)
    base_pos = data.xpos[model.body("base_link").id]
    feet = {}
    for leg, name in enumerate(("FL_calf", "FR_calf", "RL_calf", "RR_calf")):
        bid = model.body(name).id
        foot_world = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ _FOOT_D
        feet[leg] = R.T @ (foot_world - base_pos)
    return feet, R


def get_contact_mask(model, data):
    """从 MuJoCo 接触点读出哪只脚着地（foot geom 接触 floor）"""
    calf = {
        model.body(n).id: leg
        for leg, n in enumerate(("FL_calf", "FR_calf", "RL_calf", "RR_calf"))
    }
    mask = np.zeros(4, dtype=bool)
    for c in data.contact:
        for gid in (c.geom1, c.geom2):
            bid = model.geom_bodyid[gid]
            if bid in calf:
                mask[calf[bid]] = True
    return mask


def run_segment(env, estimator, n_steps, actions, label):
    model, data = env.model, env.data
    foot_err_all, z_err_all, v_err_all = [], [], []
    spike_steps = []

    for i in range(n_steps):
        obs, r, term, trunc, info = env.step(actions[i])
        if term:
            print(f"[{label}] 第 {i} 步狗塌倒，后续动态段数据仅用此前部分")
            break

        quat = data.qpos[3:7]
        rpy = env._quat_to_rpy(quat)
        q = data.qpos[7:]
        qd = data.qvel[6:]
        R = data.xmat[model.body("base_link").id].reshape(3, 3)
        gyro_b = R.T @ data.qvel[3:6]                  # 机身系角速度真值
        contact = get_contact_mask(model, data)

        # 1) 逐脚 FK 位置误差（所有脚都对照，不仅是支撑脚）
        feet_true, _ = get_foot_positions_b(model, data)
        for leg in range(4):
            p_fk, _, _ = leg_fk(q[3 * leg:3 * leg + 3], leg)
            foot_err_all.append(np.linalg.norm(p_fk - feet_true[leg]))

        # 2)+3) 估计器输出 vs 真值
        z_est, v_est = estimator.update(q, qd, rpy, gyro_b, contact)
        z_true = float(data.xpos[model.body("base_link").id, 2])
        v_true = R.T @ data.qvel[0:3]
        if contact.any():
            z_e = abs(z_est - z_true)
            v_e = np.abs(v_est - v_true)
            z_err_all.append(z_e)
            v_err_all.append(v_e)
            if z_e > 0.005 or v_e.max() > 0.05:
                spike_steps.append(i)

    def report(name, arr, unit, thr):
        if len(arr) == 0:
            print(f"  {name}: 无数据")
            return
        a = np.asarray(arr)
        print(f"  {name}: max={a.max():.4f}{unit} mean={a.mean():.4f}{unit} "
              f"{'✅' if a.max() < thr else '❌ 超阈值 ' + str(thr)}")

    print(f"\n── {label} ──")
    report("脚位置误差", foot_err_all, "m", 0.002)
    report("高度误差", z_err_all, "m", 0.005)
    v_flat = np.concatenate(v_err_all) if v_err_all else []
    report("线速度误差(逐轴)", v_flat, "m/s",
           0.05 if label.startswith("静态") else 0.15)
    print(f"  超阈值尖峰出现步号: {spike_steps[:12]}"
          f"{'（共 %d 步）' % len(spike_steps) if len(spike_steps) > 12 else ''}")


def main():
    env = DogEnv(rand_config=RandConfig(enabled=False))
    env.reset(seed=0)
    # 模拟"标定后"：z_offset=脚球半径 0.022（球心离地，实机标定时该常数含进 offset）
    estimator = KinEstimator(lpf_alpha=1.0, z_offset=0.022)  # 关低通看裸误差

    # 静态段：零策略
    zero = np.zeros(12)
    run_segment(env, estimator, 200, [zero] * 200, "静态段（零策略站立 200 步 = 4s）")

    # 动态段：平滑小随机动作（插值生成，避免冲击塌倒）
    rng = np.random.default_rng(1)
    keyframes = rng.uniform(-0.25, 0.25, size=(15, 12))
    actions = []
    for k in range(14):
        for f in range(20):
            t = f / 20
            actions.append((1 - t) * keyframes[k] + t * keyframes[k + 1])
    actions += [np.zeros(12)] * 20
    run_segment(env, estimator, 300, actions[:300],
                "动态段（平滑随机动作 300 步 = 6s）")

    env.close()


if __name__ == "__main__":
    main()
