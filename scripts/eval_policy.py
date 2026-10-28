#!/usr/bin/env python3
"""eval_policy.py —— 策略的确定性评估（不渲染，只出数）

和 render_jax.py 的区别：那个是为了出视频，只打印总位移；这个按**分段指令**
跑，逐段给出「机身系跟踪」和「世界系直线度」两套指标，用来判断能不能走直线。

支持两种 checkpoint：
  --ckpt checkpoints/dog_cmd_mjx_teacher.pkl   教师（99 维，Flax params）
  --npz  checkpoints/dog_cmd_mjx_student.npz   学生部署包（61 维，numpy 前向）
                                                ——这条路径就是部署端要用的推理逻辑

指标说明：
  vx/vy/yaw  机身系实测均值（= 奖励里用的量，衡量"跟得准不准"）
  dx/dy      世界系位移
  dy/dx      直线度（纯前进段应 < 0.05）
  heading    航向漂移角 = atan2(dy, dx)，单位度（纯前进段应 < 3°）

用法：
    python scripts/eval_policy.py --ckpt checkpoints/dog_cmd_mjx_teacher.pkl
    python scripts/eval_policy.py --npz  checkpoints/dog_cmd_mjx_student.npz
    python scripts/eval_policy.py --npz ... --cmds "0:0.3,0,0;300:0,0,0"
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
import mujoco                   # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from envs.dog_env_mjx import DogEnvMJX, EnvConfig  # noqa: E402
from algos.networks_jax import make_teacher_actor, make_student  # noqa: E402

DT = 0.02

# 默认评估序列：(名称, 指令, 步数)
DEFAULT_PLAN = [
    ("stand",   (0.0, 0.0, 0.0), 100),
    ("fwd_0.2", (0.2, 0.0, 0.0), 300),
    ("fwd_0.3", (0.3, 0.0, 0.0), 300),
    ("fwd_0.5", (0.5, 0.0, 0.0), 300),
    ("stop",    (0.0, 0.0, 0.0), 150),
    ("yaw_0.5", (0.0, 0.0, 0.5), 200),
    ("vy_0.2",  (0.0, 0.2, 0.0), 200),
    ("bwd_0.3", (-0.3, 0.0, 0.0), 200),
]


# ------------------------------------------------------------------ 策略
class TeacherPolicy:
    """99 维：61 标准 obs + 38 特权，Flax params"""

    def __init__(self, ckpt_path: str):
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        self.params = ck["params"]
        rms = ck.get("obs_rms")
        if rms is None:
            raise SystemExit("教师 checkpoint 缺 obs_rms")
        self.mean = jnp.asarray(rms.mean)
        self.var = jnp.asarray(rms.var)
        import jax
        actor = make_teacher_actor()
        self._apply = jax.jit(lambda p, o: actor.apply(p["actor"], o))
        self.dim = 99

    def action(self, obs61, priv38):
        o = jnp.concatenate([obs61, priv38], axis=-1)
        o = (o - self.mean) / jnp.sqrt(self.var + 1e-8)
        mu, _ = self._apply(self.params, o)
        return mu


class StudentNpzPolicy:
    """61 维：部署包 npz，纯 numpy 前向（= 部署端推理逻辑）"""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.w = [d["actor_w1"], d["actor_w2"], d["actor_w3"]]
        self.b = [d["actor_b1"], d["actor_b2"], d["actor_b3"]]
        self.mean = d["obs_mean"].astype(np.float64)
        self.var = d["obs_var"].astype(np.float64)
        self.dim = 61

    def action(self, obs61, priv38):
        x = np.asarray(obs61, dtype=np.float64)
        x = (x - self.mean[:61]) / np.sqrt(self.var[:61] + 1e-8)
        for i in range(2):
            x = np.maximum(x @ self.w[i] + self.b[i], 0.0)
        mu = x @ self.w[2] + self.b[2]
        return jnp.asarray(mu.astype(np.float32))


class StudentPklPolicy:
    """61 维：学生 Flax checkpoint（续训/对比用）"""

    def __init__(self, ckpt_path: str):
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        self.params = ck["params"]
        rms = ck.get("obs_rms")
        if rms is None:
            raise SystemExit("学生 pkl 缺 obs_rms")
        self.mean = jnp.asarray(rms.mean)
        self.var = jnp.asarray(rms.var)
        import jax
        m = make_student()
        self._apply = jax.jit(lambda p, o: m.apply(p, o))
        self.dim = 61

    def action(self, obs61, priv38):
        o = (obs61 - self.mean) / jnp.sqrt(self.var + 1e-8)
        mu, _, _ = self._apply(self.params, o)
        return mu


# ------------------------------------------------------------------ 主流程
def make_env():
    cfg = EnvConfig(
        dr_enabled=False,                    # 标称环境
        max_steps=10 ** 9,                   # 关截断
        cmd_resample_steps=(10 ** 9, 10 ** 9),   # 关内部重采样（由本脚本喂指令）
        gait_mode="sine",
    )
    return DogEnvMJX(n_envs=1, config=cfg)


def yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def run_plan(policy, env, plan, seed=0):
    step_fn = jax.jit(lambda st, a, m: env.step(st, a, m))
    mx = env._mx_eff
    st, obs = env.reset(jax.random.PRNGKey(seed), env._dr_model)

    w, x, y, z = [float(v) for v in np.asarray(st.dx.qpos[0, 3:7])]
    yaw0 = yaw_of((w, x, y, z))
    rows = []
    for name, cmd, n in plan:
        c = jnp.array([cmd], dtype=jnp.float32)
        p0 = np.asarray(st.dx.qpos[0, 0:3]).copy()
        yaw_a = yaw0 if not rows else rows[-1]["yaw_end"]
        vx_l, vy_l, wz_l, z_l, a_l = [], [], [], [], []
        obs_l = []
        fell = None
        for i in range(n):
            st = st._replace(command=c)
            priv = env.get_privileged_obs(st)
            act = policy.action(obs, priv)
            act = jnp.clip(act, -1.0, 1.0)
            a_l.append(float(jnp.mean(jnp.abs(act))))
            obs_l.append(np.asarray(obs[0], dtype=np.float64))
            st, obs, rew, term, trunc, info = step_fn(st, act, mx)
            vx_l.append(float(info["lin_vel_b"][0, 0]))
            vy_l.append(float(info["lin_vel_b"][0, 1]))
            wz_l.append(float(info["ang_vel_b"][0, 2]))
            z_l.append(float(info["base_z"][0]))
            if bool(term[0]):
                fell = i + 1
                break
        p1 = np.asarray(st.dx.qpos[0, 0:3])
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        yaw_now = yaw_of(np.asarray(st.dx.qpos[0, 3:7]))
        n_done = len(vx_l)
        rows.append({
            "name": name, "cmd": cmd, "steps": n_done, "fell": fell,
            "vx": float(np.mean(vx_l)), "vy": float(np.mean(vy_l)),
            "wz": float(np.mean(wz_l)), "z": float(np.mean(z_l)),
            "act": float(np.mean(a_l)) if a_l else 0.0,
            "obs": np.array(obs_l) if obs_l else np.zeros((0, 61)),
            "dx": dx, "dy": dy,
            "yaw_end": yaw_now,
            "dheading": float(np.degrees(yaw_now - yaw_a)),
        })
        if fell:
            break
    return rows


def report_obs_stats(rows):
    """与 run_policy.py 的 _report_obs_stats 对齐，用于逐维对比实机/仿真"""
    BLOCKS = [("base_z", 0, 1), ("g_b", 1, 4), ("lin_b", 4, 7),
              ("ang_b", 7, 10), ("q_sim", 10, 22), ("dq_sim", 22, 34),
              ("a_prev", 34, 46), ("phase", 46, 48), ("cmd", 58, 61)]
    for r in rows:
        O = r.get("obs")
        if O is None or len(O) == 0:
            continue
        print(f"\n--- 观测逐块统计【仿真】{r['name']}（{len(O)} 帧）---")
        for nm, a, b in BLOCKS:
            m = O[:, a:b].mean(axis=0)
            s = O[:, a:b].std(axis=0)
            print(f"{nm:<8}{b - a:>4}   {np.array2string(m, precision=3, max_line_width=200)}")
            print(f"{'':<8}{'':>4}   std {np.array2string(s, precision=3, max_line_width=200)}")


def report(rows, policy_name):
    print(f"\n=== {policy_name} ===")
    print(f"{'段':<9}{'指令(vx,vy,yaw)':>20}{'实测(机身系)':>26}{'世界位移':>18}"
          f"{'直线度':>9}{'航向漂':>8}{'z':>7}{'|a|':>6}{'结果':>8}")
    print("-" * 124)
    for r in rows:
        c = r["cmd"]
        straight = (r["dy"] / r["dx"]) if abs(r["dx"]) > 1e-3 else float("nan")
        cmd_s = "({:+.2f},{:+.2f},{:+.2f})".format(c[0], c[1], c[2])
        meas_s = "{:+.3f}/{:+.3f}/{:+.3f}".format(r["vx"], r["vy"], r["wz"])
        pos_s = "{:+.2f}/{:+.2f}".format(r["dx"], r["dy"])
        res_s = ("摔@%d" % r["fell"]) if r["fell"] else "存活"
        print(f"{r['name']:<9}{cmd_s:>20}{meas_s:>26}{pos_s:>18}"
              f"{straight:>9.3f}{r['dheading']:>8.1f}{r['z']:>7.3f}"
              f"{r.get('act', 0.0):>6.2f}{res_s:>8}")
    print("-" * 124)
    print("判读：纯前进段的 dy/dx 应 <0.05、航向漂 <3°；vx 应贴近指令。")


def main():
    ap = argparse.ArgumentParser(description="策略确定性评估（分段 + 直线度）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="教师 Flax checkpoint（99 维）")
    g.add_argument("--npz", help="学生部署包（61 维，numpy 前向）")
    g.add_argument("--student", help="学生 Flax checkpoint（61 维）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-stats", action="store_true",
                    help="额外打印每段观测的逐块 mean/std，用于与实机"
                         "run_policy 退出时打印的同一张表逐维对比")
    args = ap.parse_args()

    if args.ckpt:
        policy = TeacherPolicy(args.ckpt)
        name = f"教师 {Path(args.ckpt).name}"
    elif args.npz:
        policy = StudentNpzPolicy(args.npz)
        name = f"学生(npz) {Path(args.npz).name}"
    else:
        policy = StudentPklPolicy(args.student)
        name = f"学生(pkl) {Path(args.student).name}"

    print(f"策略：{name}  输入 {policy.dim} 维")
    env = make_env()
    rows = run_plan(policy, env, DEFAULT_PLAN, seed=args.seed)
    report(rows, name)
    if args.obs_stats:
        report_obs_stats(rows)

    out = REPO / "logs" / "eval_policy_last.json"
    out.parent.mkdir(exist_ok=True)
    rows_json = [{k: v for k, v in r.items() if k != "obs"} for r in rows]
    out.write_text(json.dumps(rows_json, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n明细已存 {out}")


if __name__ == "__main__":
    main()
