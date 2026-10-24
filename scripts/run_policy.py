#!/usr/bin/env python3
"""run_policy.py —— 学生策略上实机（固定前进指令，直线测试版）

设计原则：第一次上机只验证「仿真里已经稳的事」，所以
  · 指令固定 (vx, 0, 0)：不支持手柄、不转向、不横移（策略本来也不会）
  · 两级门槛：先 `--dry-run`（电机 kp=0、腿软，只跑数据链路，狗不动），
    确认 obs 合理后再 `--go`
  · 动作限幅 ±1 + 变化率限制（每步最多 0.15），防突发
  · base_z 低于阈值 → 立即停止策略、回到站姿保持（不摔）
  · 退出流程固定：动作淡出 → 站姿保持 → 趴下 → 失能

观测 61 维（与训练一致，见 agents.md 决策 5.1）：
  [0] base_z（FK 估计器）  [1:4] projected gravity  [4:7] 机身系线速度
  [7:10] 陀螺  [10:22] 关节角(sim 系)  [22:34] 关节速度(sim 系)
  [34:46] 上一步动作  [46:58] Q_DEFAULT 常量  [58:61] 指令

用法（NUC，先让狗趴好）：
  sudo python3 scripts/run_policy.py --dry-run                 # 门槛 1：腿软验证
  sudo python3 scripts/run_policy.py --go --vx 0.2 --duration 8 # 门槛 2：真跑
  sudo python3 scripts/run_policy.py --go --vx 0.2 --no-imu     # 没有 IMU 时的降级
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deploy.motor_client import MotorBus                       # noqa: E402
from deploy.kin_estimator import KinEstimator, leg_fk, rpy_to_R  # noqa: E402

# ---------------------------------------------------------------- 常量
JOINTS = [(ch, mid) for ch in (0, 1, 2, 3) for mid in (0x01, 0x02, 0x03)]
CH_NAMES = {0: "FL", 1: "FR", 2: "RL", 3: "RR"}
MID_NAMES = {0x01: "hip", 0x02: "thigh", 0x03: "calf"}

# 实机→仿真符号矩阵（2026-09-26 实测）
S = [+1, +1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1]
FRONT_CALF = (2, 5)
REAR_CALF = (8, 11)

# 仿真当前 Q_DEFAULT（2026-10-23 加电实测标定），用于拼 obs[46:58] 与动作基准
Q_DEFAULT_NEW = np.array([
    -0.0134, 0.8611, -1.7340,
    0.0160, 0.8531, -1.7100,
    0.0088, 0.8801, -1.7210,
    -0.0133, 0.9111, -1.6946], dtype=np.float64)
# 部署映射的参照点（**保持旧值不变**，见 agents.md 部署映射说明）
Q_DEFAULT_SIM_REF = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)

# 步态常量（与 envs/gait.py 完全一致）
A_HIP, A_THIGH, A_CALF = 0.10, 0.35, 0.35
FREQ_MIN, FREQ_MAX = 2.0, 4.0
DT = 0.02

JNT_LO = np.array([-1.0, -1.5, -2.5] * 4)
JNT_HI = np.array([1.0, 3.0, -0.85] * 4)

CTRL_HZ = 50.0
KP_FINAL, KD_FINAL = 150.0, 2.0
RISE_SEC = 10.0


# ---------------------------------------------------------------- 交互
_line_q: "queue.Queue[str]" = queue.Queue()


def _stdin_watcher() -> None:
    while True:
        try:
            _line_q.put(input())
        except EOFError:
            return


def got_line() -> bool:
    try:
        _line_q.get_nowait()
        return True
    except queue.Empty:
        return False


# ---------------------------------------------------------------- 姿态工具
def load_json(name: str) -> dict:
    p = REPO_ROOT / "deploy" / name
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def load_pose(name: str) -> np.ndarray:
    return np.asarray(load_json(f"{name}_calibration.json")["q_default_real"], dtype=np.float64)


def apply_trim(q: np.ndarray, tf: float, tr: float) -> np.ndarray:
    out = q.copy()
    for i in FRONT_CALF:
        out[i] = q[i] + S[i] * (-tf)
    for i in REAR_CALF:
        out[i] = q[i] + S[i] * (-tr)
    return out


def smooth(t: float, T: float) -> float:
    x = min(max(t / T, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * x))


def blend(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    return a + (b - a) * s


def quat_to_rpy(q) -> np.ndarray:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- 基准步态（numpy 版，与 envs/gait.py 对齐）
def gait_np(step: int, cmd) -> np.ndarray:
    vx, vy, yaw = float(cmd[0]), float(cmd[1]), float(cmd[2])
    t = min(max(abs(vx) / 1.5, 0.0), 1.0)
    freq = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * t
    phase = (step * DT * freq * 2 * math.pi) % (2 * math.pi)
    pa, pb = phase, phase + math.pi

    amp_thigh = A_THIGH * min(max(abs(vx) / 1.0, 0.0), 1.0)
    amp_calf = A_CALF * min(max(abs(vx) / 1.0, 0.0), 1.0)
    amp_hip = A_HIP * min(max(abs(vy) / 0.5, 0.0), 1.0)
    yaw_hip = A_HIP * 0.5 * min(max(yaw / 2.0, -1.0), 1.0)
    hip_sign = 1.0 if vy >= 0 else -1.0

    fl_hip = amp_hip * hip_sign + yaw_hip
    fr_hip = -amp_hip * hip_sign - yaw_hip
    rl_hip = amp_hip * hip_sign - yaw_hip
    rr_hip = -amp_hip * hip_sign + yaw_hip

    delta = 0.0 if vx >= 0 else math.pi
    g = np.array([
        fl_hip, amp_thigh * math.sin(pa), amp_calf * math.sin(pa + delta),
        fr_hip, amp_thigh * math.sin(pb), amp_calf * math.sin(pb + delta),
        rl_hip, amp_thigh * math.sin(pb), amp_calf * math.sin(pb + delta),
        rr_hip, amp_thigh * math.sin(pa), amp_calf * math.sin(pa + delta),
    ])
    return g


# ---------------------------------------------------------------- 策略
class StudentNpz:
    def __init__(self, path):
        d = np.load(path)
        self.w = [d["actor_w1"].astype(np.float64), d["actor_w2"].astype(np.float64),
                  d["actor_w3"].astype(np.float64)]
        self.b = [d["actor_b1"].astype(np.float64), d["actor_b2"].astype(np.float64),
                  d["actor_b3"].astype(np.float64)]
        self.mean = d["obs_mean"].astype(np.float64)[:61]
        self.var = d["obs_var"].astype(np.float64)[:61]
        self.dim = 61

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        for i in range(2):
            x = np.maximum(x @ self.w[i] + self.b[i], 0.0)
        return x @ self.w[2] + self.b[2]


# ---------------------------------------------------------------- IMU（复用 test_imu 的解析器）
class ImuReader:
    def __init__(self, port, baud=115200, gyro_deg=False):
        import serial
        from test_imu import ImuParser, TYPE_GYR, TYPE_QUA
        self.ser = serial.Serial(port, baud, timeout=0.02)
        self.parser = ImuParser()
        self._gyr_t, self._qua_t = TYPE_GYR, TYPE_QUA
        self.gyro_deg = gyro_deg

    def latest(self):
        chunk = self.ser.read(4096)
        if chunk:
            self.parser.feed(chunk)
        qua = self.parser.latest.get(self._qua_t)
        gyr = self.parser.latest.get(self._gyr_t)
        if qua is None or gyr is None:
            return None, None
        g = np.asarray(gyr, dtype=np.float64)
        if self.gyro_deg:
            g = np.deg2rad(g)
        return np.asarray(qua, dtype=np.float64), g

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------- 主流程
class Runner:
    def __init__(self, args):
        self.args = args
        self.q_lie = load_pose("lie")
        self.q_stand = load_pose("stance")
        tri = load_json("posture_trim.json") if (REPO_ROOT / "deploy" / "posture_trim.json").exists() else {}
        self.q_eff = apply_trim(self.q_stand, float(tri.get("front", 0.0)),
                                float(tri.get("rear", 0.0)))
        self.policy = StudentNpz(args.npz)
        self.kin = KinEstimator(lpf_alpha=0.5)
        self.cmd = np.array([args.vx, 0.0, 0.0])
        self.imu = None
        self.last_action = np.zeros(12)
        self.step = 0
        self.base_z = 0.0

    # ---- 工具 ----
    def send(self, bus, q_real, kp, kd):
        for ch, mid in JOINTS:
            i = ch * 3 + (mid - 1)
            bus.send_mit(ch, mid, float(q_real[i]), 0.0, kp, kd, 0.0)

    def read_q(self, bus):
        q = np.full(12, np.nan)
        dq = np.full(12, np.nan)
        for ch, mid in JOINTS:
            fb = bus.get_feedback(ch, mid, max_age=0.3)
            i = ch * 3 + (mid - 1)
            if fb and "pos" in fb:
                q[i] = fb["pos"]
                dq[i] = fb.get("vel", 0.0)
        return q, dq

    @staticmethod
    def real_to_sim(q_real, dq_real, q_eff):
        q_sim = np.array([Q_DEFAULT_SIM_REF[i] + S[i] * (q_real[i] - q_eff[i])
                          for i in range(12)])
        dq_sim = np.array([S[i] * dq_real[i] for i in range(12)])
        return q_sim, dq_sim

    @staticmethod
    def sim_to_real(q_sim, q_eff):
        return np.array([q_eff[i] + S[i] * (q_sim[i] - Q_DEFAULT_SIM_REF[i])
                         for i in range(12)])

    @staticmethod
    def contact_mask(q_sim):
        zs = [leg_fk(q_sim[3 * i:3 * i + 3], i)[0][2] for i in range(4)]
        thr = min(zs) + 0.03
        return np.array([z < thr for z in zs])

    def build_obs(self, q_sim, dq_sim, gyro, g_b, lin_b):
        return np.concatenate([
            [self.base_z], g_b, lin_b, gyro,
            q_sim, dq_sim, self.last_action,
            Q_DEFAULT_NEW, self.cmd])

    # ---- 主入口 ----
    def run(self):
        args = self.args
        dry = args.dry_run
        if args.imu_port:
            self.imu = ImuReader(args.imu_port, gyro_deg=args.imu_gyro_deg)
            print(f"IMU 已打开 {args.imu_port}")

        threading.Thread(target=_stdin_watcher, daemon=True).start()
        bus = MotorBus()
        kp_target = 0.0 if dry else KP_FINAL

        print("=" * 78)
        print(" 学生策略上实机 —— " + ("【dry-run】电机 kp=0，狗不会动" if dry else "【GO】真实运行"))
        print("=" * 78)
        print(f" 指令 vx={args.vx:+.2f} m/s，运行 {args.duration:.0f}s，kp={kp_target:.0f}")
        print(f" 策略 {args.npz}")
        if args.imu_port is None:
            print(" ⚠ 未接 IMU：projected gravity 固定 [0,0,-1]、陀螺置零（降级模式）")

        phase, t_phase = "WAIT", 0.0
        q_cmd = self.q_lie.copy()
        kp_now = 0.0
        t_last = time.time()
        run_t0 = 0.0
        n_loop, n_slow = 0, 0
        next_t = time.time()

        try:
            bus.enable_channels([0, 1, 2, 3])
            for ch, mid in JOINTS:
                bus.enable_motor(ch, mid)
                time.sleep(0.02)
                bus.send_mit(ch, mid, float(self.q_lie[ch * 3 + (mid - 1)]), 0.0, 0.0, 0.0, 0.0)

            print("\n电机已使能（kp=0，腿软）。请把狗摆成趴姿，按 Enter 开始起立…")

            while True:
                now = time.time()
                dt = now - t_last
                t_last = now
                t_phase += dt
                line = got_line()

                # ---------------- 状态机 ----------------
                if phase == "WAIT":
                    kp_now, q_cmd = 0.0, self.q_lie
                    if line:
                        phase, t_phase = "RISE", 0.0
                        print("\n>>> 起立中，请勿触碰狗 <<<")

                elif phase == "RISE":
                    s = smooth(t_phase, RISE_SEC)
                    q_cmd = blend(self.q_lie, self.q_eff, s)
                    kp_now = kp_target * min(t_phase / 3.0, 1.0)
                    if t_phase >= RISE_SEC:
                        phase, t_phase = "SETTLE", 0.0
                        print(f"\n>>> 已站立（kp={kp_target:.0f}）。松手，稳定 2 秒…")

                elif phase == "SETTLE":
                    kp_now, q_cmd = kp_target, self.q_eff
                    if t_phase >= 2.0:
                        phase = "WAIT_RUN"
                        print("\n>>> 数据链路自检（看 obs 是否合理），确认后按 Enter 启动策略")

                elif phase == "WAIT_RUN":
                    kp_now, q_cmd = kp_target, self.q_eff
                    if line:
                        phase, t_phase = "RUN", 0.0
                        run_t0 = time.time()
                        self.step = 0
                        self.last_action[:] = 0.0
                        print(f"\n>>> 策略运行 {args.duration:.0f}s，请跟走并随时准备扶 <<<")

                elif phase == "RUN":
                    kp_now = kp_target
                    if time.time() - run_t0 >= args.duration or (not dry and line):
                        phase, t_phase = "FADE", 0.0
                        print("\n>>> 结束，动作淡出并回到站姿 <<<")
                        continue
                    if not dry and self.base_z < args.z_abort:
                        phase, t_phase = "FADE", 0.0
                        print(f"\n>>> ⚠ base_z={self.base_z:.3f} 低于阈值，中止 <<<")
                        continue

                elif phase == "FADE":
                    kp_now, q_cmd = kp_target, self.q_eff
                    if t_phase >= 1.0:
                        phase, t_phase = "WAIT_SIT", 0.0
                        print("\n>>> 已回到站姿保持。按 Enter 趴下并失能 <<<")

                elif phase == "WAIT_SIT":
                    kp_now, q_cmd = kp_target, self.q_eff
                    if line:
                        phase, t_phase = "SIT", 0.0
                        print("\n>>> 趴下中 <<<")

                elif phase == "SIT":
                    s = smooth(t_phase, RISE_SEC)
                    q_cmd = blend(self.q_eff, self.q_lie, s)
                    kp_now = kp_target
                    if t_phase >= RISE_SEC:
                        phase, t_phase = "LOWER", 0.0

                elif phase == "LOWER":
                    r = min(t_phase / 2.0, 1.0)
                    kp_now = kp_target * (1.0 - r)
                    q_cmd = self.q_lie
                    if r >= 1.0:
                        print("\n>>> 已趴下，失能 <<<")
                        break

                # ---------------- 数据链路 ----------------
                q_real, dq_real = self.read_q(bus)
                valid = not np.any(np.isnan(q_real))
                if valid and phase in ("SETTLE", "WAIT_RUN", "RUN", "FADE", "WAIT_SIT"):
                    q_sim, dq_sim = self.real_to_sim(q_real, dq_real, self.q_eff)
                    if self.imu is not None:
                        qua, gyr = self.imu.latest()
                    else:
                        qua, gyr = None, None
                    if qua is not None:
                        rpy = quat_to_rpy(qua)
                        R = quat_to_mat(qua)
                        g_b = -R[2, :]
                    else:
                        rpy = np.zeros(3)
                        R = np.eye(3)
                        g_b = np.array([0.0, 0.0, -1.0])
                    if gyr is None:
                        gyr = np.zeros(3)
                    contact = self.contact_mask(q_sim)
                    z_est, v_est = self.kin.update(q_sim, dq_sim, rpy, gyr, contact)
                    self.base_z = float(z_est)
                    lin_b = np.asarray(v_est, dtype=np.float64)

                    if phase == "RUN":
                        obs = self.build_obs(q_sim, dq_sim, gyr, g_b, lin_b)
                        a = self.policy.act(obs)
                        a = np.clip(a, -1.0, 1.0)
                        # 变化率限制：每步最多 0.15
                        delta_a = np.clip(a - self.last_action, -0.15, 0.15)
                        a = self.last_action + delta_a
                        self.last_action = a.copy()
                        g = gait_np(self.step, self.cmd)
                        q_des_sim = np.clip(Q_DEFAULT_NEW + g + 0.2 * a, JNT_LO, JNT_HI)
                        q_cmd = self.sim_to_real(q_des_sim, self.q_eff)
                        self.step += 1
                        if self.step % 25 == 0:
                            print(f"    t={self.step * DT:5.2f}s  z={self.base_z:.3f}  "
                                  f"g_b=[{g_b[0]:+.2f},{g_b[1]:+.2f},{g_b[2]:+.2f}]  "
                                  f"v_b=[{lin_b[0]:+.2f},{lin_b[1]:+.2f}]  "
                                  f"|a|={np.abs(a).mean():.2f}")
                    elif phase in ("SETTLE", "WAIT_RUN") and n_loop % 25 == 0:
                        print(f"    z={self.base_z:.3f}  g_b=[{g_b[0]:+.2f},{g_b[1]:+.2f},{g_b[2]:+.2f}]  "
                              f"陀螺=[{gyr[0]:+.2f},{gyr[1]:+.2f},{gyr[2]:+.2f}]  "
                              f"v_b=[{lin_b[0]:+.2f},{lin_b[1]:+.2f},{lin_b[2]:+.2f}]  "
                              f"contact={contact.astype(int)}")

                # ---------------- 下发 ----------------
                self.send(bus, q_cmd, kp_now, KD_FINAL if kp_now > 0 else 0.0)

                # 50Hz 节拍
                n_loop += 1
                next_t += 1.0 / CTRL_HZ
                sleep = next_t - time.time()
                if sleep < 0:
                    n_slow += 1
                    next_t = time.time()
                else:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            print("\n\n!!! Ctrl+C：立即失能（狗会摔）!!!")
        except Exception:
            import traceback
            print("\n\n!!! 异常退出 !!!")
            traceback.print_exc()
        finally:
            if self.imu is not None:
                self.imu.close()
            for ch, mid in JOINTS:
                try:
                    bus.disable_motor(ch, mid)
                except Exception:
                    pass
            if n_loop:
                print(f"\n循环统计：{n_loop} 次，其中 {n_slow} 次没跟上 50Hz"
                      f"（{100.0 * n_slow / max(1, n_loop):.1f}%）")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def main():
    ap = argparse.ArgumentParser(description="学生策略上实机（固定前进指令）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="门槛 1：kp=0 腿软，只验证数据链路")
    g.add_argument("--go", action="store_true", help="门槛 2：真实运行")
    ap.add_argument("--npz", default=str(REPO_ROOT / "checkpoints" / "dog_cmd_mjx_student.npz"))
    ap.add_argument("--vx", type=float, default=0.2, help="前进指令（默认 0.2 m/s）")
    ap.add_argument("--duration", type=float, default=8.0, help="运行时长（秒）")
    ap.add_argument("--z-abort", type=float, default=0.26, help="base_z 低于此值立即中止")
    ap.add_argument("--imu-port", default=None, help="IMU 串口，如 /dev/ttyACM0")
    ap.add_argument("--imu-gyro-deg", action="store_true", help="IMU 陀螺输出是 °/s")
    args = ap.parse_args()
    if not Path(args.npz).exists():
        sys.exit(f"找不到部署包 {args.npz}（先在服务器上跑完蒸馏）")
    Runner(args).run()


if __name__ == "__main__":
    main()
