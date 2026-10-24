#!/usr/bin/env python3
"""run_policy.py —— 学生策略上实机（单键/手柄触发，直线测试版）

输入源（--input）：
  key    键盘（默认）：Enter 推进阶段；**a 切换走动/停**；q 退出
  pad    手柄：Enter 推进阶段；**按住按钮（默认 A/0）走，松开停**（真 deadman）
  enter  无触发：进入 RUN 后一直走，按 Enter 停（旧行为）

指令固定为 (vx, 0, 0)——策略不会转向/横移，所以不提供纵向以外的输入。
vx 从 0 到目标值有 0.3 s 线性斜坡，避免步态幅值突跳。

两级门槛：
  1. --dry-run：电机 kp=0、腿软，完整跑数据链路（电机反馈/IMU/FK/拼 obs/推理），狗不动
  2. --go：真实运行，动作变化率限 0.15，base_z 低于阈值立即中止

用法（NUC，先让狗趴好）：
  sudo python3 scripts/run_policy.py --dry-run --imu-port /dev/ttyACM0              # 键盘
  sudo python3 scripts/run_policy.py --dry-run --input pad --imu-port /dev/ttyACM0  # 手柄
  sudo python3 scripts/run_policy.py --go --vx 0.3 --input pad --imu-port /dev/ttyACM0
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
from deploy.kin_estimator import KinEstimator, leg_fk          # noqa: E402

# ---------------------------------------------------------------- 常量
JOINTS = [(ch, mid) for ch in (0, 1, 2, 3) for mid in (0x01, 0x02, 0x03)]
S = [+1, +1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1]
FRONT_CALF = (2, 5)
REAR_CALF = (8, 11)

Q_DEFAULT_NEW = np.array([
    -0.0134, 0.8611, -1.7340,
    0.0160, 0.8531, -1.7100,
    0.0088, 0.8801, -1.7210,
    -0.0133, 0.9111, -1.6946], dtype=np.float64)
Q_DEFAULT_SIM_REF = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)

A_HIP, A_THIGH, A_CALF = 0.10, 0.35, 0.35
FREQ_MIN, FREQ_MAX = 2.0, 4.0
DT = 0.02
JNT_LO = np.array([-1.0, -1.5, -2.5] * 4)
JNT_HI = np.array([1.0, 3.0, -0.85] * 4)

CTRL_HZ = 50.0
KP_FINAL, KD_FINAL = 150.0, 2.0
RISE_SEC = 10.0
RAMP_SEC = 0.3


# ---------------------------------------------------------------- 姿态工具
def load_json(name: str) -> dict:
    p = REPO_ROOT / "deploy" / name
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def load_pose(name: str) -> np.ndarray:
    return np.asarray(load_json(f"{name}_calibration.json")["q_default_real"],
                      dtype=np.float64)


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
    return np.array([
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def gait_np(step: int, cmd) -> np.ndarray:
    """与 envs/gait.py 逐位对齐（对拍偏差 2.3e-6 rad）"""
    vx, vy, yaw = float(cmd[0]), float(cmd[1]), float(cmd[2])
    t = min(max(abs(vx) / 1.5, 0.0), 1.0)
    freq = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * t
    phase = (step * DT * freq * 2 * math.pi) % (2 * math.pi)
    pa, pb = phase, phase + math.pi
    amp_thigh = A_THIGH * min(max(abs(vx) / 1.0, 0.0), 1.0)
    amp_calf = A_CALF * min(max(abs(vx) / 1.0, 0.0), 1.0)
    amp_hip = A_HIP * min(max(abs(vy) / 0.5, 0.0), 1.0)
    yaw_hip = A_HIP * 0.5 * min(max(yaw / 2.0, -1.0), 1.0)
    hs = 1.0 if vy >= 0 else -1.0
    delta = 0.0 if vx >= 0 else math.pi
    return np.array([
        amp_hip * hs + yaw_hip, amp_thigh * math.sin(pa), amp_calf * math.sin(pa + delta),
        -amp_hip * hs - yaw_hip, amp_thigh * math.sin(pb), amp_calf * math.sin(pb + delta),
        amp_hip * hs - yaw_hip, amp_thigh * math.sin(pb), amp_calf * math.sin(pb + delta),
        -amp_hip * hs + yaw_hip, amp_thigh * math.sin(pa), amp_calf * math.sin(pa + delta)])


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

    def act(self, obs):
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        for i in range(2):
            x = np.maximum(x @ self.w[i] + self.b[i], 0.0)
        return x @ self.w[2] + self.b[2]


# ---------------------------------------------------------------- 输入源
class KeyInput:
    """单字符键盘：Enter 推进阶段，a 切换走/停，q 退出（cbreak 模式）"""

    def __init__(self):
        import termios
        import tty
        self._termios = termios
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.on = False

    def poll(self):
        import select
        advance = quit_ = False
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                advance = True
            elif ch in ("a", "A"):
                self.on = not self.on
            elif ch in ("q", "Q", "\x03"):
                quit_ = True
        return self.on, advance, quit_

    def close(self):
        try:
            self._termios.tcsetattr(self.fd, self._termios.TCSADRAIN, self.old)
        except Exception:
            pass


class PadInput:
    """手柄 deadman：按住按钮走，松开停；Enter 推进阶段（stdin 线模式）"""

    def __init__(self, button=0):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.joystick.init()
        n = pygame.joystick.get_count()
        if n == 0:
            raise RuntimeError("没有检测到手柄（检查 USB / 权限 / 是否被占用）")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        self.button = button
        self._prev = False
        self.q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._watch, daemon=True).start()
        print(f"手柄：{self.joy.get_name()}  "
              f"axes={self.joy.get_numaxes()} buttons={self.joy.get_numbuttons()}")

    def _watch(self):
        while True:
            try:
                self.q.put(input())
            except EOFError:
                return

    def poll(self):
        self.pg.event.pump()
        b = bool(self.joy.get_button(self.button))
        edge = b and not self._prev      # 按下沿也当"推进阶段"
        self._prev = b
        advance = edge
        try:
            self.q.get_nowait()
            advance = True
        except queue.Empty:
            pass
        return b, advance, False

    def close(self):
        try:
            self.pg.quit()
        except Exception:
            pass


class EnterInput:
    """旧行为：Enter 推进阶段，RUN 后一直走"""

    def __init__(self):
        self.q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        while True:
            try:
                self.q.put(input())
            except EOFError:
                return

    def poll(self):
        advance = False
        try:
            self.q.get_nowait()
            advance = True
        except queue.Empty:
            pass
        return True, advance, False

    def close(self):
        pass


# ---------------------------------------------------------------- IMU
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
        trim_p = REPO_ROOT / "deploy" / "posture_trim.json"
        tri = json.loads(trim_p.read_text(encoding="utf-8")) if trim_p.exists() else {}
        self.q_eff = apply_trim(self.q_stand, float(tri.get("front", 0.0)),
                                float(tri.get("rear", 0.0)))
        self.policy = StudentNpz(args.npz)
        self.kin = KinEstimator(lpf_alpha=0.5)
        self.imu = None
        self.last_action = np.zeros(12)
        self.step = 0
        self.base_z = 0.0
        self.vx_eff = 0.0
        self.walking = False

    # ---- 电机 ----
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
        q_sim = Q_DEFAULT_SIM_REF + np.array(
            [S[i] * (q_real[i] - q_eff[i]) for i in range(12)])
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

    def build_obs(self, q_sim, dq_sim, gyro, g_b, lin_b, cmd):
        return np.concatenate([[self.base_z], g_b, lin_b, gyro, q_sim, dq_sim,
                               self.last_action, Q_DEFAULT_NEW, cmd])

    # ---- 主循环 ----
    def run(self):
        args = self.args
        dry = args.dry_run
        kp_target = 0.0 if dry else KP_FINAL

        if args.imu_port:
            self.imu = ImuReader(args.imu_port, gyro_deg=args.imu_gyro_deg)
            print(f"IMU 已打开 {args.imu_port}")

        inp = {"key": KeyInput, "pad": lambda: PadInput(args.pad_button),
               "enter": EnterInput}[args.input]()

        bus = MotorBus()
        print("=" * 78)
        print(" 学生策略上实机 —— " + ("【dry-run】kp=0，狗不会动" if dry else "【GO】真实运行"))
        print("=" * 78)
        print(f" 输入源={args.input}  目标 vx={args.vx:+.2f}  最长 {args.duration:.0f}s  kp={kp_target:.0f}")
        print(f" 策略 {args.npz}")
        if args.input == "key":
            print(" 按键：Enter 推进阶段 / a 切换走停 / q 退出")
        elif args.input == "pad":
            print(f" 按键：Enter 推进阶段 / 按住按钮 {args.pad_button} 走，松开停")
        else:
            print(" 按键：Enter 推进阶段（RUN 中 Enter 停止）")
        if args.imu_port is None:
            print(" ⚠ 未接 IMU：g_b 固定 [0,0,-1]、陀螺置零 —— 不要用 --go")

        phase, t_phase = "WAIT", 0.0
        q_cmd = self.q_lie.copy()
        kp_now = 0.0
        t_last = time.time()
        run_t0 = 0.0
        n_loop = n_slow = 0
        next_t = time.time()
        last_walk_print = 0.0

        try:
            bus.enable_channels([0, 1, 2, 3])
            for ch, mid in JOINTS:
                bus.enable_motor(ch, mid)
                time.sleep(0.02)
                bus.send_mit(ch, mid, float(self.q_lie[ch * 3 + (mid - 1)]),
                             0.0, 0.0, 0.0, 0.0)
            print("\n电机已使能（kp=0，腿软）。把狗摆成趴姿，按 Enter 起立…")

            while True:
                now = time.time()
                dt = now - t_last
                t_last = now
                t_phase += dt

                walking, advance, quit_ = inp.poll()
                if quit_:
                    print("\n>>> 收到 q，退出 <<<")
                    break

                if phase == "WAIT":
                    kp_now, q_cmd = 0.0, self.q_lie
                    if advance:
                        phase, t_phase = "RISE", 0.0
                        print("\n>>> 起立中，请勿触碰狗 <<<")

                elif phase == "RISE":
                    q_cmd = blend(self.q_lie, self.q_eff, smooth(t_phase, RISE_SEC))
                    kp_now = kp_target * min(t_phase / 3.0, 1.0)
                    if t_phase >= RISE_SEC:
                        phase, t_phase = "SETTLE", 0.0
                        print(f"\n>>> 已站立（kp={kp_target:.0f}）。松手，稳定 2 秒…")

                elif phase == "SETTLE":
                    kp_now, q_cmd = kp_target, self.q_eff
                    if t_phase >= 2.0:
                        phase = "WAIT_RUN"
                        print("\n>>> 看下面的 obs 自检：g_b≈[0,0,-1]、陀螺≈0、z≈0.31")
                        print(">>> 确认后按 Enter 启动策略（之后 a/按钮 控制走停）")

                elif phase == "WAIT_RUN":
                    kp_now, q_cmd = kp_target, self.q_eff
                    if advance:
                        phase, t_phase = "RUN", 0.0
                        run_t0 = time.time()
                        self.step = 0
                        self.vx_eff = 0.0
                        self.last_action[:] = 0.0
                        print("\n>>> 策略已启动：按 a（或按住手柄按钮）让它走 <<<")

                elif phase == "RUN":
                    kp_now = kp_target
                    if time.time() - run_t0 >= args.duration:
                        print(f"\n>>> 到达 {args.duration:.0f}s 上限，自动停止 <<<")
                        phase, t_phase = "FADE", 0.0
                        continue
                    if not dry and self.base_z < args.z_abort:
                        print(f"\n>>> ⚠ base_z={self.base_z:.3f} 过低，中止 <<<")
                        phase, t_phase = "FADE", 0.0
                        continue
                    if advance:      # RUN 里按 Enter = 停止（所有输入源一致）
                        print("\n>>> Enter：停止 <<<")
                        phase, t_phase = "FADE", 0.0
                        continue
                    # vx 斜坡
                    target = args.vx if walking else 0.0
                    dv = args.vx / max(RAMP_SEC, 1e-3) * dt
                    if self.vx_eff < target:
                        self.vx_eff = min(target, self.vx_eff + dv)
                    elif self.vx_eff > target:
                        self.vx_eff = max(target, self.vx_eff - dv)
                    if walking != self.walking and now - last_walk_print > 0.2:
                        self.walking = walking
                        last_walk_print = now
                        print(f"    {'>>> 走' if walking else '>>> 停'}")

                elif phase == "FADE":
                    kp_now, q_cmd = kp_target, self.q_eff
                    self.vx_eff = max(0.0, self.vx_eff - args.vx / RAMP_SEC * dt)
                    if t_phase >= 1.0 and self.vx_eff <= 1e-3:
                        phase, t_phase = "WAIT_SIT", 0.0
                        print("\n>>> 已回到站姿保持。按 Enter 趴下并失能 <<<")

                elif phase == "WAIT_SIT":
                    kp_now, q_cmd = kp_target, self.q_eff
                    if advance:
                        phase, t_phase = "SIT", 0.0
                        print("\n>>> 趴下中 <<<")

                elif phase == "SIT":
                    q_cmd = blend(self.q_eff, self.q_lie, smooth(t_phase, RISE_SEC))
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
                if not np.any(np.isnan(q_real)) and phase in (
                        "SETTLE", "WAIT_RUN", "RUN", "FADE", "WAIT_SIT"):
                    q_sim, dq_sim = self.real_to_sim(q_real, dq_real, self.q_eff)
                    qua, gyr = self.imu.latest() if self.imu is not None else (None, None)
                    if qua is not None:
                        rpy = quat_to_rpy(qua)
                        R = quat_to_mat(qua)
                        g_b = -R[2, :]
                    else:
                        rpy, R, g_b = np.zeros(3), np.eye(3), np.array([0.0, 0.0, -1.0])
                    if gyr is None:
                        gyr = np.zeros(3)
                    contact = self.contact_mask(q_sim)
                    z_est, v_est = self.kin.update(q_sim, dq_sim, rpy, gyr, contact)
                    self.base_z = float(z_est)
                    lin_b = np.asarray(v_est, dtype=np.float64)

                    if phase == "RUN":
                        cmd = np.array([self.vx_eff, 0.0, 0.0])
                        obs = self.build_obs(q_sim, dq_sim, gyr, g_b, lin_b, cmd)
                        a = np.clip(self.policy.act(obs), -1.0, 1.0)
                        a = self.last_action + np.clip(a - self.last_action, -0.15, 0.15)
                        self.last_action = a.copy()
                        g = gait_np(self.step, cmd)
                        q_des_sim = np.clip(Q_DEFAULT_NEW + g + 0.2 * a, JNT_LO, JNT_HI)
                        q_cmd = self.sim_to_real(q_des_sim, self.q_eff)
                        self.step += 1
                        if self.step % 25 == 0:
                            print(f"    t={self.step * DT:5.1f}s vx={self.vx_eff:+.2f} "
                                  f"z={self.base_z:.3f} "
                                  f"g_b=[{g_b[0]:+.2f},{g_b[1]:+.2f}] "
                                  f"v_b=[{lin_b[0]:+.2f},{lin_b[1]:+.2f}] "
                                  f"|a|={np.abs(a).mean():.2f}")
                    elif phase in ("SETTLE", "WAIT_RUN") and n_loop % 25 == 0:
                        print(f"    z={self.base_z:.3f} "
                              f"g_b=[{g_b[0]:+.2f},{g_b[1]:+.2f},{g_b[2]:+.2f}] "
                              f"陀螺=[{gyr[0]:+.2f},{gyr[1]:+.2f},{gyr[2]:+.2f}] "
                              f"v_b=[{lin_b[0]:+.2f},{lin_b[1]:+.2f},{lin_b[2]:+.2f}] "
                              f"contact={contact.astype(int)}")

                self.send(bus, q_cmd, kp_now, KD_FINAL if kp_now > 0 else 0.0)

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
            inp.close()
            if self.imu is not None:
                self.imu.close()
            for ch, mid in JOINTS:
                try:
                    bus.disable_motor(ch, mid)
                except Exception:
                    pass
            if n_loop:
                print(f"\n循环 {n_loop} 次，{n_slow} 次没跟上 50Hz"
                      f"（{100.0 * n_slow / max(1, n_loop):.1f}%）")
            print("电机已失能。")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def main():
    ap = argparse.ArgumentParser(description="学生策略上实机（单键/手柄触发）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="门槛 1：kp=0 腿软，只验证数据链路")
    g.add_argument("--go", action="store_true", help="门槛 2：真实运行")
    ap.add_argument("--input", choices=["key", "pad", "enter"], default="key",
                    help="输入源：key=键盘 a 切换 / pad=手柄 deadman / enter=旧行为")
    ap.add_argument("--pad-button", type=int, default=0, help="手柄按钮编号（默认 0=A）")
    ap.add_argument("--npz", default=str(REPO_ROOT / "checkpoints" / "dog_cmd_mjx_student.npz"))
    ap.add_argument("--vx", type=float, default=0.3, help="前进速度（默认 0.3 m/s）")
    ap.add_argument("--duration", type=float, default=20.0, help="RUN 阶段最长时长（秒）")
    ap.add_argument("--z-abort", type=float, default=0.26, help="base_z 低于此值立即中止")
    ap.add_argument("--imu-port", default=None, help="IMU 串口，如 /dev/ttyACM0")
    ap.add_argument("--imu-gyro-deg", action="store_true", help="IMU 陀螺输出是 °/s")
    args = ap.parse_args()
    if not Path(args.npz).exists():
        sys.exit(f"找不到部署包 {args.npz}")
    Runner(args).run()


if __name__ == "__main__":
    main()
