#!/usr/bin/env python3
"""calibrate_powered_stance.py —— 【加电】站立姿态标定（v25 前置，NUC 上运行）

为什么需要这个脚本
------------------
现有 `deploy/stance_calibration.json` 是 `calibrate_stance.py` 在 **kp=0、手扶**
状态下读的。手扶时腿放松略屈；加电 kp=150 撑起来后腿会伸直一点。
2026-10-23 证据链：
  1. 实机 kp=150 站立：底盘 270mm / 身体几何中心 320mm
  2. 仿真同一姿态的几何极限只有 308mm（kp 再大也到不了 320）
  3. 9-26 记录的稳态误差换算到仿真系后四条小腿**全部同号**（+0.036~+0.114）
  4. 0.074 rad × 小腿力臂 0.188 m/rad ≈ 1.4cm ≈ 高度差
→ 结论：q_default 比"加电实际站姿"更屈约 0.07 rad，仿真起始姿态与实机不一致。

本脚本做的事
------------
起立 → 加电保持 → 采样（逐关节中值）→ 打印与旧指令的偏差 + 换算到仿真系的偏差
→ 让你输入量到的身高 → 可选换一档 kp 再测 → 趴下失能 → 保存 JSON。

产物：deploy/stance_powered_calibration.json（含每档 kp 的角度/身高，以及
建议的新 q_default_real / q_default_sim）

用法（NUC，先让狗趴好）
----------------------
  sudo python3 scripts/calibrate_powered_stance.py                  # 只测 kp=150
  sudo python3 scripts/calibrate_powered_stance.py --kp 60          # 只测 kp=60
  sudo python3 scripts/calibrate_powered_stance.py --sweep 150,60   # 一次测两档

安全
----
  · 起立沿用 stand_up.py 的 10s 余弦平滑 + kp 3s 缓升
  · 全程 20Hz 持续下发 MIT 帧（DM 反馈是请求-应答式，不能停发）
  · 每个阶段都等你按键，不会自己往下走
  · Ctrl+C = 立即失能（狗会摔，仅应急用）
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy.motor_client import MotorBus  # noqa: E402

JOINTS = [(ch, mid) for ch in (0, 1, 2, 3) for mid in (0x01, 0x02, 0x03)]
CH_NAMES = {0: "FL", 1: "FR", 2: "RL", 3: "RR"}
MID_NAMES = {0x01: "hip", 0x02: "thigh", 0x03: "calf"}
JOINT_NAMES = [f"{CH_NAMES[c]}_{MID_NAMES[m]}" for c, m in JOINTS]

# 实机→仿真符号矩阵（2026-09-26 实测，顺序 FL,FR,RL,RR 各 髋/大腿/小腿）
S = [+1, +1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1]
FRONT_CALF = (2, 5)
REAR_CALF = (8, 11)
Q_DEFAULT_SIM = [0.0, 0.9, -1.8] * 4      # 仿真当前写死的 q_default
CALF_LEVER = 0.188                         # dz/dq_calf（m/rad），估算身高变化用

LOOP_HZ = 20.0
RISE_SEC = 10.0
KP_RAMP_SEC = 3.0
STABILIZE_SEC = 2.0
SAMPLE_SEC = 2.0

_line_q: "queue.Queue[str]" = queue.Queue()


# ---------------------------------------------------------------- 交互
def _stdin_watcher() -> None:
    """后台线程：读 stdin 一行 → 入队（主循环不能阻塞在 input 上）"""
    while True:
        try:
            line = input()
        except EOFError:
            return
        _line_q.put(line)


def drain_lines() -> list:
    out = []
    while True:
        try:
            out.append(_line_q.get_nowait())
        except queue.Empty:
            return out


# ---------------------------------------------------------------- 姿态工具
def load_pose(name: str) -> list:
    p = REPO_ROOT / "deploy" / f"{name}_calibration.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 不存在。请先跑 `python3 scripts/calibrate_stance.py {name}`，"
            f"或从已标定的机器拷 deploy/ 目录")
    data = json.loads(p.read_text(encoding="utf-8"))
    q = data.get("q_default_real", [])
    if len(q) != 12:
        raise ValueError(f"{p} 格式不对（需要 12 维 q_default_real）")
    return [float(x) for x in q]


def load_trim() -> tuple:
    p = REPO_ROOT / "deploy" / "posture_trim.json"
    if not p.exists():
        return 0.0, 0.0
    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data.get("front", 0.0)), float(data.get("rear", 0.0))


def smooth(t: float, T: float) -> float:
    x = min(max(t / T, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * x))


def blend(a: list, b: list, s: float) -> list:
    return [a[i] + (b[i] - a[i]) * s for i in range(12)]


def apply_trim(q_base: list, trim_f: float, trim_r: float) -> list:
    """在线微调（仿真系 rad）经 S 换算到实机系；正=腿抬高=小腿伸直=仿真角减"""
    q = list(q_base)
    for i in FRONT_CALF:
        q[i] = q_base[i] + S[i] * (-trim_f)
    for i in REAR_CALF:
        q[i] = q_base[i] + S[i] * (-trim_r)
    return q


def to_sim_delta(q_meas: list, q_cmd: list) -> list:
    """实机系偏差 → 仿真系偏差（逐关节乘 S）"""
    return [S[i] * (q_meas[i] - q_cmd[i]) for i in range(12)]


def parse_int_or_none(line: str):
    line = (line or "").strip().replace("mm", "").replace("MM", "")
    if not line:
        return None
    try:
        return float(line)
    except ValueError:
        return None


# ---------------------------------------------------------------- 主逻辑
class Runner:
    def __init__(self, bus: MotorBus, q_lie: list, q_stand: list,
                 trim_f: float, trim_r: float, kd: float):
        self.bus = bus
        self.q_lie = q_lie
        self.q_stand = q_stand
        self.q_hold = apply_trim(q_stand, trim_f, trim_r)
        self.trim_f, self.trim_r = trim_f, trim_r
        self.kd = kd
        self.records: list = []

    def send_all(self, q_cmd: list, kp: float) -> None:
        kd = self.kd if kp > 0 else 0.0
        for ch, mid in JOINTS:
            self.bus.send_mit(ch, mid, q_cmd[ch * 3 + (mid - 1)], 0.0, kp, kd, 0.0)

    def read_all(self) -> list:
        vals = []
        for ch, mid in JOINTS:
            fb = self.bus.get_feedback(ch, mid, max_age=0.3)
            vals.append(fb["pos"] if fb and "pos" in fb else float("nan"))
        return vals

    # 保持 + 稳定 + 采样（内部自带节拍，调用期间主状态机暂停）
    def hold_and_sample(self, kp: float, q_cmd: list, label: str) -> list:
        samples: list = [[] for _ in range(12)]
        t0 = time.time()
        next_t = t0
        while True:
            now = time.time()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue
            next_t = max(next_t + 1.0 / LOOP_HZ, now)
            self.send_all(q_cmd, kp)
            el = now - t0
            if el >= STABILIZE_SEC:
                for i, v in enumerate(self.read_all()):
                    if not math.isnan(v):
                        samples[i].append(v)
            if el >= STABILIZE_SEC + SAMPLE_SEC:
                break
        med = []
        for vs in samples:
            vs = sorted(vs)
            med.append(vs[len(vs) // 2] if vs else float("nan"))
        n = len(samples[0])
        print(f"  [{label}] 采样完成：每关节 {n} 帧")
        return med

    def report(self, kp: float, q_meas: list, q_cmd: list) -> bool:
        """打印偏差表；返回是否全部有反馈"""
        ok = not any(math.isnan(v) for v in q_meas)
        d_sim = to_sim_delta(q_meas, q_cmd) if ok else [float("nan")] * 12
        print("\n" + "-" * 76)
        print(f" kp={kp:.0f}   实测(实机系)      指令            偏差(实机)  偏差(仿真系)")
        print("-" * 76)
        for i, name in enumerate(JOINT_NAMES):
            print(f"  {name:<9s} {q_meas[i]:+9.4f}   {q_cmd[i]:+9.4f}   "
                  f"{q_meas[i] - q_cmd[i]:+9.4f}   {d_sim[i]:+9.4f}")
        print("-" * 76)
        if not ok:
            print("  ⚠ 有电机无反馈（显示 nan），本次结果不可用于写入代码")
            return False
        calfs = [d_sim[i] for i in (2, 5, 8, 11)]
        mean_calf = sum(calfs) / 4
        print("  四条小腿(仿真系): " + "  ".join(f"{v:+.3f}" for v in calfs))
        print(f"  平均 {mean_calf:+.4f} rad → 估算身高变化 "
              f"{mean_calf * CALF_LEVER * 1000:+.0f} mm（正 = 比指令站得高）")
        return True

    def save(self) -> Path | None:
        good = [r for r in self.records if r.get("ok")]
        if not good:
            print("\n⚠ 没有任何一档拿到完整反馈，不保存文件")
            return None
        out = REPO_ROOT / "deploy" / "stance_powered_calibration.json"
        first = good[0]
        d_sim = to_sim_delta(first["q_measured"], first["q_commanded"])
        payload = {
            "note": "加电站立姿态标定（由 scripts/calibrate_powered_stance.py 生成）。"
                    "用于替代 calibrate_stance.py 的手扶标定结果。",
            "unit": "rad（实机系）",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "s_matrix": S,
            "q_default_sim_old": Q_DEFAULT_SIM,
            "trim_used": {"front": self.trim_f, "rear": self.trim_r},
            "records": self.records,
            "q_default_real_powered": [round(v, 4) for v in first["q_measured"]],
            "q_default_sim_powered": [round(Q_DEFAULT_SIM[i] + d_sim[i], 4)
                                      for i in range(12)],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\n已保存：{out}")
        print("  q_default_real_powered =", payload["q_default_real_powered"])
        print("  q_default_sim_powered  =", payload["q_default_sim_powered"])
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="加电站立姿态标定")
    ap.add_argument("--kp", type=float, default=150.0, help="标定 kp（默认 150）")
    ap.add_argument("--kd", type=float, default=2.0, help="标定 kd（默认 2）")
    ap.add_argument("--sweep", type=str, default=None,
                    help='kp 序列，如 "150,60"（先到第一个，再逐档切）')
    ap.add_argument("--rise", type=float, default=RISE_SEC, help="起立/趴下时长（秒）")
    args = ap.parse_args()

    if args.sweep:
        kps = [float(t) for t in args.sweep.replace("，", ",").split(",") if t.strip()]
    else:
        kps = [args.kp]
    if not kps:
        sys.exit("kp 序列为空")

    try:
        q_lie = load_pose("lie")
        q_stand = load_pose("stance")
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)
    trim_f0, trim_r0 = load_trim()

    threading.Thread(target=_stdin_watcher, daemon=True).start()
    bus = MotorBus()
    runner = Runner(bus, q_lie, q_stand, trim_f0, trim_r0, args.kd)

    print("=" * 76)
    print(" 加电站立姿态标定（powered stance）")
    print("=" * 76)
    print(f" 起立 {args.rise:.0f}s，kd={args.kd:.1f}，kp 序列 = {kps}")
    if trim_f0 or trim_r0:
        print(f" 沿用已固化微调：前={trim_f0:+.3f} 后={trim_r0:+.3f}")
    print(" 流程：Enter 起立 → 松手 → 自动采样 → 输入身高(mm) → 下一档/趴下")
    print(" ⚠ 站立时 Ctrl+C 会立即失能，狗会摔，仅应急用")
    print("\n请让狗趴好、周围无人、手放在急停旁，按 Enter 起立…")

    kp = kps[0]
    idx_kp = 0
    kp_now = 0.0
    kp_from = 0.0
    q_cmd = list(q_lie)
    q_from = list(q_lie)
    phase, t_phase = "WAIT_RISE", 0.0
    t_last = time.time()

    try:
        bus.enable_channels([0, 1, 2, 3])
        for ch, mid in JOINTS:
            bus.enable_motor(ch, mid)
            time.sleep(0.02)
            bus.send_mit(ch, mid, q_lie[ch * 3 + (mid - 1)], 0.0, 0.0, 0.0, 0.0)

        while True:
            now = time.time()
            dt = now - t_last
            t_last = now
            t_phase += dt
            lines = drain_lines()

            if phase == "WAIT_RISE":
                kp_now, q_cmd = 0.0, list(q_lie)
                if lines:
                    phase, t_phase = "RISE", 0.0
                    print("\n>>> 起立中，请勿触碰狗 <<<")

            elif phase == "RISE":
                s = smooth(t_phase, args.rise)
                q_cmd = blend(q_lie, runner.q_hold, s)
                kp_now = kp * min(t_phase / KP_RAMP_SEC, 1.0)
                if t_phase >= args.rise:
                    phase = "SAMPLE"
                    print(f"\n>>> 已站立（kp={kp:.0f}）。请松手，马上采样 <<<")

            elif phase == "SAMPLE":
                kp_now, q_cmd = kp, runner.q_hold
                q_meas = runner.hold_and_sample(kp, q_cmd, f"kp={kp:.0f}")
                ok = runner.report(kp, q_meas, q_cmd)
                runner.records.append({
                    "kp": kp, "kd": args.kd, "ok": ok,
                    "q_commanded": [round(v, 4) for v in q_cmd],
                    "q_measured": [round(v, 4) for v in q_meas],
                    "height_mm": None,
                })
                print("\n>>> 量身高：地面 → 【身体几何中心】的毫米数（如 320），"
                      "直接回车跳过：")
                phase, t_phase = "ASK_HEIGHT", 0.0

            elif phase == "ASK_HEIGHT":
                kp_now, q_cmd = kp, runner.q_hold
                if lines:
                    h = parse_int_or_none(lines[0])
                    runner.records[-1]["height_mm"] = h
                    print(f"  身高记录：{h if h is not None else '未填'}")
                    idx_kp += 1
                    if idx_kp < len(kps):
                        print(f"\n>>> 下一档 kp={kps[idx_kp]:.0f}：狗会上升/下沉几厘米，"
                              f"请扶稳后按 Enter")
                        phase, t_phase = "WAIT_NEXT", 0.0
                    else:
                        print("\n>>> 测量结束，按 Enter 趴下并失能")
                        phase, t_phase = "WAIT_SIT", 0.0

            elif phase == "WAIT_NEXT":
                kp_now, q_cmd = kp, runner.q_hold
                if lines:
                    kp_from, kp = kp, kps[idx_kp]
                    kp_now, q_from = kp_from, list(q_cmd)
                    phase, t_phase = "KP_RAMP", 0.0
                    print(f"\n>>> kp {kp_from:.0f} → {kp:.0f} 过渡中 <<<")

            elif phase == "KP_RAMP":
                s = min(t_phase / KP_RAMP_SEC, 1.0)
                kp_now = kp_from + (kp - kp_from) * s
                q_cmd = list(q_from)
                if s >= 1.0:
                    phase = "SAMPLE"
                    print(f"\n>>> kp={kp:.0f} 到位，马上采样 <<<")

            elif phase == "WAIT_SIT":
                kp_now, q_cmd = kp, runner.q_hold
                if lines:
                    q_from = list(q_cmd)
                    phase, t_phase = "SIT", 0.0
                    print("\n>>> 趴下中 <<<")

            elif phase == "SIT":
                s = smooth(t_phase, args.rise)
                q_cmd = blend(q_from, q_lie, s)
                kp_now = kp
                if t_phase >= args.rise:
                    phase, t_phase = "LOWER", 0.0

            elif phase == "LOWER":
                r = min(t_phase / 2.0, 1.0)
                kp_now = kp * (1.0 - r)
                q_cmd = list(q_lie)
                if r >= 1.0:
                    print("\n>>> 已趴下，kp=0，保存并失能 <<<")
                    break

            runner.send_all(q_cmd, kp_now)
            time.sleep(1.0 / LOOP_HZ)

        runner.save()
        print("\n 下一步：把 deploy/stance_powered_calibration.json 拷回开发机，")
        print("         在那里跑 python3 scripts/apply_powered_stance.py 看 diff")

    except KeyboardInterrupt:
        print("\n\n!!! 应急失能（狗会直接摔）!!!")
    except Exception:
        import traceback
        print("\n\n!!! 异常退出，堆栈如下 !!!")
        traceback.print_exc()
    finally:
        for ch, mid in JOINTS:
            try:
                bus.disable_motor(ch, mid)
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
