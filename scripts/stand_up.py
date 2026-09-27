#!/usr/bin/env python3
"""stand_up.py —— 趴姿→站姿→趴姿 全流程（单人操作，免扶狗）

流程：
  1. 狗趴在地上，运行脚本，电机使能但 kp=0（腿软态，不动）
  2. 按 Enter：10 秒余弦平滑插值从趴姿升到站姿，kp 同步缓升
  3. 站姿保持阶段可在线微调前后腿高度（作用在小腿关节，~1s 平滑生效）：
       r 0.1   后腿抬高 0.1 rad（输入 r -0.1 则降低）
       f 0.1   前腿抬高 0.1 rad（输入 f -0.1 则降低）
     数值为相对当前微调目标的增量，屏幕上会显示累计微调量
     （启动时自动加载 deploy/posture_trim.json 作为初始微调，没有则为 0）
  4. 按 Enter（空行）：10 秒平滑降回趴姿 → kp 缓降到 0 → 失能退出
  5. Ctrl+C：立即失能（应急，狗直接摔，慎用）

用法（NUC）：
  sudo python3 scripts/stand_up.py [kp] [kd] [rise_sec]
    kp       默认 100（实机实测站立所需，仿真对齐值 60 偏软）
    kd       默认 2
    rise_sec 默认 10（起立/趴下各 10 秒）

前置：需先跑过
  sudo python3 scripts/calibrate_stance.py stance   # 站姿标定
  sudo python3 scripts/calibrate_stance.py lie      # 趴姿标定
"""
from __future__ import annotations

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

# 实机→仿真符号矩阵（2026-09-26 实测，顺序 FL,FR,RL,RR 各 髋/大腿/小腿）
S = [+1, +1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1]
# 小腿关节在 12 维中的索引：FL=2, FR=5, RL=8, RR=11
REAR_CALF = (8, 11)
FRONT_CALF = (2, 5)

KP_FINAL = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
KD_FINAL = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
RISE_SEC = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
LOWER_KP_SEC = 2.0       # 趴下后 kp 缓降到 0 的时长
LOOP_HZ = 20.0
TRIM_ALPHA = 0.05        # 微调低通系数（20Hz 下约 1s 时间常数）

_line_q = queue.Queue()  # stdin 行队列（空行=Enter）


def _stdin_watcher() -> None:
    """后台线程：每按一次 Enter/输入一行，入队"""
    while True:
        try:
            line = input()
        except EOFError:
            return
        _line_q.put(line)


def drain_lines() -> list:
    lines = []
    while True:
        try:
            lines.append(_line_q.get_nowait())
        except queue.Empty:
            break
    return lines


def load_pose(name: str) -> list:
    p = REPO_ROOT / "deploy" / f"{name}_calibration.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    q = data.get("q_default_real", [])
    if len(q) != 12:
        raise ValueError(f"{p} 格式不对（需要 12 维 q_default_real）")
    return q


def load_trim() -> tuple:
    """加载已固化的站姿微调 deploy/posture_trim.json（不存在则返回 0,0）"""
    p = REPO_ROOT / "deploy" / "posture_trim.json"
    if not p.exists():
        return 0.0, 0.0
    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data.get("front", 0.0)), float(data.get("rear", 0.0))


def smooth(t: float, T: float) -> float:
    """余弦平滑 0→1，两端速度为零"""
    x = min(max(t / T, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * x))


def blend(a: list, b: list, s: float) -> list:
    return [a[i] + (b[i] - a[i]) * s for i in range(12)]


def apply_trim(q_base: list, trim_f: float, trim_r: float) -> list:
    """微调（仿真系 rad）经 S 换算到实机系，正=腿抬高=小腿伸直=仿真角减"""
    q = list(q_base)
    for i in FRONT_CALF:
        q[i] = q_base[i] + S[i] * (-trim_f)
    for i in REAR_CALF:
        q[i] = q_base[i] + S[i] * (-trim_r)
    return q


def main() -> None:
    try:
        q_lie = load_pose("lie")
        q_stand = load_pose("stance")
    except Exception as e:
        print(f"错误：标定文件缺失 —— {e}")
        print("请先跑：calibrate_stance.py stance 和 calibrate_stance.py lie")
        sys.exit(1)

    trim_f0, trim_r0 = load_trim()

    threading.Thread(target=_stdin_watcher, daemon=True).start()

    bus = MotorBus()
    phase = "WAIT"         # WAIT → RISE → STAND → SIT → LOWER → 退出
    t_phase = 0.0
    q_cmd = list(q_lie)
    q_sit_start = list(q_stand)
    kp, kd = 0.0, 0.0
    trim_f, trim_r = trim_f0, trim_r0          # 微调目标（含初始加载）
    trim_f_act, trim_r_act = trim_f0, trim_r0  # 微调平滑值

    try:
        bus.enable_channels([0, 1, 2, 3])
        for ch, mid in JOINTS:
            bus.enable_motor(ch, mid)
            time.sleep(0.02)
            idx = ch * 3 + (mid - 1)
            bus.send_mit(ch, mid, q_lie[idx], 0.0, 0.0, 0.0, 0.0)

        print("=" * 60)
        print(" 趴姿→站姿→趴姿 全流程（站立中可在线微调腿高）")
        print("=" * 60)
        print(f"增益目标 kp={KP_FINAL:.0f} kd={KD_FINAL:.1f}，起立/趴下各 {RISE_SEC:.0f}s")
        if trim_f0 != 0.0 or trim_r0 != 0.0:
            print(f"已加载站姿微调（posture_trim.json）：前={trim_f0:+.3f} 后={trim_r0:+.3f}")
        print("电机已使能（kp=0，腿软）。确认狗趴好后：")
        print("  按 Enter → 起立")
        print("  站立中：r 0.1=后腿抬高 / f 0.1=前腿抬高（负数=降低）")
        print("  按 Enter（空行）→ 趴下并失能")
        print("  ⚠ 站立时 Ctrl+C 会立即失能，狗直接摔，仅应急用")

        step = 0
        t_last = time.time()
        while True:
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            t_phase += dt

            # 微调低通平滑
            trim_f_act += TRIM_ALPHA * (trim_f - trim_f_act)
            trim_r_act += TRIM_ALPHA * (trim_r - trim_r_act)

            lines = drain_lines()

            if phase == "WAIT":
                kp, kd = 0.0, 0.0
                q_cmd = q_lie
                if lines:
                    phase, t_phase = "RISE", 0.0
                    print("\n>>> 起立中，请勿触碰狗 <<<")

            elif phase == "RISE":
                s = smooth(t_phase, RISE_SEC)
                q_cmd = blend(q_lie, apply_trim(q_stand, trim_f_act, trim_r_act), s)
                kp = KP_FINAL * min(t_phase / 3.0, 1.0)   # kp 前 3 秒先到位
                kd = KD_FINAL * min(t_phase / 3.0, 1.0)
                if t_phase >= RISE_SEC:
                    phase, t_phase = "STAND", 0.0
                    print("\n>>> 已站立（kp 满值）")
                    print(">>> 微调：r 0.1=后腿抬高 / f 0.1=前腿抬高；空行 Enter=趴下 <<<")

            elif phase == "STAND":
                kp, kd = KP_FINAL, KD_FINAL
                q_cmd = apply_trim(q_stand, trim_f_act, trim_r_act)
                for line in lines:
                    parts = line.split()
                    if not parts:
                        # 空行 = 趴下
                        q_sit_start = list(q_cmd)
                        phase, t_phase = "SIT", 0.0
                        print("\n>>> 趴下中 <<<")
                        break
                    if len(parts) == 2 and parts[0] in ("r", "f"):
                        try:
                            dv = float(parts[1])
                        except ValueError:
                            print(f"  （无法解析：{line!r}，示例：r 0.1）")
                            continue
                        if parts[0] == "r":
                            trim_r += dv
                        else:
                            trim_f += dv
                        print(f"  微调目标：前={trim_f:+.3f}  后={trim_r:+.3f}")
                    else:
                        print(f"  （无法解析：{line!r}，示例：r 0.1 / f -0.05）")

            elif phase == "SIT":
                s = smooth(t_phase, RISE_SEC)
                q_cmd = blend(q_sit_start, q_lie, s)
                kp, kd = KP_FINAL, KD_FINAL
                if t_phase >= RISE_SEC:
                    phase, t_phase = "LOWER", 0.0

            elif phase == "LOWER":
                r = min(t_phase / LOWER_KP_SEC, 1.0)
                kp = KP_FINAL * (1.0 - r)
                kd = KD_FINAL * (1.0 - r)
                q_cmd = q_lie
                if t_phase >= LOWER_KP_SEC:
                    print("\n>>> 已趴下，kp=0，失能退出 <<<")
                    print(f">>> 本次微调总量：前={trim_f:+.3f}  后={trim_r:+.3f}"
                          f"（含初始加载 前={trim_f0:+.3f} 后={trim_r0:+.3f}；有调整可反馈固化）")
                    break

            for ch, mid in JOINTS:
                idx = ch * 3 + (mid - 1)
                bus.send_mit(ch, mid, q_cmd[idx], 0.0, kp, kd, 0.0)

            # 每 250ms 打印一次误差网格
            if step % 5 == 0:
                errs = []
                for ch, mid in JOINTS:
                    fb = bus.get_feedback(ch, mid, max_age=0.2)
                    idx = ch * 3 + (mid - 1)
                    if fb and "pos" in fb:
                        errs.append(fb["pos"] - q_cmd[idx])
                    else:
                        errs.append(float("nan"))
                print("\n" + "-" * 60)
                print(f" 阶段={phase:5s}  t={t_phase:5.1f}s  kp={kp:5.1f}  kd={kd:3.1f}"
                      f"  微调 前={trim_f_act:+.3f} 后={trim_r_act:+.3f}")
                for ch in (0, 1, 2, 3):
                    vals = []
                    for mid in (0x01, 0x02, 0x03):
                        e = errs[ch * 3 + (mid - 1)]
                        vals.append(f"{MID_NAMES[mid]:6s}={e:+.4f}")
                    print(f"  {CH_NAMES[ch]:3s}  " + "  |  ".join(vals))
                valid = [abs(e) for e in errs if not math.isnan(e)]
                if valid:
                    m = max(valid)
                    print(f" 最大误差={m:.4f} rad ({math.degrees(m):.1f}°)", end="")
                    print("  （起立中误差大属正常，看 STAND 阶段）")

            step += 1
            time.sleep(1.0 / LOOP_HZ)

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
