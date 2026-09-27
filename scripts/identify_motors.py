#!/usr/bin/env python3
"""identify_motors.py —— 手扭电机辨识关节映射

以 kp=0/kd=0（零力矩）使能全部 12 台电机，手可直接扭动关节；
持续轮询反馈，检测到某关节角度偏离基线超过阈值时，
实时打印该关节的中文名称（如"左前(FL)小腿关节"）与当前角度，
用于肉眼核对 (CH, MotorID) ↔ 实际关节 的映射关系。

用法（NUC）：
  sudo python3 scripts/identify_motors.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy.motor_client import MotorBus  # noqa: E402

# 2026-09-25 用户确认的映射（待本脚本验证）
CH_NAMES = {0: "左前(FL)", 1: "右前(FR)", 2: "左后(RL)", 3: "右后(RR)"}
MID_NAMES = {0x01: "髋关节", 0x02: "大腿关节", 0x03: "小腿关节"}
JOINTS = [(ch, mid) for ch in (0, 1, 2, 3) for mid in (0x01, 0x02, 0x03)]

DELTA_TH = 0.03   # rad：偏离基线超过此值判定为"正在被扭动"
POLL_DT = 0.02    # s：相邻两台电机的轮询间隔（整圈约 240ms）
KD_POLL = 0.5     # 轮询用 MIT 帧 kd（扫描已验证 kd=1 能触发回复；0.5 手扭仍轻松）


def joint_name(ch: int, mid: int) -> str:
    return f"{CH_NAMES[ch]}{MID_NAMES[mid]}"


def raw_mode(bus: MotorBus, ch: int, mid: int) -> None:
    """单关节原始帧监视：高速轮询一台电机，单行显示 8 字节原始数据，
    扭动关节时标出变化的字节序号，用于核对反馈帧位布局。"""
    bus.enable_channels([0, 1, 2, 3])
    bus.enable_motor(ch, mid)
    time.sleep(0.05)
    print(f"原始帧监视：CH{ch} ID=0x{mid:02X}（{joint_name(ch, mid)}）")
    print("扭动该关节，观察哪些字节在变化；Ctrl+C 退出\n")
    prev = None
    while True:
        bus.send_mit(ch, mid, 0.0, 0.0, 0.0, KD_POLL, 0.0)
        time.sleep(0.03)
        fb = bus.get_feedback(ch, mid, max_age=0.3)
        if fb is None:
            continue
        raw = fb.get("raw", "")
        mark = ""
        if prev is not None and raw != prev:
            changed = [str(i) for i in range(min(len(raw), 16) // 2)
                       if raw[i * 2:i * 2 + 2] != prev[i * 2:i * 2 + 2]]
            mark = "  变化字节: " + ",".join(changed)
        prev = raw
        print(f"\rRAW[{' '.join(raw[i:i+2] for i in range(0, len(raw), 2))}]"
              f"  pos={fb.get('pos', 0):+9.4f} vel={fb.get('vel', 0):+7.3f}"
              f" tau={fb.get('tau', 0):+6.2f}{mark}   ", end="", flush=True)


def main() -> None:
    bus = MotorBus()
    try:
        # 原始帧监视模式：identify_motors.py raw <ch> <mid>
        if len(sys.argv) >= 4 and sys.argv[1] == "raw":
            raw_mode(bus, int(sys.argv[2]), int(sys.argv[3], 0))
            return
        bus.enable_channels([0, 1, 2, 3])
        # 全部使能 + 轻阻尼 MIT 帧（kd 很小，手扭安全）
        for ch, mid in JOINTS:
            bus.enable_motor(ch, mid)
            time.sleep(0.02)
            bus.send_mit(ch, mid, 0.0, 0.0, 0.0, KD_POLL, 0.0)
        print(f"12 台电机已使能（kp=0/kd={KD_POLL} 轻阻尼，可手扭）")
        print("用手扭动任一关节，屏幕会显示对应名称与角度；Ctrl+C 退出\n")

        baseline: dict = {}
        t_status = 0.0
        grid_printed = False
        while True:
            latest = {}  # (ch, mid) -> pos 本轮最新读数
            online = 0
            for ch, mid in JOINTS:
                # DM 反馈是请求-应答式：每发一帧才回一帧，须持续轮询
                bus.send_mit(ch, mid, 0.0, 0.0, 0.0, KD_POLL, 0.0)
                time.sleep(POLL_DT)
                fb = bus.get_feedback(ch, mid, max_age=0.5)
                if fb is None or "pos" not in fb:
                    continue
                online += 1
                pos = fb["pos"]
                latest[(ch, mid)] = pos
                if (ch, mid) not in baseline:
                    baseline[(ch, mid)] = pos
            # 每 0.5s 原地重画 4 行角度网格（每条腿一行），直接盯数字变化
            if time.time() - t_status > 0.5:
                lines = []
                for ch in (0, 1, 2, 3):
                    cells = []
                    for mid in (0x01, 0x02, 0x03):
                        name = MID_NAMES[mid].replace("关节", "")
                        p = latest.get((ch, mid))
                        if p is not None:
                            d = p - baseline.get((ch, mid), p)
                            cells.append(f"{name} {p:+9.4f}(Δ{d:+.3f})")
                        else:
                            cells.append(f"{name}      --      ")
                    lines.append(f"CH{ch} {CH_NAMES[ch]}: " + "  ".join(cells))
                if grid_printed:
                    sys.stdout.write("\033[4A")  # 光标上移 4 行原地重画
                for ln in lines:
                    sys.stdout.write("\033[2K" + ln + "\n")  # 清行再写，防残留
                sys.stdout.flush()
                grid_printed = True
                t_status = time.time()
    except KeyboardInterrupt:
        print("\n\n退出，失能全部电机…")
    finally:
        # 只失能电机，不调 bus.close()：SDK 关设备时 C 线程会触发
        # libusb 断言崩溃，进程退出后内核会自动回收 USB 设备
        try:
            bus.disable_all(JOINTS)
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
