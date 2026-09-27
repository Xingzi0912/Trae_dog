#!/usr/bin/env python3
"""calibrate_stance.py —— 实机站立位标定

用手把狗摆成标准站立姿势（四足着地，机身水平，髋/大腿/小腿自然伸直），
然后按 Enter，脚本读取 12 台电机当前角度，输出 q_default_real 数组。

用法（NUC，先让狗摆好姿势再执行）：
  sudo python3 scripts/calibrate_stance.py [名称]
    名称默认 stance（站姿），可指定 lie（趴姿）等，
    输出到 deploy/{名称}_calibration.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy.motor_client import MotorBus  # noqa: E402

CALIB_NAME = sys.argv[1] if len(sys.argv) > 1 else "stance"

JOINTS = [(ch, mid) for ch in (0, 1, 2, 3) for mid in (0x01, 0x02, 0x03)]
CH_NAMES = {0: "FL", 1: "FR", 2: "RL", 3: "RR"}
MID_NAMES = {0x01: "hip", 0x02: "thigh", 0x03: "calf"}


def main() -> None:
    bus = MotorBus()
    try:
        bus.enable_channels([0, 1, 2, 3])
        # 先全部使能+零力矩，方便手调整姿态
        for ch, mid in JOINTS:
            bus.enable_motor(ch, mid)
            time.sleep(0.02)
            bus.send_mit(ch, mid, 0.0, 0.0, 0.0, 0.0, 0.0)

        print("=" * 50)
        print(f" 实机{CALIB_NAME}位标定")
        print("=" * 50)
        print("当前电机处于 kp=0/kd=0 零力矩模式，可手调姿态。")
        print(f"请把狗摆成「{CALIB_NAME}」姿势（趴姿用于起立脚本，站姿用于部署）。")
        input("\n摆好后按 Enter 读取角度…")

        # 快速轮询 5 轮取中值，滤掉单帧野值
        readings: dict = {j: [] for j in JOINTS}
        for _ in range(5):
            for ch, mid in JOINTS:
                bus.send_mit(ch, mid, 0.0, 0.0, 0.0, 0.0, 0.0)
                time.sleep(0.02)
                fb = bus.get_feedback(ch, mid, max_age=0.3)
                if fb and "pos" in fb:
                    readings[(ch, mid)].append(fb["pos"])
            time.sleep(0.05)

        q_default = []
        missing = []
        print("\n标定结果（弧度）：")
        print("-" * 50)
        for ch, mid in JOINTS:
            vals = readings[(ch, mid)]
            if not vals:
                print(f"  CH{ch} ID=0x{mid:02X}  —— 无反馈！")
                missing.append((ch, mid))
                q_default.append(0.0)
                continue
            # 取中值（3 个数的中位数）
            vals_sorted = sorted(vals)
            med = vals_sorted[len(vals_sorted) // 2]
            q_default.append(med)
            print(f"  {CH_NAMES[ch]}_{MID_NAMES[mid]:6s}  "
                  f"(CH{ch} ID=0x{mid:02X})  = {med:+.4f} rad  "
                  f"(raw={vals})")

        # 按策略 12 维顺序输出：FL, FR, RL, RR（每腿 hip/thigh/calf）
        arr = [round(x, 4) for x in q_default]
        print("\n" + "=" * 50)
        print(" q_default_real（策略 12 维顺序，可直接贴到部署代码）：")
        print("-" * 50)
        print(f"  FL_hip, FL_thigh, FL_calf   = {arr[0]:+.4f}, {arr[1]:+.4f}, {arr[2]:+.4f}")
        print(f"  FR_hip, FR_thigh, FR_calf   = {arr[3]:+.4f}, {arr[4]:+.4f}, {arr[5]:+.4f}")
        print(f"  RL_hip, RL_thigh, RL_calf   = {arr[6]:+.4f}, {arr[7]:+.4f}, {arr[8]:+.4f}")
        print(f"  RR_hip, RR_thigh, RR_calf   = {arr[9]:+.4f}, {arr[10]:+.4f}, {arr[11]:+.4f}")
        print("\n JSON 数组（一键复制）：")
        print(json.dumps(arr, ensure_ascii=False))

        if missing:
            print("\n" + "!" * 50)
            print("!!! 有电机无反馈，本次标定结果不可用于部署，不保存文件 !!!")
            for ch, mid in missing:
                print(f"    缺失：{CH_NAMES[ch]}_{MID_NAMES[mid]} (CH{ch} ID=0x{mid:02X})")
            print("请检查对应电机接线/CAN 通道后重新运行本脚本。")
            sys.exit(1)

        # 同时保存到文件
        out = REPO_ROOT / "deploy" / f"{CALIB_NAME}_calibration.json"
        out.write_text(json.dumps({
            "q_default_real": arr,
            "unit": "rad",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": f"{CALIB_NAME}位标定，kp=0/kd=0 手调姿态后采样"
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已保存到 {out}")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        # 只失能电机，不调 bus.close()：SDK 关设备时 C 线程会触发
        # libusb 断言崩溃，进程退出后内核会自动回收 USB 设备
        try:
            for ch, mid in JOINTS:
                try:
                    bus.disable_motor(ch, mid)
                except Exception:
                    pass
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
