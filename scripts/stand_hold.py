#!/usr/bin/env python3
"""stand_hold.py —— 零策略站立保持测试

用法（NUC）：
  1. 先用手把狗摆成标准站立姿势（和 calibrate_stance 时一样）
  2. sudo python3 scripts/stand_hold.py [kp] [kd]     # 默认 kp=60 kd=2
  3. 按 Enter 开始 10 秒 kp 缓升（0→目标值），此过程手继续轻扶
  4. 缓升结束后松手，观察狗能否自主站稳
  5. 按 Ctrl+C 退出，电机失能

安全：10 秒缓升避免从 kp=0 软态直接切到高 kp 产生冲击。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy.motor_client import MotorBus  # noqa: E402

JOINTS = [(ch, mid) for ch in (0, 1, 2, 3) for mid in (0x01, 0x02, 0x03)]
CH_NAMES = {0: "FL", 1: "FR", 2: "RL", 3: "RR"}
MID_NAMES = {0x01: "hip", 0x02: "thigh", 0x03: "calf"}

KP_FINAL = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
KD_FINAL = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
RAMP_SEC = 10.0          # 缓升时长
LOOP_HZ = 20.0           # 控制循环 20Hz（50ms）


def load_q_default() -> list[float]:
    calib_path = REPO_ROOT / "deploy" / "stance_calibration.json"
    if calib_path.exists():
        data = json.loads(calib_path.read_text(encoding="utf-8"))
        return data.get("q_default_real", [])
    return []


def main() -> None:
    q_default = load_q_default()
    if len(q_default) != 12:
        print("错误：deploy/stance_calibration.json 缺失或格式不对，请先跑 calibrate_stance.py")
        sys.exit(1)

    bus = MotorBus()
    try:
        bus.enable_channels([0, 1, 2, 3])
        for ch, mid in JOINTS:
            bus.enable_motor(ch, mid)
            time.sleep(0.02)
            bus.send_mit(ch, mid, q_default[ch * 3 + (mid - 1)], 0.0, 0.0, 0.0, 0.0)

        print("=" * 60)
        print(" 零策略站立保持测试（kp 10 秒缓升）")
        print("=" * 60)
        print(f"目标增益：kp={KP_FINAL:.0f}  kd={KD_FINAL:.1f}")
        print("当前电机 kp=0/kd=0，请用手把狗摆成标准站立姿势。")
        input("\n摆好后按 Enter 开始 10 秒缓升（期间继续轻扶）…")

        start_t = time.time()
        step = 0
        while True:
            t = time.time() - start_t
            if t < RAMP_SEC:
                ratio = t / RAMP_SEC
                kp = KP_FINAL * ratio
                kd = KD_FINAL * ratio
                phase = "RAMP"
            else:
                kp = KP_FINAL
                kd = KD_FINAL
                phase = "HOLD"

            # 下发目标角度，缓升期间 kp/kd 在涨
            for ch, mid in JOINTS:
                idx = ch * 3 + (mid - 1)
                bus.send_mit(ch, mid, q_default[idx], 0.0, kp, kd, 0.0)

            # 读反馈并打印误差
            errs: list[float] = []
            for ch, mid in JOINTS:
                fb = bus.get_feedback(ch, mid, max_age=0.2)
                idx = ch * 3 + (mid - 1)
                target = q_default[idx]
                if fb and "pos" in fb:
                    err = fb["pos"] - target
                    errs.append(err)
                else:
                    errs.append(float("nan"))

            # 格式化打印：4 行 × 3 关节
            if step % 5 == 0:  # 每 5 步（250ms）刷新一次屏幕
                print("\n" + "=" * 60)
                print(f" 阶段={phase}  时间={t:5.1f}s  kp={kp:5.1f}  kd={kd:4.1f}")
                print("-" * 60)
                for ch in (0, 1, 2, 3):
                    vals = []
                    for mid in (0x01, 0x02, 0x03):
                        idx = ch * 3 + (mid - 1)
                        e = errs[idx]
                        vals.append(f"{MID_NAMES[mid]:6s}={e:+.4f}")
                    print(f"  {CH_NAMES[ch]:3s}  " + "  |  ".join(vals))
                max_err_vals = [abs(e) for e in errs if not math.isnan(e)]
                if max_err_vals:
                    max_err = max(max_err_vals)
                    print(f"\n  最大关节误差 = {max_err:.4f} rad  ({math.degrees(max_err):.1f}°)")
                else:
                    print(f"\n  （暂无任何电机反馈，feedback_count={bus.feedback_count()}）")

            step += 1
            time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        print("\n\n用户中断，正在失能电机…")
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
