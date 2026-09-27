#!/usr/bin/env python3
"""scan_motors.py —— 机器狗全电机 ID 扫描工具（scan_tool C# 版的 Python 移植）

扫描 LinkX-4C 各 CAN 通道（CH0~CH3）上连接的电机，确定 Motor ID。
策略（与 C# 版一致）：
  对每个 (通道, ID=0x01~0x1E)：
    失能 → 使能(等200ms) → 清计数 → 发阻尼 MIT 帧(kd=1, 等200ms)
    → 期间收到反馈帧则该 ID 有电机 → 立即失能

安全性：电机只被短暂使能且 MIT 帧 kp=0/kd=1（纯阻尼），不产生主动运动，
        每台电机扫完立即失能。

用法（NUC）：
  sudo python3 scripts/scan_motors.py            # 扫 CH0~CH3
  sudo python3 scripts/scan_motors.py 0 1        # 只扫 CH0、CH1
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy.motor_client import MotorBus, MASTER_ID_OFFSET  # noqa: E402

ID_MIN = 0x01
ID_MAX = 0x1E


def scan_channel(bus: MotorBus, ch: int,
                 found: List[Tuple[int, int, int]]) -> None:
    print(f"-------- 扫描 CH{ch} --------")
    channel_has_motor = False

    for mid in range(ID_MIN, ID_MAX + 1):
        # 1) 失能（清上次状态）
        bus.disable_motor(ch, mid)
        time.sleep(0.02)

        # 2) 使能
        bus.enable_motor(ch, mid)
        time.sleep(0.20)

        # 3) 使能后的反馈检查
        fb_enable = bus.get_feedback(ch, mid, max_age=0.25)

        # 4) 发阻尼 MIT 帧（kp=0, kd=1）触发反馈
        bus.send_mit(ch, mid, p_des=0.0, v_des=0.0, kp=0.0, kd=1.0, tau=0.0)
        time.sleep(0.20)
        fb_mit = bus.get_feedback(ch, mid, max_age=0.25)

        # 5) 判定
        if fb_mit is not None or fb_enable is not None:
            master_id = mid + MASTER_ID_OFFSET
            # 优先展示解码出的实时位置（核对反馈解码是否正确的关键证据）
            fb = fb_mit or fb_enable
            pos_str = f"pos={fb['pos']:+.3f}rad" if "pos" in fb else f"raw={fb.get('raw')}"
            print(f"  CH{ch}  ID=0x{mid:02X} ({mid:2d})  ✓ 发现电机  "
                  f"Master=0x{master_id:02X}  {pos_str}")
            found.append((ch, mid, master_id))
            channel_has_motor = True

        # 6) 立即失能，防止发热/误动
        bus.disable_motor(ch, mid)
        time.sleep(0.02)

    if not channel_has_motor:
        print(f"  (CH{ch} 未发现任何电机)")
    print()


def main() -> int:
    channels = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3]

    print("========================================")
    print(" 机器狗全电机 ID 扫描工具 (Python)")
    print("========================================")
    print(f" 扫描通道：{channels}")
    print(f" Motor ID 范围：0x{ID_MIN:02X} ~ 0x{ID_MAX:02X}")
    print(" 策略：失能 → 使能 → 阻尼MIT → 查反馈")
    print("=======================================\n")

    bus = MotorBus()
    found: List[Tuple[int, int, int]] = []
    try:
        bus.enable_channels(channels)
        print(f"[通道已开启] {channels}\n")

        t0 = time.time()
        for ch in channels:
            scan_channel(bus, ch, found)

        print("========================================")
        print(" 扫描结果汇总")
        print("========================================")
        if not found:
            print("未发现任何电机！\n可能原因：")
            print("  1. CAN 线未接好 / 接反")
            print("  2. 电机电源未开")
            print("  3. CANFD 模式未关闭（用 LinkX Configurator 关）")
            print("  4. 波特率不是 1Mbps（用 LinkX Configurator 设）")
            print("  5. 电机 ID 超过 0x1E")
            return 1

        print(f"共发现 {len(found)} 台电机，耗时 {time.time()-t0:.0f}s：\n")
        print("  通道  Motor ID  Master ID")
        print("  ----  --------  ---------")
        for ch, mid, master in found:
            print(f"  CH{ch}    0x{mid:02X} ({mid:2d})    0x{master:02X} ({master:2d})")
        print("\n说明：控制帧发 Motor ID；反馈帧 ID = Master ID = Motor ID + 0x10")
        print("      若发现 12 台且 pos 读数与肉眼姿态吻合，反馈解码即正确。")
        return 0
    finally:
        # 兜底：把扫描过的所有 ID 都失能一遍
        try:
            bus.disable_all([(ch, mid) for ch in channels
                             for mid in range(ID_MIN, ID_MAX + 1)])
            bus.close()
        except Exception:
            pass
        # 一次性 CLI 工具：C SDK 的析构线程在关闭 USB 时可能触发断言崩溃。
        # 扫描结果已确认正确，flush 后直接 os._exit(0) 跳过 Python/C 清理。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
