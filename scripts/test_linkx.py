"""test_linkx.py —— LinkX-4C / USB2CANFD 连接与总线监听测试（NUC Linux）

用途：
  1. 确认 dmcan SDK 能找到并打开 USB 设备（libusb 通道、权限）
  2. 探测设备实际通道数（LinkX4C = 4 通道 ch0~3）
  3. 监听 CAN 总线上的帧，按 (通道, CAN_ID) 统计 —— 上电后电机若主动上报，
     可直接看到 12 个电机的反馈 ID；什么都没有也正常（电机未使能/未配置主动上报）

用法（NUC）：
  python3 scripts/test_linkx.py                     # 默认找所有设备，监听 5 秒
  python3 scripts/test_linkx.py --type linkx4c      # 仅找 LinkX4C
  python3 scripts/test_linkx.py --seconds 10
  权限不足时先看脚本末尾排查清单，或临时 sudo 运行。
"""
import argparse
import sys
import threading
import time
from pathlib import Path

# 让脚本能 import 仓库内 vendor 的 deploy/dmcan
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "deploy"))

from dmcan import DmCanContext, dmcan_device_type  # noqa: E402


TYPE_MAP = {
    "all": None,
    "linkx4c": dmcan_device_type.LinkX4C,
    "single": dmcan_device_type.USB2CANFD,
    "dual": dmcan_device_type.USB2CANFD_DUAL,
}


class FrameStats:
    """接收线程里只做计数，主线程定时读取打印"""

    def __init__(self):
        self._lock = threading.Lock()
        self.total = 0
        self.per_channel = {}          # ch -> count
        self.per_id = {}               # (ch, id) -> count
        self.last_ts = None
        self.errors = 0

    def on_recv(self, _dev, frame):
        h = frame.head
        with self._lock:
            self.total += 1
            self.per_channel[h.channel] = self.per_channel.get(h.channel, 0) + 1
            key = (h.channel, h.can_id)
            self.per_id[key] = self.per_id.get(key, 0) + 1
            self.last_ts = h.timestamp

    def on_error(self, _dev, _frame):
        with self._lock:
            self.errors += 1

    def snapshot(self):
        with self._lock:
            return self.total, dict(self.per_channel), dict(self.per_id), self.errors


def main():
    ap = argparse.ArgumentParser(description="LinkX-4C 连接与 CAN 监听测试")
    ap.add_argument("--type", choices=list(TYPE_MAP), default="all", help="设备类型过滤")
    ap.add_argument("--device-index", type=int, default=0, help="使用第几个设备")
    ap.add_argument("--seconds", type=float, default=5.0, help="监听时长（秒）")
    args = ap.parse_args()

    stats = FrameStats()

    # 1. 上下文 + 查找设备
    context = DmCanContext()
    print("== SDK 后端版本 ==")
    context.print_version()

    dev_type = TYPE_MAP[args.type]
    count = context.find_devices(dev_type)
    print(f"\n== find_devices({args.type}) -> 找到 {count} 个设备 ==")
    context.show_all_devices()

    if count == 0:
        print("\n[!] 未找到设备，排查清单：")
        print("    1) USB 线是否插好、LinkX-4C 是否上电：lsusb 能否看到设备（拔掉对比）")
        print("    2) libusb 是否安装：sudo apt install libusb-1.0-0")
        print("    3) USB 权限：sudo 运行本脚本试一次；或配置 udev 规则（见脚本头注释/对话说明）")
        print("    4) 设备被别的程序占用（上位机/其他脚本），先关闭")
        return 1

    # 2. 打开设备
    device = context.get_device(args.device_index)
    if not device.open():
        print("[!] 设备打开失败（常见原因：被占用 / 权限不足，试试 sudo）")
        return 1
    print("\n== 设备打开成功，固件版本 ==")
    device.print_version()

    # 3. 探测通道（LinkX4C 标称 0~3；以 get_channel_baudrate 成功为准）
    enabled = []
    for ch in range(4):
        device.enable_channel(ch, True)
        info = device.get_channel_baudrate(ch)
        if info is not None:
            enabled.append(ch)
            print(f"  ch{ch}: 存在 | canfd={bool(info.canfd)} "
                  f"arb={info.can_baudrate} data={info.canfd_baudrate} "
                  f"sp={info.can_sp:.2f}")
    print(f"== 实际可用通道：{enabled} ==")

    # 4. 注册回调并监听
    device.hook_recv_callback(stats.on_recv)
    device.hook_err_callback(stats.on_error)

    print(f"\n== 开始监听 {args.seconds}s（回调计数，不打印原始帧）==")
    t0 = time.time()
    last_total = 0
    while time.time() - t0 < args.seconds:
        time.sleep(0.5)
        total, per_ch, _, errors = stats.snapshot()
        rate = (total - last_total) / 0.5
        last_total = total
        print(f"  t={time.time()-t0:4.1f}s 累计帧={total:6d} "
              f"近0.5s速率={rate:7.0f} 帧/s 各通道={per_ch} err={errors}")

    # 5. 汇总
    total, per_ch, per_id, errors = stats.snapshot()
    print("\n== 监听汇总 ==")
    print(f"  总帧数: {total}   总线错误回调: {errors}")
    print(f"  各通道帧数: {per_ch}")
    if per_id:
        print("  按 (通道, CAN_ID) 统计（ID 为十进制，hex 见括号）：")
        for (ch, can_id), n in sorted(per_id.items(), key=lambda kv: -kv[1]):
            print(f"    ch{ch}  ID={can_id:4d} (0x{can_id:03X})  x {n}")
    else:
        print("  没收到任何帧 —— 电机未使能/未主动上报时这是正常的；")
        print("  若总线上确有设备在发帧，检查波特率是否一致、终端电阻 120Ω。")

    # 6. 清理
    for ch in enabled:
        device.enable_channel(ch, False)
    device.close()
    context.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
