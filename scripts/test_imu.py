"""DM-IMU-L1 USB 读取测试脚本

用途：在 NUC 或本机用 USB 虚拟串口直读 IMU，验证 4 件事：
    1. 串口能打开、有数据流
    2. 四种数据帧（加速度/角速度/欧拉角/四元数）解析正确
    3. 单位判定：静止时加速度 Z ≈ 9.8 (m/s²) 还是 1.0 (g)
    4. 静止时陀螺仪读数 ≈ 0；顺带测实际输出频率

协议（说明书 V1.2 USB 帧，float 小端序）：
    加速度/角速度/欧拉角帧（19 字节）：55 AA ID TYPE f1[4] f2[4] f3[4] CRC16[2] 0A
    四元数帧（23 字节）：              55 AA ID 04   W[4] X[4] Y[4] Z[4] CRC16[2] 0A
    TYPE：01=加速度 02=角速度 03=欧拉角(Roll,Pitch,Yaw) 04=四元数(W,X,Y,Z)

用法：
    python scripts/test_imu.py --list                # 列出可用串口
    python scripts/test_imu.py --port COM5           # Windows
    python scripts/test_imu.py --port /dev/ttyACM0   # NUC/Linux

依赖：pip install pyserial
Linux 打不开串口报 Permission denied 时：sudo usermod -aG dialout $USER 后重新登录
"""

import argparse
import struct
import time

import serial
from serial.tools import list_ports

HEAD = b"\x55\xaa"
TAIL = 0x0A
FRAME_LEN_3F = 19  # 加速度/角速度/欧拉角帧长
FRAME_LEN_4F = 23  # 四元数帧长
TYPE_ACC, TYPE_GYR, TYPE_EUL, TYPE_QUA = 1, 2, 3, 4


def crc16_xmodem(data: bytes) -> int:
    """CRC-16/XMODEM（poly=0x1021，初值 0），与说明书附录四查表法等价"""
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class ImuParser:
    """字节流滑窗解析：找帧头 → 按类型定帧长 → 校帧尾 → 取 float"""

    def __init__(self):
        self.buf = bytearray()
        self.latest = {}    # type -> (floats 元组)
        self.counts = {}    # type -> 累计帧数
        self.crc_fail = 0

    def feed(self, data: bytes):
        self.buf.extend(data)
        while True:
            i = self.buf.find(HEAD)
            if i < 0:
                del self.buf[:-1]      # 保留 1 字节防帧头被切断
                return
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < 4:
                return
            ftype = self.buf[3]
            if ftype == TYPE_QUA:
                flen = FRAME_LEN_4F
            elif ftype in (TYPE_ACC, TYPE_GYR, TYPE_EUL):
                flen = FRAME_LEN_3F
            else:                      # 伪帧头，丢 1 字节重新同步
                del self.buf[:1]
                continue
            if len(self.buf) < flen:
                return
            frame = bytes(self.buf[:flen])
            if frame[-1] != TAIL:
                del self.buf[:1]
                continue
            # CRC 覆盖 ID..payload（范围按常见约定；失败只计数不丢帧，避免误杀）
            crc_recv = struct.unpack("<H", frame[-3:-1])[0]
            if crc16_xmodem(frame[2:-3]) != crc_recv:
                self.crc_fail += 1
            n = 4 if ftype == TYPE_QUA else 3
            self.latest[ftype] = struct.unpack(f"<{n}f", frame[4:4 + 4 * n])
            self.counts[ftype] = self.counts.get(ftype, 0) + 1
            del self.buf[:flen]


def main():
    ap = argparse.ArgumentParser(description="DM-IMU-L1 USB 读取测试")
    ap.add_argument("--port", help="串口名，如 COM5 或 /dev/ttyACM0")
    ap.add_argument("--list", action="store_true", help="列出可用串口")
    ap.add_argument("--baud", type=int, default=115200, help="波特率（USB CDC 下无意义）")
    args = ap.parse_args()

    if args.list:
        for p in list_ports.comports():
            print(f"{p.device}   {p.description}")
        return
    if not args.port:
        ap.error("需要 --port（先用 --list 查看可用串口）")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"打开 {args.port} 失败：{e}")
        print("Linux 下 Permission denied：sudo usermod -aG dialout $USER 后重新登录")
        return

    parser = ImuParser()
    print(f"已打开 {args.port}，按 Ctrl+C 退出\n")
    t_last, t0 = time.time(), time.time()
    counts_prev = {}

    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                parser.feed(chunk)

            now = time.time()
            if now - t_last < 0.5:
                continue
            dt = now - t_last
            t_last = now

            # 各类型实测频率（帧数差分 / 时间窗）
            hz = {
                t: (parser.counts.get(t, 0) - counts_prev.get(t, 0)) / dt
                for t in (TYPE_ACC, TYPE_GYR, TYPE_EUL, TYPE_QUA)
            }
            counts_prev = dict(parser.counts)

            print(f"── t={now - t0:6.1f}s ──────────────────────────")
            if TYPE_ACC in parser.latest:
                x, y, z = parser.latest[TYPE_ACC]
                unit = ("m/s²" if abs(abs(z) - 9.81) < 1.5
                        else "g" if abs(abs(z) - 1.0) < 0.15 else "?")
                print(f"acc {hz[TYPE_ACC]:7.1f}Hz  x={x:8.3f} y={y:8.3f} "
                      f"z={z:8.3f}   单位判定: {unit}")
            if TYPE_GYR in parser.latest:
                x, y, z = parser.latest[TYPE_GYR]
                mag = (x * x + y * y + z * z) ** 0.5
                print(f"gyr {hz[TYPE_GYR]:7.1f}Hz  x={x:8.3f} y={y:8.3f} "
                      f"z={z:8.3f}   |ω|={mag:.3f}")
            if TYPE_EUL in parser.latest:
                roll, pitch, yaw = parser.latest[TYPE_EUL]
                print(f"eul {hz[TYPE_EUL]:7.1f}Hz  roll={roll:7.2f} "
                      f"pitch={pitch:7.2f} yaw={yaw:7.2f}  (deg?)")
            if TYPE_QUA in parser.latest:
                w, x, y, z = parser.latest[TYPE_QUA]
                norm = (w * w + x * x + y * y + z * z) ** 0.5
                print(f"qua {hz[TYPE_QUA]:7.1f}Hz  w={w:7.3f} x={x:7.3f} "
                      f"y={y:7.3f} z={z:7.3f}  |q|={norm:.3f}")
            if parser.crc_fail:
                print(f"CRC 校验累计失败 {parser.crc_fail} 帧"
                      f"（持续增长则说明 CRC 覆盖范围与约定不符，不影响读数）")
            print()
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        total = sum(parser.counts.values())
        elapsed = time.time() - t0
        print("已退出。总结：")
        print(f"  总帧数 {total}，平均 {total / elapsed:.0f} 帧/s，运行 {elapsed:.0f}s")
        print(f"  各类型帧数 {parser.counts}")
        print()
        print("请依次手动检查并记录：")
        print("  1. 静止时 gyr 各轴 ≈ 0（|ω| < 0.05）")
        print("  2. 抬起狗头：pitch 符号是正还是负（与 MuJoCo 约定比对）")
        print("  3. 向左倾：roll 符号是正还是负")
        print("  4. 手动绕某轴 ~90°/s 旋转：gyr 峰值 ≈1.57 → rad/s；≈90 → °/s")
        print("  5. acc 单位判定栏显示 m/s² 还是 g")


if __name__ == "__main__":
    main()
