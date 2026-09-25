#!/usr/bin/env python3
"""DM MIT 模式电机客户端（协议以 robot-dog/scan_tool C# 代码为准移植）

提供两层：
  - MIT 帧编解码（纯函数，无硬件依赖，可单测）
  - MotorBus：基于 vendor 的 dmcan SDK 操作 LinkX-4C，
    支持使能/失能、MIT 指令下发、反馈帧接收

ID 约定（DM MIT）：
  控制帧发往 Motor ID（0x01~0x1E）
  反馈帧 CAN ID = Master ID = Motor ID + 0x10
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# J8009-2EC MIT 参数范围（与 scan_tool Program.cs 完全一致）
P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -45.0, 45.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0
T_MIN, T_MAX = -54.0, 54.0

# 使能/失能帧（发往 Motor ID）
ENABLE_FRAME = bytes([0xFF] * 7 + [0xFC])
DISABLE_FRAME = bytes([0xFF] * 7 + [0xFD])

MASTER_ID_OFFSET = 0x10


def float_to_uint(value: float, lo: float, hi: float, bits: int) -> int:
    """浮点数 → 无符号整数（MIT 编码，自动截断到量程）"""
    value = min(max(value, lo), hi)
    span = hi - lo
    max_int = (1 << bits) - 1
    return int((value - lo) / span * max_int)


def uint_to_float(uint: int, lo: float, hi: float, bits: int) -> float:
    """无符号整数 → 浮点数（反馈帧解码用）"""
    span = hi - lo
    max_int = (1 << bits) - 1
    return uint / max_int * span + lo


def pack_mit(p_des: float, v_des: float, kp: float, kd: float,
             tau: float) -> bytes:
    """打包 MIT 控制帧（8 字节：p16 | v12 | kp12 | kd12 | t12 = 64bit）"""
    p_uint = float_to_uint(p_des, P_MIN, P_MAX, 16)
    v_uint = float_to_uint(v_des, V_MIN, V_MAX, 12)
    kp_uint = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    kd_uint = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    t_uint = float_to_uint(tau, T_MIN, T_MAX, 12)

    data = bytearray(8)
    data[0] = p_uint >> 8
    data[1] = p_uint & 0xFF
    data[2] = v_uint >> 4
    data[3] = ((v_uint & 0x0F) << 4) | (kp_uint >> 8)
    data[4] = kp_uint & 0xFF
    data[5] = kd_uint >> 4
    data[6] = ((kd_uint & 0x0F) << 4) | (t_uint >> 8)
    data[7] = t_uint & 0xFF
    return bytes(data)


def unpack_mit_reply(payload: bytes) -> Dict[str, float]:
    """解码电机反馈帧（标准 MIT reply：p16 | v12 | t12）

    待实机首帧核对：DM 反馈帧与 Unitree MIT reply 同构，
    若实测解码数值量级不对，以抓到的原始帧为准调整位布局。
    """
    p_uint = (payload[0] << 8) | payload[1]
    v_uint = (payload[2] << 4) | (payload[3] >> 4)
    t_uint = ((payload[3] & 0x0F) << 8) | payload[4]
    return {
        "pos": uint_to_float(p_uint, P_MIN, P_MAX, 16),
        "vel": uint_to_float(v_uint, V_MIN, V_MAX, 12),
        "tau": uint_to_float(t_uint, T_MIN, T_MAX, 12),
    }


class MotorBus:
    """LinkX-4C 总线：多通道多电机的使能/失能/指令/反馈"""

    def __init__(self, device_index: int = 0):
        # vendor dmcan 包在 deploy/dmcan
        deploy_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(deploy_dir))
        from dmcan import DmCanContext  # noqa: E402

        self._context = DmCanContext()
        self._context.print_version()
        cnt = self._context.find_devices()
        if cnt == 0:
            raise RuntimeError("未找到任何 DM USB 设备（检查 USB / libusb / 权限 / 是否被占用）")
        self._context.show_all_devices()
        self.device = self._context.get_device(device_index)
        if not self.device.open():
            raise RuntimeError("设备打开失败（权限不足试 sudo，或设备被占用）")
        self.device.print_version()

        self._channels: List[int] = []
        # feedback: motor_id -> (时间, dict)，按通道隔离
        self._fb_lock = threading.Lock()
        self._feedback: Dict[Tuple[int, int], Tuple[float, dict]] = {}
        self.device.hook_recv_callback(self._on_recv)

    # ---------- 通道 ----------

    def enable_channels(self, channels: List[int]) -> None:
        for ch in channels:
            self.device.enable_channel(ch, True)
        self._channels = list(channels)

    def close(self) -> None:
        try:
            for ch in self._channels:
                try:
                    self.device.enable_channel(ch, False)
                except Exception:
                    pass
        finally:
            self.device.close()
            self._context.destroy()

    # ---------- 指令 ----------

    def _send(self, channel: int, motor_id: int, payload: bytes) -> None:
        self.device.send_can(channel, motor_id, 8, payload)

    def enable_motor(self, channel: int, motor_id: int) -> None:
        self._send(channel, motor_id, ENABLE_FRAME)

    def disable_motor(self, channel: int, motor_id: int) -> None:
        self._send(channel, motor_id, DISABLE_FRAME)

    def send_mit(self, channel: int, motor_id: int, p_des: float,
                 v_des: float = 0.0, kp: float = 0.0, kd: float = 0.0,
                 tau: float = 0.0) -> None:
        self._send(channel, motor_id, pack_mit(p_des, v_des, kp, kd, tau))

    def disable_all(self, motor_map: List[Tuple[int, int]]) -> None:
        """批量失能（退出/异常时调用）"""
        for ch, mid in motor_map:
            try:
                self.disable_motor(ch, mid)
            except Exception:
                pass

    # ---------- 反馈 ----------

    def _on_recv(self, _dev, frame) -> None:
        h = frame.head
        master_id = h.can_id
        # 只收反馈帧：Master ID 范围 0x11~0x2E
        if not (0x11 <= master_id <= 0x2E):
            return
        motor_id = master_id - MASTER_ID_OFFSET
        dlen = h.dlc if h.dlc <= 8 else {9: 12, 10: 16, 11: 20, 12: 24}.get(h.dlc, 8)
        payload = bytes(frame.payload[i] for i in range(min(dlen, 8)))
        try:
            state = unpack_mit_reply(payload)
        except Exception:
            state = {"raw": payload.hex()}
        with self._fb_lock:
            self._feedback[(h.channel, motor_id)] = (time.time(), state)

    def get_feedback(self, channel: int, motor_id: int,
                     max_age: float = 0.1) -> Optional[dict]:
        """取某电机最新反馈，超过 max_age 视为过期返回 None"""
        with self._fb_lock:
            item = self._feedback.get((channel, motor_id))
        if item is None:
            return None
        ts, state = item
        if time.time() - ts > max_age:
            return None
        return state

    def feedback_count(self) -> int:
        with self._fb_lock:
            return len(self._feedback)
