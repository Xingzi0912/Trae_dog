import ctypes
from ctypes import *
from enum import Enum, IntEnum



class dmcan_device_type(IntEnum):
    USB2CANFD = 0
    USB2CANFD_DUAL=1
    LinkX4C=2

class dmcan_channel_can_info(Structure):
    _pack_ = 1
    _fields_ = [

        ("channel",c_uint8),
        ("canfd",c_bool),
        ("can_baudrate",c_uint32),
        ("canfd_baudrate",c_uint32),
        ("can_sp",c_float),
        ("canfd_sp",c_float)
    ]

class dmcan_channel_can_config(Structure):
    _pack_ = 1
    _fields_ = [
        ("channel",c_uint8),
        ("can_fd",c_uint8),
        ("can_seg1",c_uint8),
        ("can_seg2", c_uint8),
        ("can_sjw", c_uint8),
        ("can_prescaler", c_uint8),
        ("canfd_seg1", c_uint8),
        ("canfd_seg2", c_uint8),
        ("canfd_sjw", c_uint8),
        ("canfd_prescaler", c_uint8)
    ]

class usb_rx_frame_head(Structure):
    _pack_ = 1
    _fields_ = [

        ("can_id",c_uint32,29),
        ("esi",c_uint32,1),
        ("ext", c_uint32, 1),
        ("rtr", c_uint32, 1),
        ("timestamp", c_uint64),
        ("channel",c_uint8),
        ("canfd",c_uint8,1),
        ("dir",c_uint8,1),
        ("brs",c_uint8,1),
        ("ack",c_uint8,1),
        ("dlc",c_uint8,4),
        ("reserved",c_uint16),
    ]



class usb_rx_frame(Structure):
    _pack_ = 1
    _fields_ = [
        ("head",usb_rx_frame_head),
        ("payload",c_uint8*64),
    ]

    def show(self):
        """简略显示CAN帧信息"""
        dlc_to_len = {9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64}
        actual_len = self.head.dlc if self.head.dlc <= 8 else dlc_to_len.get(self.head.dlc, 64)

        payload_hex = ' '.join(f'{self.payload[i]:02X}' for i in range(min(actual_len, 8)))
        if actual_len > 8:
            payload_hex += '...'

        print(f"ID:{'E' if self.head.ext else 'S'}{self.head.can_id:08X} TS:{self.head.timestamp} CH:{self.head.channel} "
              f"{'FD' if self.head.canfd else 'CA'} {'TX' if self.head.dir else 'RX'} "
              f"DLC:{self.head.dlc}({actual_len}) "
              f"RTR:{self.head.rtr} ESI:{self.head.esi} BRS:{self.head.brs} ACK:{self.head.ack} "
              f"DATA:[{payload_hex}]")

class dmcan_context(Structure):
    pass

class dmcan_device_handle(Structure):
    pass


DEV_RECV_CALLBACK=CFUNCTYPE(None,POINTER(dmcan_device_handle),POINTER(usb_rx_frame))
DEV_SENT_CALLBACK=CFUNCTYPE(None,POINTER(dmcan_device_handle),POINTER(usb_rx_frame))
DEV_ERR_CALLBACK=CFUNCTYPE(None,POINTER(dmcan_device_handle),POINTER(usb_rx_frame))