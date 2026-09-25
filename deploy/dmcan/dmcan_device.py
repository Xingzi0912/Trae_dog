import ctypes
from collections.abc import Callable
from ctypes import *
from typing import Optional, Any

from dmcan.dmcan_def import dmcan_device_handle, dmcan_channel_can_config, dmcan_channel_can_info, DEV_RECV_CALLBACK, \
    DEV_SENT_CALLBACK, DEV_ERR_CALLBACK, usb_rx_frame


class DmCanDevice:


    def __init__(self, dll: ctypes.CDLL,handle:POINTER(dmcan_device_handle),index:int):
        self.index = index
        self.dll = dll
        self._handle = handle
        self._recv_callback:Optional[Callable]=None
        self._sent_callback:Optional[Callable]=None
        self._error_callback:Optional[Callable]=None

        self._recv_cb_wrapper:Optional[Callable]=None
        self._sent_cb_wrapper:Optional[Callable]=None
        self._error_cb_wrapper:Optional[Callable]=None

        self._init_funcs()


    def _init_funcs(self):

        self.dll.dmcan_device_open.argtypes=[POINTER(dmcan_device_handle)]
        self.dll.dmcan_device_open.restype=c_bool

        self.dll.dmcan_device_close.argtypes=[POINTER(dmcan_device_handle)]
        self.dll.dmcan_device_close.restype=None

        self.dll.dmcan_device_get_version.argtypes = [POINTER(dmcan_device_handle),c_char_p, c_size_t]
        self.dll.dmcan_device_get_version.restype = None

        self.dll.dmcan_device_print_version.argtypes=[POINTER(dmcan_device_handle)]
        self.dll.dmcan_device_print_version.restype=None

        self.dll.dmcan_device_enable_channel.argtypes=[POINTER(dmcan_device_handle),c_uint8]
        self.dll.dmcan_device_enable_channel.restype=None

        self.dll.dmcan_device_disable_channel.argtypes=[POINTER(dmcan_device_handle),c_uint8]
        self.dll.dmcan_device_disable_channel.restype=None

        self.dll.dmcan_device_get_channel_baudrate.argtypes=[POINTER(dmcan_device_handle),c_uint8,POINTER(dmcan_channel_can_info)]
        self.dll.dmcan_device_get_channel_baudrate.restype=c_bool

        self.dll.dmcan_device_get_channel_baudrate_details.argtypes = [POINTER(dmcan_device_handle), c_uint8,POINTER(dmcan_channel_can_config)]
        self.dll.dmcan_device_get_channel_baudrate_details.restype = c_bool

        self.dll.dmcan_device_set_channel_baudrate.argtypes=[POINTER(dmcan_device_handle),c_uint8,dmcan_channel_can_info]
        self.dll.dmcan_device_set_channel_baudrate.restype=c_bool

        self.dll.dmcan_device_set_channel_baudrate_details.argtypes = [POINTER(dmcan_device_handle), c_uint8,dmcan_channel_can_config]
        self.dll.dmcan_device_set_channel_baudrate_details.restype = c_bool

        self.dll.dmcan_device_hook_recv_callback.argtypes=[POINTER(dmcan_device_handle),DEV_RECV_CALLBACK]
        self.dll.dmcan_device_hook_recv_callback.restype=None

        self.dll.dmcan_device_hook_sent_callback.argtypes = [POINTER(dmcan_device_handle), DEV_SENT_CALLBACK]
        self.dll.dmcan_device_hook_sent_callback.restype = None

        self.dll.dmcan_device_hook_err_callback.argtypes = [POINTER(dmcan_device_handle), DEV_ERR_CALLBACK]
        self.dll.dmcan_device_hook_err_callback.restype = None

        self.dll.dmcan_device_send_can.argtypes=[POINTER(dmcan_device_handle),c_uint8,c_uint32,c_bool,c_bool,c_bool,c_bool,c_uint8,POINTER(c_uint8)]
        self.dll.dmcan_device_send_can.restype=c_bool

        self.dll.dmcan_device_send_can_details.argtypes=[POINTER(dmcan_device_handle),c_uint8,c_uint32,c_uint16,c_uint32,c_int,c_uint32,c_bool,c_bool,c_bool,c_bool,c_bool,c_bool,c_uint8,POINTER(c_uint8)]
        self.dll.dmcan_device_send_can_details.restype=c_bool

        self.dll.dmcan_device_fill_can_queue.argtypes=[POINTER(dmcan_device_handle),c_uint8,c_uint32,c_bool,c_bool,c_bool,c_bool,c_uint8,POINTER(c_uint8)]
        self.dll.dmcan_device_fill_can_queue.restype=c_bool

        self.dll.dmcan_device_can_queue_send.argtypes=[POINTER(dmcan_device_handle)]
        self.dll.dmcan_device_can_queue_send.restype=c_bool


    def open(self)->bool:
        if not self._handle:
            return False
        return self.dll.dmcan_device_open(self._handle)

    def close(self):
        if not self._handle:
            return None
        return self.dll.dmcan_device_close(self._handle)

    def get_version(self)->str:
        if not self._handle:
            return None
        buf=create_string_buffer(256)
        self.dll.dmcan_device_get_version(self._handle,buf,len(buf))
        return buf.value.decode('utf-8')

    def print_version(self):
        if not self._handle:
            return None
        return self.dll.dmcan_device_print_version(self._handle)

    def enable_channel(self,channel:int,enable:bool)->bool:
        if not self._handle:
            return False
        if enable:
            return self.dll.dmcan_device_enable_channel(self._handle,channel)
        else:
            return self.dll.dmcan_device_disable_channel(self._handle,channel)

    def get_channel_baudrate(self,channel:int)->dmcan_channel_can_info:
        if not self._handle:
            return None

        info=dmcan_channel_can_info()
        if self.dll.dmcan_device_get_channel_baudrate(self._handle,channel,info):
            return info
        else:
            return None

    def set_channel_baudrate(self,channel:int,info:dmcan_channel_can_info)->bool:
        if not self._handle:
            return False
        if info:
            return self.dll.dmcan_device_set_channel_baudrate(self._handle,channel,info)
        else:
            return False

    def send_can(self,channel:int,can_id:int,dlen:int,payload:bytes,canfd:bool=False,ext:bool=False,rtr:bool=False,brs:bool=False)->bool:
        if rtr:
            dlen = 0

        if len(payload) < dlen:
            raise ValueError(f"Payload length {len(payload)} < {dlen}")

            # 转换 payload 为 c_uint8 数组
        data = (c_uint8 * dlen)(*payload[:dlen])

        return self.dll.dmcan_device_send_can(
            self._handle, channel, can_id, canfd, ext, rtr, brs, dlen, data)

    def fill_can_queue(self,channel:int,can_id:int,dlen:int,payload:bytes,canfd:bool=False,ext:bool=False,rtr:bool=False,brs:bool=False)->bool:
        if rtr:
            dlen = 0

        if len(payload) < dlen:
            raise ValueError(f"Payload length {len(payload)} < {dlen}")

            # 转换 payload 为 c_uint8 数组
        data = (c_uint8 * dlen)(*payload[:dlen])

        return self.dll.dmcan_device_fill_can_queue(
            self._handle, channel, can_id, canfd, ext, rtr, brs, dlen, data)

    def send_can_queue(self):
        if not self._handle:
            return None
        return self.dll.dmcan_device_can_queue_send(self._handle)


    def hook_recv_callback(self, callback: Callable[[Any, usb_rx_frame], None]):

        self._recv_callback = callback

        @DEV_RECV_CALLBACK
        def wrapper(handle_ptr, frame_ptr):
            if frame_ptr:
                frame = frame_ptr.contents
                self._recv_callback(self, frame)

        self._recv_cb_wrapper = wrapper
        self.dll.dmcan_device_hook_recv_callback(self._handle, wrapper)

    def hook_sent_callback(self, callback: Callable[[Any, usb_rx_frame], None]):

        self._sent_callback = callback

        @DEV_SENT_CALLBACK
        def wrapper(handle_ptr, frame_ptr):
            if frame_ptr:
                frame = frame_ptr.contents
                self._sent_callback(self, frame)

        self._sent_cb_wrapper = wrapper
        self.dll.dmcan_device_hook_sent_callback(self._handle, wrapper)

    def hook_err_callback(self, callback: Callable[[Any, usb_rx_frame], None]):

        self._error_callback = callback

        @DEV_ERR_CALLBACK
        def wrapper(handle_ptr, frame_ptr):
            if frame_ptr:
                frame = frame_ptr.contents
                self._error_callback(self, frame)

        self._error_cb_wrapper = wrapper
        self.dll.dmcan_device_hook_err_callback(self._handle, wrapper)
