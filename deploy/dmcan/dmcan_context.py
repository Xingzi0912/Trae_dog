import sys

from ctypes import *
from pathlib import Path
from typing import Optional, List
from dmcan.dmcan_def import dmcan_device_handle, dmcan_context, dmcan_device_type
from dmcan.dmcan_device import DmCanDevice


def find_backend_dll_path():
    """
    查找后端 DLL 路径

    Returns:
        str: DLL 的绝对路径，找不到返回 None
    """
    # 获取平台对应的文件名
    if sys.platform == "win32":
        dll_name = "libdm_device.dll"
    elif sys.platform == "darwin":
        dll_name = "libdm_device.dylib"
    elif sys.platform == "linux":
        dll_name = "libdm_device.so"
    else:
        return None

    # 获取项目根目录（假设当前文件在 core/ 或项目根目录）
    current_file = Path(__file__).resolve()

    # 尝试多个可能的位置
    possible_paths = [
        # 相对于当前文件
        current_file.parent / "dlls" / dll_name,
        current_file.parent.parent / "dlls" / dll_name,
        # 相对于当前工作目录
        Path.cwd() / "dlls" / dll_name,
        Path.cwd() / dll_name,
        # 系统路径（直接使用文件名）
        Path(dll_name),
    ]

    # 返回第一个存在的路径
    for path in possible_paths:
        if path.exists():
            return str(path)

    # 找不到，返回默认相对路径（让调用者处理错误）
    return f"./dlls/{dll_name}"

class DmCanContext:

    def __init__(self):

        self.dll_path=find_backend_dll_path()
        # winmode 仅 Windows ctypes 支持；Linux/macOS 下传该参数会 TypeError
        if sys.platform == "win32":
            self.dll=CDLL(self.dll_path,winmode=0)
        else:
            self.dll=CDLL(self.dll_path)
        self._ctx=POINTER(dmcan_context)()
        self._devices:List[dmcan_device_handle]=[]

        self._init_funcs()

        self.dll.dmcan_context_create(self._ctx)

    def _init_funcs(self):

        self.dll.dmcan_context_create.argtypes=[POINTER(POINTER(dmcan_context))]
        self.dll.dmcan_context_create.restype=None

        self.dll.dmcan_context_destroy.argtypes=[POINTER(dmcan_context)]
        self.dll.dmcan_context_destroy.restype=None

        self.dll.dmcan_print_version.argtypes=[POINTER(dmcan_context)]
        self.dll.dmcan_print_version.restype=None

        self.dll.dmcan_find_devices.argtypes=[POINTER(dmcan_context)]
        self.dll.dmcan_find_devices.restype=c_int

        self.dll.dmcan_find_devices_with_type.argtypes=[POINTER(dmcan_context),c_int]
        self.dll.dmcan_find_devices_with_type.restype=c_int

        self.dll.dmcan_show_all_devices.argtypes=[POINTER(dmcan_context)]
        self.dll.dmcan_show_all_devices.restype=None

        self.dll.dmcan_device_get.argtypes=[POINTER(dmcan_context),POINTER(POINTER(dmcan_device_handle)),c_int]
        self.dll.dmcan_device_get.restype=c_bool

    def destroy(self):
        if self._ctx:
            self.dll.dmcan_context_destroy(self._ctx)
            self._ctx=None

    def print_version(self):
        if self._ctx:
            self.dll.dmcan_print_version(self._ctx)

    def find_devices(self, type:Optional[dmcan_device_type]=None) -> int:
        if self._ctx is None:
            return 0

        device_cnt=0
        if type is None:
            device_cnt= self.dll.dmcan_find_devices(self._ctx)
        else:
            device_cnt= self.dll.dmcan_find_devices_with_type(self._ctx, type)



        self._devices.clear()

        for i in range(device_cnt):

            dev_handle = POINTER(dmcan_device_handle)()

            ret=self.dll.dmcan_device_get(self._ctx,byref(dev_handle),i)

            if not ret:
                continue

            if not dev_handle:
                continue
            device=DmCanDevice(self.dll,dev_handle,i)
            self._devices.append(device)

        return device_cnt


    def show_all_devices(self):
        if self._ctx is None:
            return
        self.dll.dmcan_show_all_devices(self._ctx)

    def get_device(self,index:int)->DmCanDevice:
        return self._devices[index]


    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        for device in self._devices:
            device.close()
        self.destroy()

    def __del__(self):
        if hasattr(self, '_ctx'):
            self.destroy()