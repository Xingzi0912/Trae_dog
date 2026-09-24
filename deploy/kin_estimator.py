"""腿部 FK 状态估计器（实机部署层）

原理（支撑脚约束，纯代数、无积分、不随时间漂移）：
    假设支撑脚在世界系中静止于地面（z=0、v=0），
    用 12 个关节角做正运动学（FK）得到脚相对 base 的位置 p_i，
    反解 base 高度；对时间求导（解析雅可比）反解 base 线速度。

        base_z_i = -[ R_wb · p_i ]_z                 （单脚高度估计）
        v_base_b = -ω_b × p_i - ṗ_i                 （机身系线速度）
    多只支撑脚取平均 → 一阶低通 → 加标定常数 offset。

输入（update）：
    q[12]       关节角（训练坐标系，rad），顺序 [FL,FR,RL,RR]×[hip,thigh,calf]
    qd[12]      关节速度（rad/s），同序
    rpy[3]      base 姿态（rad，IMU 欧拉角换算后）
    gyro[3]     base 角速度（rad/s，机身系，IMU 直出）
    contact[4]  支撑脚掩码（True=该脚着地，由步态相位/电流/足端开关给出）

几何参数与 envs/dog_positional.xml 完全一致（修改 XML 连杆必须同步本文件）。

标定：deploy/kin_calib.json 存 {"z_offset": float}，由 scripts/calibrate_kin.py 生成。
"""

import json
from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# 腿几何（dog_positional.xml）
# ------------------------------------------------------------
# FK 链：从 base 到脚球心，4 段固定平移 + 3 个旋转关节
#   段 0: base → hip body        p0
#   关节 0: hip 侧摆，绕局部 X
#   段 1: hip → thigh body       p1
#   关节 1: thigh 大腿，绕局部 Y
#   段 2: thigh → calf body      p2
#   关节 2: calf 小腿，绕局部 Y
#   段 3: calf → 脚球心          p3
_AX_X = np.array([1.0, 0.0, 0.0])
_AX_Y = np.array([0.0, 1.0, 0.0])
_JOINT_AXES = (_AX_X, _AX_Y, _AX_Y)

# 腿索引 0=FL 1=FR 2=RL 3=RR；左右侧 y 符号
_SX = (1.0, -1.0, 1.0, -1.0)      # hip y、thigh 偏移 y
_FX = (1.0, 1.0, -1.0, -1.0)      # hip x
_HIP_X = 0.22337
_HIP_Y = 0.06
_THIGH_Y = 0.1039
_L_THIGH = 0.22
_FOOT_D = np.array([-0.002, 0.0, -0.24])   # calf → 脚球心

CALIB_PATH = Path(__file__).parent / "kin_calib.json"


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """通用轴角旋转矩阵（Rodrigues）"""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _leg_chain(leg: int):
    """返回某条腿的 (4 段平移, 3 关节轴)"""
    sx, fx = _SX[leg], _FX[leg]
    offsets = (
        np.array([_HIP_X * fx, _HIP_Y * sx, 0.0]),
        np.array([0.0, _THIGH_Y * sx, 0.0]),
        np.array([0.0, 0.0, -_L_THIGH]),
        _FOOT_D,
    )
    return offsets, _JOINT_AXES


# 预计算 4 条腿的链（常量）
_LEG_CHAINS = [_leg_chain(i) for i in range(4)]


def leg_fk(q3: np.ndarray, leg: int):
    """单腿正运动学

    返回：
        p_foot[3]    脚球心相对 base 的位置（base frame）
        R[3×3]       脚相对 base 的旋转
        joints       每个关节 [(axis_b, origin_b)]（base frame，供雅可比使用）
    """
    offsets, axes = _LEG_CHAINS[leg]
    R = np.eye(3)
    p = np.zeros(3)
    joints = []
    for k in range(3):
        p = p + R @ offsets[k]          # 进入关节原点
        joints.append((R @ axes[k], p.copy()))
        R = R @ _rot_axis(axes[k], float(q3[k]))
    p = p + R @ offsets[3]              # 脚球心
    return p, R, joints


def leg_jacobian(q3: np.ndarray, leg: int):
    """单腿几何雅可比（脚相对 base 的线速度部分）

    旋转关节：j_k = axis_k_b × (p_foot − origin_k_b)；ṗ = J @ q̇
    """
    p_foot, _, joints = leg_fk(q3, leg)
    J = np.zeros((3, 3))
    for k, (axis_b, origin_b) in enumerate(joints):
        J[:, k] = np.cross(axis_b, p_foot - origin_b)
    return J, p_foot


def rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """[roll,pitch,yaw]（rad）→ 世界相对 base 的旋转 R_wb = Rz(yaw)Ry(pitch)Rx(roll)"""
    r, p, y = rpy
    return _rot_axis(np.array([0.0, 0.0, 1.0]), y) @ \
        _rot_axis(np.array([0.0, 1.0, 0.0]), p) @ \
        _rot_axis(np.array([1.0, 0.0, 0.0]), r)


class KinEstimator:
    """FK 状态估计器：50Hz 调用 update → 读 base_z / lin_vel_b

    参数：
        lpf_alpha     输出一阶低通系数（50Hz 下 0.5≈11Hz 截止，越小越平滑延迟越大）
        z_offset      高度标定常数（m）；None 时自动读 deploy/kin_calib.json，无文件则 0
    """

    def __init__(self, lpf_alpha: float = 0.5, z_offset: float = None):
        self.alpha = lpf_alpha
        if z_offset is None:
            z_offset = self.load_calib_offset()
        self.z_offset = float(z_offset)
        self._z = None       # 低通后的高度（不含 offset）
        self._v = None       # 低通后的机身系线速度

    @staticmethod
    def load_calib_offset() -> float:
        if CALIB_PATH.exists():
            return float(json.loads(CALIB_PATH.read_text())["z_offset"])
        return 0.0

    def update(self, q: np.ndarray, qd: np.ndarray, rpy: np.ndarray,
               gyro: np.ndarray, contact: np.ndarray):
        """跑一次估计。contact 全 False 时保持上次输出（全部脚摆动相）。

        返回 (base_z, lin_vel_b)；base_z 已含标定 offset。
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        rpy = np.asarray(rpy, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        contact = np.asarray(contact, dtype=bool)

        R_wb = rpy_to_R(rpy)
        z_list, v_list = [], []
        for leg in range(4):
            if not contact[leg]:
                continue
            q3 = q[3 * leg:3 * leg + 3]
            qd3 = qd[3 * leg:3 * leg + 3]
            J, p_foot = leg_jacobian(q3, leg)
            p_dot = J @ qd3                       # 脚相对 base 的速度（base frame）
            # 高度：脚世界 z = base_z + [R_wb·p]_z = 0
            z_list.append(-(R_wb @ p_foot)[2])
            # 速度：0 = v_base_b + ω_b × p + ṗ
            v_list.append(-np.cross(gyro, p_foot) - p_dot)

        if z_list:
            z_raw = float(np.mean(z_list))
            v_raw = np.mean(v_list, axis=0)
            self._z = z_raw if self._z is None else \
                self.alpha * z_raw + (1 - self.alpha) * self._z
            self._v = v_raw if self._v is None else \
                self.alpha * v_raw + (1 - self.alpha) * self._v

        if self._z is None:
            return 0.0, np.zeros(3)
        return self._z + self.z_offset, self._v.copy()
