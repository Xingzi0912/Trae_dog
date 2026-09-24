"""FK 高度常数标定（实机操作脚本，在 NUC 上交互运行）

前提（必须先完成）：
    1. 12 个电机零点已标定并保存（站立读数 ≈ [0, 0.9, -1.8]×4）
    2. 狗四脚均匀着地、标准立正姿态，地面水平

流程：
    脚本用关节角跑 FK 得 z_fk → 你用尺子量 base 参考点真实高度 z_true
    → offset = z_true - z_fk → 多次采样平均 → 写入 deploy/kin_calib.json
    该常数同时吃掉脚球半径、装配偏差、零位残差等全部固定系统误差。

用法：
    python3 scripts/calibrate_kin.py                    # 用站立默认角（先试流程）
    python3 scripts/calibrate_kin.py --q 0 .9 -1.8 ...  # 12 个实测关节角
    python3 scripts/calibrate_kin.py --qfile q.json     # 电机读取脚本输出的 {"q":[...]}
    python3 scripts/calibrate_kin.py --samples 3        # 采样 3 次取平均
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy.kin_estimator import CALIB_PATH, KinEstimator, rpy_to_R, leg_fk


def fk_height(q: np.ndarray, rpy: np.ndarray) -> float:
    """四脚全接触平均的 FK 高度（不含 offset）"""
    R_wb = rpy_to_R(rpy)
    zs = []
    for leg in range(4):
        p_foot, _, _ = leg_fk(q[3 * leg:3 * leg + 3], leg)
        zs.append(-(R_wb @ p_foot)[2])
    return float(np.mean(zs))


def main():
    ap = argparse.ArgumentParser(description="FK 高度常数标定")
    ap.add_argument("--q", nargs=12, type=float, help="12 个实测关节角（rad）")
    ap.add_argument("--qfile", help='JSON 文件，含 {"q":[12 个数]}（电机读取脚本输出）')
    ap.add_argument("--rpy", nargs=3, type=float, default=[0, 0, 0],
                    help="立正时 IMU 姿态角（rad），默认水平 0")
    ap.add_argument("--samples", type=int, default=1, help="采样次数取平均（默认 1）")
    args = ap.parse_args()

    q_file = None
    if args.qfile:
        q_file = np.asarray(json.loads(Path(args.qfile).read_text())["q"],
                            dtype=np.float64)
        assert q_file.shape == (12,), "qfile 中 q 必须是 12 维"

    print("=" * 56)
    print("FK 高度常数标定")
    print("确认：电机零点已标定？狗已水平立正四脚均匀着地？(y/n)")
    if input("> ").strip().lower() != "y":
        print("请先完成前提条件再运行。")
        return

    offsets = []
    for s in range(args.samples):
        # 关节角优先级：命令行 --q / --qfile / 站立默认角
        if args.q is not None:
            q = np.asarray(args.q, dtype=np.float64)
            src = "命令行输入"
        elif q_file is not None:
            q = q_file
            src = f"文件 {args.qfile}"
        else:
            q = np.tile([0, 0.9, -1.8], 4)
            src = "站立默认角"

        z_fk = fk_height(q, np.asarray(args.rpy))
        print(f"\n[采样 {s + 1}/{args.samples}] 关节角来源：{src}")
        print("  12 关节角：", np.round(q, 3).tolist())
        print(f"  FK 计算高度 z_fk = {z_fk * 1000:.1f} mm")
        while True:
            raw = input("  用尺量 base_link 参考点离地真实高度 z_true（mm）：").strip()
            try:
                z_true = float(raw) / 1000.0
                break
            except ValueError:
                print("  请输入数字（毫米）")
        offsets.append(z_true - z_fk)
        print(f"  → 本次 offset = {offsets[-1] * 1000:+.1f} mm")

    offset = float(np.mean(offsets))
    print("\n" + "=" * 56)
    print(f"各次 offset：{[round(o * 1000, 1) for o in offsets]} mm")
    print(f"最终 z_offset = {offset * 1000:+.1f} mm")

    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps({"z_offset": offset}, indent=2))
    print(f"已写入 {CALIB_PATH}")

    # 验证：重新加载估计器，用同一组姿态跑一次
    est = KinEstimator()
    contact = np.array([True] * 4)
    q_final = np.asarray(args.q if args.q is not None else
                         (q_file if q_file is not None else
                          np.tile([0, 0.9, -1.8], 4)), dtype=np.float64)
    z_check, _ = est.update(q_final, np.zeros(12),
                            np.asarray(args.rpy), np.zeros(3), contact)
    print(f"\n验证：重载标定后估计高度 = {z_check * 1000:.1f} mm"
          f"（应≈你刚量的真实高度，残差为 FK 姿态简化误差）")
    print("标定完成。后续 KinEstimator 自动读取该文件。")


if __name__ == "__main__":
    main()
