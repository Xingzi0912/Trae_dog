#!/usr/bin/env python3
"""apply_powered_stance.py —— 把加电姿态标定写回仿真（开发机运行，默认 dry-run）

输入：deploy/stance_powered_calibration.json（由 NUC 上的
      scripts/calibrate_powered_stance.py 生成）
输出：打印/写入 5 个文件的改动
      · envs/dog_env_mjx.py   Q_DEFAULT
      · envs/gait.py          Q_DEFAULT
      · envs/dog_env.py       Q_DEFAULT（旧 CPU 路径）
      · envs/dog_mjx.xml      keyframe qpos / ctrl（含重算的 base z）
      · envs/dog_positional.xml 同上

用法：
    python scripts/apply_powered_stance.py                 # 只看 diff（默认）
    python scripts/apply_powered_stance.py --write         # 真正写入（留 .bak）
    python scripts/apply_powered_stance.py --write --reset-trim
        # 额外把 deploy/posture_trim.json 的前/后微调清零
        # （因为新姿态是在"已调平"的状态下测的，调平量已经含在新 q_default 里）

注意：keyframe 的 base z 用 MuJoCo 正运动学重算，让四只脚球刚好贴地
      （脚球中心 z = 0.022），避免 reset 时先穿透再弹起的瞬态。
"""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FOOT_RADIUS = 0.022
Q_SIM_OLD = [0.0, 0.9, -1.8] * 4


# ------------------------------------------------------------------ 工具
def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def replace_q_default(text: str, values) -> str:
    """把 `VAR = np.array(...)` 的数组内容换掉（括号配平，最稳）"""
    var = "Q_DEFAULT"
    i = text.find(f"{var} = np.array(")
    if i < 0:
        raise RuntimeError(f"没找到 `{var} = np.array(`")
    j = text.find("(", i)
    depth = 0
    k = j
    while k < len(text):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                break
        k += 1
    body = ", ".join(f"{v:.4f}" for v in values)
    return text[:j + 1] + f"[{body}]" + text[k:]


def footnote_z(q_sim, scene_xml: Path) -> float:
    """给定 12 关节角，求四脚球刚好贴地（球心 z=0.022）时的 base z"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    d.qpos[0:3] = [0.0, 0.0, 0.30]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:] = np.asarray(q_sim, dtype=np.float64)
    mujoco.mj_forward(m, d)
    zmin = min(d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)][2]
               for n in FOOT_NAMES)
    return 0.30 + (FOOT_RADIUS - zmin)


def replace_keyframe(text: str, q_sim, z: float) -> str:
    """把 <keyframe> 里整个 <key .../> 元素换掉（保留缩进）

    旧实现按行替换，会漏掉折行的 `ctrl="..." />`，产生重复属性和坏 XML。
    """
    i = text.find("<keyframe>")
    if i < 0:
        raise RuntimeError("没找到 <keyframe>")
    j = text.find("</keyframe>", i)
    block = text[i:j]
    # 注意从 <keyframe> 之后开始找，否则 "<key" 会匹配到 "<keyframe>" 本身
    k = block.find("<key ", len("<keyframe>"))
    if k < 0:
        raise RuntimeError("<keyframe> 里没有 <key ...>")
    k_end = block.find("/>", k)
    if k_end < 0:
        raise RuntimeError("<key> 元素没有自闭合 />")
    qpos = "0 0 {:.4f} 1 0 0 0 ".format(z) + " ".join(f"{v:.4f}" for v in q_sim)
    ctrl = " ".join(f"{v:.4f}" for v in q_sim)
    el = f'<key name="home" qpos="{qpos}" ctrl="{ctrl}" />'
    return text[:i] + block[:k] + el + block[k_end + 2:] + text[j:]


def show_diff(name: str, old: str, new: str) -> None:
    if old == new:
        print(f"  {name}: 无变化")
        return
    print(f"  {name}:")
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(),
                                     fromfile=f"{name} (旧)", tofile=f"{name} (新)",
                                     lineterm="", n=1):
        print("    " + line)


def write_with_backup(p: Path, new: str) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    try:
        shutil.copy2(p, bak)
        note = f"备份 {bak.name}"
    except Exception as e:                      # 沙箱/只读目录下可能失败
        note = f"⚠ 备份失败({type(e).__name__})，靠 git 回滚"
    p.write_text(new, encoding="utf-8")
    print(f"    已写入 {p.relative_to(REPO)}（{note}）")


BASE_Z_PAT = __import__("re").compile(r"(init_base_z(?:_range)?: tuple = )\([^)]*\)")


def patch_base_z(text: str, z_lo: float, z_hi: float) -> str:
    """把 init_base_z / init_base_z_range 的区间换成新站姿对应的区间"""
    return BASE_Z_PAT.sub(lambda m: f"{m.group(1)}({z_lo:.2f}, {z_hi:.2f})", text)


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="把加电姿态写回仿真")
    ap.add_argument("--input", default=str(REPO / "deploy" / "stance_powered_calibration.json"))
    ap.add_argument("--write", action="store_true", help="真正写入（默认只打印 diff）")
    ap.add_argument("--reset-trim", action="store_true",
                    help="同时把 deploy/posture_trim.json 的前/后微调清零")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"找不到 {src}\n先把 NUC 上的 deploy/stance_powered_calibration.json 拷过来，"
                 f"或用 --input 指定路径")
    data = json.loads(read(src))
    q_real_new = data.get("q_default_real_powered")
    q_sim_new = data.get("q_default_sim_powered")
    if not q_real_new or len(q_real_new) != 12:
        sys.exit("JSON 里缺 q_default_real_powered（12 维）")
    if not q_sim_new or len(q_sim_new) != 12:
        sys.exit("JSON 里缺 q_default_sim_powered（12 维）")
    q_sim_new = [float(v) for v in q_sim_new]

    print("=" * 78)
    print(" 加电姿态写回仿真")
    print("=" * 78)
    print(f" 输入：{src}")
    rec = (data.get("records") or [{}])[0]
    print(f" 来源记录：kp={rec.get('kp')}, kd={rec.get('kd')}, "
          f"身高={rec.get('height_mm')} mm, 微调使用={data.get('trim_used')}")
    print("\n 仿真系姿态变化（新 - 旧）：")
    for leg, base in zip(("FL", "FR", "RL", "RR"), (0, 3, 6, 9)):
        print(f"   {leg}: hip {q_sim_new[base] - Q_SIM_OLD[base]:+.4f}  "
              f"thigh {q_sim_new[base + 1] - Q_SIM_OLD[base + 1]:+.4f}  "
              f"calf {q_sim_new[base + 2] - Q_SIM_OLD[base + 2]:+.4f}")
    calfs = [q_sim_new[i] - Q_SIM_OLD[i] for i in (2, 5, 8, 11)]
    print(f"   小腿平均 {sum(calfs) / 4:+.4f} rad → 估算升高 "
          f"{sum(calfs) / 4 * 0.188 * 1000:+.0f} mm")

    # --- 新 keyframe 的 base z ---
    scene_mjx = REPO / "envs" / "scene_mjx.xml"
    z_new = footnote_z(q_sim_new, scene_mjx)
    z_old = footnote_z(Q_SIM_OLD, scene_mjx)
    print(f"\n keyframe base z: {z_old:.4f} → {z_new:.4f} m"
          f"（+{(z_new - z_old) * 1000:.0f} mm，四脚球刚好贴地）")

    targets = [
        ("envs/dog_env_mjx.py", "pyz"),
        ("envs/gait.py", "py"),
        ("envs/dog_env.py", "pyz"),
        ("envs/dog_mjx.xml", "xml"),
        ("envs/dog_positional.xml", "xml"),
    ]
    z_lo, z_hi = z_new - 0.03, z_new + 0.01

    print("\n" + "-" * 78)
    plans = []
    for rel, kind in targets:
        p = REPO / rel
        old = read(p)
        try:
            if kind == "xml":
                new = replace_keyframe(old, q_sim_new, z_new)
            else:
                new = replace_q_default(old, q_sim_new)
                if kind == "pyz":
                    new = patch_base_z(new, z_lo, z_hi)
        except Exception as e:
            print(f"  {rel}: 跳过（{e}）")
            continue
        show_diff(rel, old, new)
        plans.append((p, new))

    # --- posture_trim ---
    trim_p = REPO / "deploy" / "posture_trim.json"
    if args.reset_trim and trim_p.exists():
        old = read(trim_p)
        obj = json.loads(old)
        obj["front"], obj["rear"] = 0.0, 0.0
        obj["note"] = (obj.get("note", "") +
                       " ｜2026-10-23：加电姿态标定后调平量已含在 q_default 里，置零。")
        new = json.dumps(obj, indent=2, ensure_ascii=False)
        show_diff("deploy/posture_trim.json", old, new)
        plans.append((trim_p, new))

    print("=" * 78)
    if not args.write:
        print(" 这是 dry-run，没有写入任何文件。确认无误后加 --write 再跑一次。")
    else:
        print(" 开始写入：")
        for p, new in plans:
            write_with_backup(p, new)

    if trim_p.exists() and not args.reset_trim:
        print("\n ⚠ 部署侧【保持原样】，不要改：")
        print("   · deploy/stance_calibration.json 和 posture_trim.json 都不动")
        print("   · 因为仿真里的 Q_DEFAULT 已经换成了\"实机加电后实际保持的姿态\"，")
        print("     部署映射 q_real = q_real_eff + S·(q_sim − q_default_sim) 在零动作时")
        print("     正好落到这次实测的姿态上（推导：q_real_eff + S·Δsim = 实测位）")
        print("   · 唯一要跟着改的是实机拼 obs 时的 [46:58] 常量 → 填新的 Q_DEFAULT")
    elif args.reset_trim:
        print("\n ⚠ 你加了 --reset-trim：只有在【同时】把 stance_calibration.json 换成"
              "实测姿态时才需要，否则会和仿真错位。")

    z_lo2, z_hi2 = z_new - 0.03, z_new + 0.01
    print("\n" + "-" * 78)
    print(f" reset 高度区间已一并改为 ({z_lo2:.2f}, {z_hi2:.2f})"
          f"（新站姿 kinematic 高度 {z_new:.4f}）")
    print(f"   3) 顺带确认终止阈值 z_terminate=0.25 仍然合适"
          f"（新站姿 {z_new:.3f}，0.25 相当于塌 {1000 * (z_new - 0.25):.0f}mm）")
    if args.write:
        print("\n 完成。建议接着做：")
        print("   python scripts/test_env_mjx.py             # 环境自检")
        print("   python scripts/probe_gait.py --mode sine   # 新姿态下基准步态是否还稳")


if __name__ == "__main__":
    main()
