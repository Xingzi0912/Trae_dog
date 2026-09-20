"""启动 MuJoCo 交互式 viewer，可视化零策略下狗的站立姿态

用法：
    d:\data\Trae\deep-learning\.venv\Scripts\python.exe d:\data\Trae\dog_rl\scripts\view_stand.py

操作（viewer 窗口激活时）：
    鼠标拖拽  旋转视角
    鼠标滚轮  缩放
    右键拖拽  平移
    Space     暂停/继续物理仿真
    Esc       退出 viewer
"""

import sys
from pathlib import Path

# 让脚本能在 dog_rl 项目外被直接 python 调用（不需要安装包）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

from envs.dog_env import DogEnv


def main():
    env = DogEnv()
    obs, info = env.reset()
    print(f"[reset] base_z = {info['base_z']:.4f} m")
    print()
    print("MuJoCo viewer 启动中 ...")
    print("  拖拽鼠标看不同角度，滚轮缩放，Space 暂停，Esc 退出")
    print("  零策略会持续跑 1000 步，之后保持持帧让你慢慢看")
    print()

    # launch_passive 是非阻塞 viewer，可在循环里 step + sync 看动态
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        action = np.zeros(12, dtype=np.float32)  # 零策略
        step = 0
        try:
            # 阶段一：跑零策略 1000 步，看动态演化
            while viewer.is_running() and step < 1000:
                obs, r, term, trunc, info = env.step(action)
                viewer.sync()
                step += 1
                if step % 100 == 0:
                    print(f"  step {step:4d}  base_z = {info['base_z']:.4f} m")
                if term:
                    print(f"  狗塌了 at step {step}！")
                    break

            # 阶段二：跑完 1000 步后持帧，让用户继续观察静态站立姿态
            if viewer.is_running() and not term:
                print()
                print(f"  完成 {step} 步，狗稳稳站着 ✅")
                print("  viewer 保持开启，关闭窗口或按 Esc 退出")
                while viewer.is_running():
                    # 不再 step，只刷新画面（狗维持当前姿态）
                    viewer.sync()
        except KeyboardInterrupt:
            print("\n手动退出")
        finally:
            env.close()


if __name__ == "__main__":
    main()
