#!/usr/bin/env bash
# 4090 服务器训练脚本（2026-10-12，决策 6/7/8）
#
# 用法：
#   tmux new -s dog_train
#   bash scripts/train_server.sh
#
# 可选参数：
#   --steps 20000000    总训练步数
#   --n_envs 2048       并行环境数
#   --mem 0.9           XLA 显存占比（0.0-1.0，默认 0.9）
#   --target_kl_mb 0.2  per-mb 累积 KL 冻结阈值（不传 = train_jax.py 里
#                       PPOConfig 的默认 0.1；v13 实验用的是 0.2）
#   --resume            从 checkpoints/dog_cmd_mjx_teacher.pkl 续训
#                       （Adam 动量不保存，前 1-2 update 步长偏保守；
#                        该文件不存在会直接报错）
#
#   bash scripts/train_server.sh --steps 20000000 --n_envs 4096 --mem 1.0
#   bash scripts/train_server.sh --target_kl_mb 0.2
#
# 注意事项：
#   - tmux 内运行，SSH 断开不中断
#   - tee 双写文件+屏幕，断线后可查日志
#   - 所有生效参数都在启动头部打印，避免配置静默不一致
#     （2026-10-16：脚本没传 --target_kl_mb，日志里 kl_lim 变成 0.100）
#   - checkpoints/dog_cmd_mjx_teacher.pkl 是固定路径，会被本次训练
#     每 save_interval 个 update 覆写；要保留旧的就先 cp 一份

set -euo pipefail
cd "$(dirname "$0")/.."

# 默认参数
STEPS=${STEPS:-20000000}
N_ENVS=${N_ENVS:-2048}
MEM=${MEM:-0.9}
# 留空 = 不传该 flag，由 train_jax.py 的 PPOConfig 默认值（0.1）决定
TARGET_KL_MB=${TARGET_KL_MB:-}
RESUME=${RESUME:-0}
EXTRA=()          # 未识别参数原样透传给 train_jax.py

# 解析命令行参数（覆盖默认）
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps) STEPS="$2"; shift 2 ;;
        --n_envs) N_ENVS="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --target_kl_mb) TARGET_KL_MB="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        # 其余参数原样透传（如 --cmd_vx/--cmd_vy/--cmd_yaw/--zero_cmd_prob/--gait_mode）
        *) EXTRA+=("$1"); shift ;;
    esac
done

# GPU 显存限制
export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM

# mem=1.0 会导致 OOM：GPU 驱动/CUDA 上下文占了 ~500MB-1GB，
# JAX 试图占 100% 永远达不到 → 分配失败。建议 ≤0.9
if (( $(echo "$MEM > 0.95" | bc -l) )); then
    echo "⚠️  警告：--mem $MEM > 0.95 几乎必然 OOM（GPU 驱动占用无法释放）"
    echo "   建议用 --mem 0.9（留 ~2.4GB 给驱动/CUDA 上下文）"
    echo ""
fi

# 组装训练命令（数组形式：set -u 下旧 bash 展开空 ${arr[@]} 会报错，
# 而 CMD 永远非空，用 "${CMD[@]}" 安全）
CMD=(python -u train_jax.py --steps "$STEPS" --n_envs "$N_ENVS")
# 透传参数（bash 4.4+ 的 set -u 对空数组友好；旧 bash 用下面这个写法保平安）
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    CMD+=("${EXTRA[@]}")
fi
if [[ -n "$TARGET_KL_MB" ]]; then
    CMD+=(--target_kl_mb "$TARGET_KL_MB")
fi
if [[ "$RESUME" == "1" ]]; then
    CMD+=(--resume)
fi

echo "=== dog_rl 服务器训练 ==="
echo "steps=$STEPS, n_envs=$N_ENVS, mem=$MEM"
if [[ -n "$TARGET_KL_MB" ]]; then
    echo "target_kl_mb=$TARGET_KL_MB"
else
    echo "target_kl_mb=未指定（用 train_jax.py 的 PPOConfig 默认 0.1）"
fi
echo "resume=$RESUME"
echo "cmd: ${CMD[*]}"
echo ""

# 创建日志目录
mkdir -p logs checkpoints

# 运行训练（tee 双写）
"${CMD[@]}" 2>&1 | tee "logs/run_$(date +%Y%m%d_%H%M%S).log"
