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
#
#   bash scripts/train_server.sh --steps 20000000 --n_envs 4096 --mem 1.0
#
# 注意事项：
#   - tmux 内运行，SSH 断开不中断
#   - tee 双写文件+屏幕，断线后可查日志

set -euo pipefail
cd "$(dirname "$0")/.."

# 默认参数
STEPS=${STEPS:-20000000}
N_ENVS=${N_ENVS:-2048}
MEM=${MEM:-0.9}

# 解析命令行参数（覆盖默认）
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps) STEPS="$2"; shift 2 ;;
        --n_envs) N_ENVS="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# GPU 显存限制
export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM

echo "=== dog_rl 服务器训练 ==="
echo "steps=$STEPS, n_envs=$N_ENVS, mem=$MEM"
echo ""

# 创建日志目录
mkdir -p logs checkpoints

# 运行训练（-u 无缓冲，tee 双写）
python -u train_jax.py \
    --steps "$STEPS" \
    --n_envs "$N_ENVS" \
    2>&1 | tee "logs/run_$(date +%Y%m%d_%H%M%S).log"
