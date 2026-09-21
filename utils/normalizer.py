"""RunningMeanStd —— 观测在线归一化（Step 3.1 稳定性修复）

动机：
    58 维 obs 各分量尺度差异大（base_z≈0.27、关节角≈1.8、yaw 可累计到
    几十），不做归一化时 critic 的 MSE 被大尺度分量主导，估值不稳
    （1M 实验 vloss 反复冲到 40~70），间接放大策略梯度噪声。

做法（同 OpenAI baselines / CleanRL）：
    并行增量算法（Chan et al.）批量合并均值/方差，O(1) 每维额外内存；
    normalize(x) = clip((x - mean) / sqrt(var + eps), -10, 10)。

注意：
    - 训练时每个原始 obs 先 update 再 normalize
    - 评估/部署时只 normalize（冻结统计量），绝不 update
    - 统计量必须随策略一起保存（checkpoint），否则实机推理尺度不一致
"""

import numpy as np


class RunningMeanStd:
    """逐维跟踪 mean / var，支持批量更新与 state_dict 存取"""

    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        # count 从 eps 起步：初期方差≈1，避免头几个样本把方差压到 0
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        """合并一个样本（也可传入 (N, dim) 批量）"""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        # 新均值 = 加权平均
        self.mean = self.mean + delta * (batch_count / total)
        # 新方差 = 两批方差 + 均值差的修正项
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """用当前统计量归一化，±10 裁剪防离群点（不更新统计量）"""
        x = np.asarray(x, dtype=np.float64)
        out = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(out, -10.0, 10.0).astype(np.float32)

    # ------------------------------------------------------------
    # 随 checkpoint 存取（数值为 numpy，torch.save 可直接序列化）
    # ------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = sd["mean"].copy()
        self.var = sd["var"].copy()
        self.count = float(sd["count"])
