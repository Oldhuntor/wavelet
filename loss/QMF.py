import torch
import torch.nn as nn
import torch.nn.functional as F


class QMFLoss(nn.Module):
    """
    针对 PyTorch 1D 卷积核的 QMF 约束损失。
    假设输入 h 和 g 形状为 (Num_Filters, 1, L) 或 (Num_Filters, Channels, L)。
    """

    def __init__(self, weight_energy=1.0, weight_orth=1.0, weight_mirror=1.0):
        super().__init__()
        self.wE = weight_energy
        self.wO = weight_orth
        self.wM = weight_mirror

    def forward(self, h, g):
        # 假设 h, g 形状为 (C_out, C_in, L)

        # 将输入展开为 (Num_Filters, L) 以便逐个处理 (Num_Filters = C_out * C_in)
        # ------------------------------------------------------------------
        h_kernels = h.view(-1, h.size(-1))
        g_kernels = g.view(-1, g.size(-1))
        L = h_kernels.size(-1)

        if L == 0:
            return torch.tensor(0.0, dtype=h.dtype, device=h.device)

        # (1) L_energy: 能量归一 (L2 范数平方接近 1)
        # 逐个滤波器计算 L2 范数平方
        energy_h = torch.sum(h_kernels.pow(2), dim=1)
        energy_g = torch.sum(g_kernels.pow(2), dim=1)

        # 惩罚项是所有滤波器L2范数偏离 1 的平方和的平均
        L_energy = torch.mean((energy_h - 1).pow(2)) + torch.mean((energy_g - 1).pow(2))

        # (2) L_orth: 零移位内积接近零
        # 逐个滤波器计算内积 (点乘后求和)
        inner_hg = torch.sum(h_kernels * g_kernels, dim=1)
        L_orth = torch.mean(inner_hg.pow(2))  # 对所有滤波器的内积平方求平均

        # (3) L_mirror: QMF 镜像关系
        # QMF: g[n] = (-1)^n * h[L-1-n]

        # 1. 创建逐个滤波器内部的符号交替序列 (形状: L)
        alt_signs_seq = torch.tensor([(-1) ** i for i in range(L)],
                                     dtype=h.dtype, device=h.device)

        # 2. 翻转 h (时间反转: h[L-1-n])
        h_flipped = torch.flip(h_kernels, dims=[1])

        # 3. 计算镜像目标: mirror_target = (-1)^n * h[L-1-n]
        # 使用广播机制将 alt_signs_seq 应用于所有滤波器
        mirror_target = h_flipped * alt_signs_seq

        # 4. 计算 g 偏离目标的均方误差 (MSE)
        L_mirror = torch.mean((g_kernels - mirror_target) ** 2)

        # 总损失加权求和
        total_loss = self.wE * L_energy + self.wO * L_orth + self.wM * L_mirror

        return total_loss