import torch
import torch.nn.functional as F

# ====================================================
# 1. 定义 Haar 滤波器（QMF 对）
# ====================================================
sqrt2 = torch.sqrt(torch.tensor(2.0))
h = torch.tensor([1 / sqrt2, 1 / sqrt2])  # low-pass filter
g = torch.tensor([1 / sqrt2, -1 / sqrt2])  # high-pass filter

# 将它们转为列向量形式便于卷积计算
h = h.view(1, 1, -1)
g = g.view(1, 1, -1)


# ====================================================
# 2. 检查同层内滤波器正交性
# ====================================================
def check_same_layer_orthogonality(h, g):
    # 卷积核反转，用于相关计算
    h_rev = torch.flip(h, dims=[2])
    g_rev = torch.flip(g, dims=[2])

    # 内积等价于卷积结果在零位点处的值
    cross_corr_hg = F.conv1d(h_rev, g)  # h 与 g 的相关
    cross_corr_hh = F.conv1d(h_rev, h)
    cross_corr_gg = F.conv1d(g_rev, g)

    print("=== 同层内滤波器关系 ===")
    print("⟨h,h⟩ =", float(cross_corr_hh))
    print("⟨g,g⟩ =", float(cross_corr_gg))
    print("⟨h,g⟩ =", float(cross_corr_hg))


check_same_layer_orthogonality(h, g)


# ====================================================
# 3. 构造多层小波基并检查跨层正交性
# ====================================================
def upsample_filter(f):
    """在滤波器之间插入零，实现尺度扩展 (dyadic scaling)"""
    f_up = torch.zeros((f.shape[0], f.shape[1], f.shape[2] * 2 - 1))
    f_up[..., ::2] = f
    return f_up


def check_cross_layer_orthogonality(h, g):
    # 第一层和第二层的小波滤核（第二层相当于上采样后再平滑）
    h_lvl2 = upsample_filter(h)
    g_lvl2 = upsample_filter(g)

    h_rev_lvl2 = torch.flip(h_lvl2, dims=[2])

    corr_hlvl_glvl = F.conv1d(h_rev_lvl2, g_lvl2)

    print("\n=== 跨层之间关系 ===")
    print("⟨h_level₂ , g_level₂⟩ =", float(corr_hlvl_glvl))


check_cross_layer_orthogonality(h, g)

# ====================================================
# 4. 理论解释输出结果
# ====================================================
print("\n说明：")
print("- ⟨h,h⟩ 和 ⟨g,g⟩ 应接近 1（归一化能量）")
print("- ⟨h,g⟩ 应接近 0（低高频子带互不干扰）")
print("- 不同尺度间 ⟨h_level₂ , g_level₂⟩ ≈ 0 表示跨层子空间也近似正交")
