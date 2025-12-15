import torch
import torch.nn as nn
import torch.nn.functional as F

## 这个是用了3*3 卷积核在level2 的代码


# ----------------------------
# 辅助函数：正交性正则
# ----------------------------
def orthogonality_loss(conv_weight):
    """
    conv_weight: tensor of shape (out_channels, in_channels, *kernel_dims)
    We flatten each filter to vector and compute Gram matrix G = W W^T,
    encourage off-diagonal elements to be small and diagonal close to 1 (optional).
    Returns scalar loss.
    """
    W = conv_weight.view(conv_weight.size(0), -1)  # (out_channels, K)
    # normalize each filter to unit length to focus on orthogonality
    W_norm = F.normalize(W, p=2, dim=1)
    G = torch.matmul(W_norm, W_norm.t())  # (out, out) Gram matrix
    I = torch.eye(G.size(0), device=G.device)
    # Off-diagonal should be close to 0. Diagonal should be close to 1.
    loss_off = ((G - I) ** 2).sum() - ((torch.diagonal(G - I)) ** 2).sum()
    loss_diag = ((torch.diagonal(G) - 1.0) ** 2).sum()
    # combine (we primarily penalize off-diagonal)
    return loss_off + 0.1 * loss_diag


# ----------------------------
# 模块：1D 卷积 bank（用于低通或高通）
# - 多个 filters（out_channels = n_filters），in_channels 可配置（通常是 1 或 k）
# - 并提供访问 weight 以加入正交正则
# ----------------------------
class Conv1DBank(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size, stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # 保持长度（默认）
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=n_filters,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        # 可选：使用正交初始化（对每个输出 filter 单独正交化）
        # self.reset_parameters()

    def reset_parameters(self):
        # Kaiming init then orthonormalize via QR on flattened filters
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        # Orthonormalize rows if possible
        W = self.conv.weight.data.view(self.conv.out_channels, -1)  # (out, K)
        if W.size(0) <= W.size(1):
            # QR on transpose to orthonormalize row vectors approx
            try:
                q, r = torch.linalg.qr(W.t(), mode='reduced')  # q: (K, out)
                W_ortho = q.t()
                self.conv.weight.data.copy_(W_ortho.view_as(self.conv.weight.data))
            except Exception:
                pass  # QR may fail on some shapes or dtypes; ignore
        # bias left as is

    def forward(self, x):
        # x: (B, in_channels, T)
        return self.conv(x)  # (B, n_filters, T_out)


# ----------------------------
# 模块：2D 卷积 bank，用于对 (g x L2) 矩阵做 2D 卷积（低通/高通）
# 输入 expected shape: (B, in_ch=1, H=g, W=L2) 或 (B, in_ch=g, H=1, W=L2) 取决实现
# 我们采用输入 (B, 1, H=g, W=L2) -> 输出 (B, out_channels, H_out, W_out)
# ----------------------------
class Conv2DBank(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        # try QR orthonormalization on flattened 2D filters across out_channels
        W = self.conv.weight.data.view(self.conv.out_channels, -1)
        if W.size(0) <= W.size(1):
            try:
                q, r = torch.linalg.qr(W.t(), mode='reduced')  # q: (K, out)
                W_ortho = q.t()
                self.conv.weight.data.copy_(W_ortho.view_as(self.conv.weight.data))
            except Exception:
                pass

    def forward(self, x):
        # x: (B, in_channels, H, W)
        return self.conv(x)  # (B, out_channels, H_out, W_out)


# ----------------------------
# 一层（level）定义：实现你描述的流程
# - 输入：x (B, 1, T) 或上层输出 (B, C_prev, T_prev)
# - 步骤：
#   1) 用 k 个 1D 低通滤波器扫描时间序列 => low_bank_out (B, k, L1)  (并作为 skip)
#   2) 用 g 个 1D 高通滤波器扫描 low_bank_out（输入 channels = k）=> gl2 (B, g, L2)
#   3) 将 gl2 视为矩阵 (B, 1, H=g, W=L2)，对其做 2D 低通/高通卷积得到 kl3, gl3（2D conv 输出）
#   4) 返回 low_skip (B,k,L1), kl3, gl3, 还可以返回用于下一级的某个输出（例如低频的 downsample）
# ----------------------------
class WaveletLikeLevel(nn.Module):
    def __init__(self,
                 in_channels=1,
                 k_low=8, low_kernel=7,
                 g_high=8, high_kernel=5,
                 conv2d_out_k=8, conv2d_out_g=8,
                 conv2d_kernel=(3,3)):
        super().__init__()
        # 1D low-pass bank (out_channels = k_low)
        self.low1d = Conv1DBank(in_channels=in_channels,
                                n_filters=k_low,
                                kernel_size=low_kernel,
                                padding=(low_kernel-1)//2)
        # 1D high-pass bank applied on the low1d outputs (input channels = k_low)
        self.high1d_on_low = Conv1DBank(in_channels=k_low,
                                        n_filters=g_high,
                                        kernel_size=high_kernel,
                                        padding=(high_kernel-1)//2)
        # 2D conv banks applied on gl2 (we feed gl2 as (B, 1, H=g_high, W=L2))
        # low 2D (keeps skip), high 2D (no skip)
        self.conv2d_low = Conv2DBank(in_channels=1, out_channels=conv2d_out_k, kernel_size=conv2d_kernel)
        self.conv2d_high = Conv2DBank(in_channels=1, out_channels=conv2d_out_g, kernel_size=conv2d_kernel)

        # activation
        self.act = nn.ReLU()

    def forward(self, x):
        """
        x: (B, in_channels, T)
        returns dict with keys:
          - low1d: (B, k_low, L1)   <- low freq skip
          - gl2:   (B, g_high, L2)  <- raw high outputs (before 2D)
          - kl3:   (B, conv2d_out_k, Hk, Wk) <- 2D low output
          - gl3:   (B, conv2d_out_g, Hg, Wg) <- 2D high output
        """
        # 1) low-pass bank
        low1d = self.low1d(x)          # (B, k_low, L1)
        low1d = self.act(low1d)

        # 2) apply high-pass bank on low1d (treat low channels as input channels)
        gl2 = self.high1d_on_low(low1d)  # (B, g_high, L2)
        gl2 = self.act(gl2)

        # 3) prepare gl2 as 2D tensor: shape -> (B, 1, H=g_high, W=L2)
        gl2_2d = gl2.unsqueeze(1)  # (B, 1, g_high, L2)

        kl3 = self.conv2d_low(gl2_2d)   # (B, conv2d_out_k, Hk, Wk)
        kl3 = self.act(kl3)
        gl3 = self.conv2d_high(gl2_2d)  # (B, conv2d_out_g, Hg, Wg)
        gl3 = self.act(gl3)

        return {
            'low1d': low1d,
            'gl2': gl2,
            'kl3': kl3,
            'gl3': gl3
        }

    def orth_loss(self):
        # gather orth losses from inner convs
        loss = 0.0
        loss = loss + orthogonality_loss(self.low1d.conv.weight)
        loss = loss + orthogonality_loss(self.high1d_on_low.conv.weight)
        loss = loss + orthogonality_loss(self.conv2d_low.conv.weight)
        loss = loss + orthogonality_loss(self.conv2d_high.conv.weight)
        return loss


# ----------------------------
# 总网络：多层堆叠并最终 linear 分类
# - 会把每层的 low1d (skip) 和 每层的 kl3/gl3 flatten 后 concat，用全连接分类
# ----------------------------
class WaveletLikeClassifier(nn.Module):
    def __init__(self,
                 input_channels=1,
                 input_length=256,
                 levels=2,
                 level_configs=None,
                 final_fc_hidden=256,
                 n_classes=2,
                 orth_weight=1e-3):
        super().__init__()
        self.levels = levels
        self.orth_weight = orth_weight
        self.input_length = input_length

        # 如果没有提供每层配置，使用默认
        if level_configs is None:
            level_configs = []
            for i in range(levels):
                # 默认每层减半通道/长度或者固定配置
                level_configs.append({
                    'k_low': 8,
                    'low_kernel': 7,
                    'g_high': 8,
                    'high_kernel': 5,
                    'conv2d_out_k': 8,
                    'conv2d_out_g': 8,
                    'conv2d_kernel': (3,3)
                })

        assert len(level_configs) == levels

        # 构建每一层并记录输入通道/长度变换（我们不做严格的 downsample，这里假设 padding 保持长度）
        self.level_modules = nn.ModuleList()
        in_ch = input_channels
        for i in range(levels):
            cfg = level_configs[i]
            lvl = WaveletLikeLevel(in_channels=in_ch,
                                   k_low=cfg['k_low'],
                                   low_kernel=cfg['low_kernel'],
                                   g_high=cfg['g_high'],
                                   high_kernel=cfg['high_kernel'],
                                   conv2d_out_k=cfg['conv2d_out_k'],
                                   conv2d_out_g=cfg['conv2d_out_g'],
                                   conv2d_kernel=cfg['conv2d_kernel'])
            self.level_modules.append(lvl)
            # for next level, we could set in_ch = cfg['k_low'] (use low outputs as input)
            in_ch = cfg['k_low']

        # 最后的分类 head：根据拼接特征维度来自动构造 linear
        # 我们不知道精确的每个 kl3/gl3 的空间大小（depends on conv paddings）, 所以下面创建 head 时用 lazy linear
        # 这里先使用 an adaptive pooling 以获得固定向量
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 用在 1D 信号上
        # For 2D outputs we'll use AdaptiveAvgPool2d((1,1))
        self.global_pool2d = nn.AdaptiveAvgPool2d((1,1))

        # We'll create a final MLP after we know flattened size; use LazyLinear
        self.flatten_proj = nn.Identity()  # placeholder -> we will compute feature dim lazily in forward if needed
        # Simple classifier MLP
        self.classifier = nn.Sequential(
            nn.LazyLinear(final_fc_hidden),
            nn.ReLU(),
            nn.Linear(final_fc_hidden, n_classes)
        )

    def forward(self, x):
        """
        x: (B, 1, T)
        returns logits (B, n_classes) and also aux outputs (features) if needed
        """
        B = x.size(0)
        low_skips = []
        kl3s = []
        gl3s = []
        orth_losses = 0.0

        out = x
        for lvl in self.level_modules:
            res = lvl(out)  # dict
            # collect low skip (B, k, L1)
            low_skips.append(res['low1d'])
            # collect 2D results (kl3, gl3)
            kl3s.append(res['kl3'])
            gl3s.append(res['gl3'])
            # prepare next level's input: we use low1d as next input (works since low1d channels = in_ch of next)
            out = res['low1d']
            # accumulate orth loss
            orth_losses = orth_losses + lvl.orth_loss()

        # 合并特征：
        # - 把所有 low_skips 做 1D 全局池化 -> (B, k_i, 1) -> flatten
        low_feats = []
        for t in low_skips:
            # t: (B, k, L)
            pooled = self.global_pool(t)  # (B, k, 1)
            low_feats.append(pooled.view(B, -1))  # (B, k)

        # - 把所有 kl3 和 gl3 做 2D 全局池化 -> (B, out_ch, 1,1) -> flatten
        kl_feats = []
        for k2 in kl3s:
            # k2: (B, ch, H, W)
            p = self.global_pool2d(k2)  # (B, ch, 1,1)
            kl_feats.append(p.view(B, -1))
        gl_feats = []
        for g2 in gl3s:
            p = self.global_pool2d(g2)
            gl_feats.append(p.view(B, -1))

        # concat 所有特征
        features = torch.cat(low_feats + kl_feats + gl_feats, dim=1)  # (B, D)
        # 将 features 送进 classifier
        logits = self.classifier(features)  # (B, n_classes)
        print(features[0])
        print(features.var(dim=0))
        # 返回 logits 以及 orth loss 以便训练时加权
        return {
            'logits': logits,
            'features': features,
            'orth_loss': orth_losses * self.orth_weight
        }


if __name__ == "__main__":
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    # ======================================================
    # 1. 构造一个模拟数据集（你可改成自己的真实数据）
    # ======================================================
    class RandomTSDataset(Dataset):
        def __init__(self, num_samples=500, seq_len=256, n_classes=3):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.n_classes = n_classes

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            x = torch.randn(1, self.seq_len)  # 时间序列
            y = torch.randint(0, self.n_classes, (1,)).item()
            return x, y

    # 创建数据集和 DataLoader
    seq_len = 256
    n_classes = 3
    dataset = RandomTSDataset(num_samples=800, seq_len=seq_len, n_classes=n_classes)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # ======================================================
    # 2. 创建模型
    # ======================================================
    model = WaveletLikeClassifier(
        input_channels=1,
        input_length=seq_len,
        levels=2,
        n_classes=n_classes,
        orth_weight=1e-3
    )

    # ======================================================
    # 3. 优化器和损失函数
    # ======================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ======================================================
    # 4. 训练（59 epochs）
    # ======================================================
    epochs = 59

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_orth_loss = 0.0

        for x, y in train_loader:
            # x: (B, 1, 256)
            # y: (B,)
            optimizer.zero_grad()

            out = model(x)  # 得到 logits 和 orth_loss

            cls_loss = criterion(out["logits"], y)
            orth_loss = out["orth_loss"]
            loss = cls_loss + orth_loss  # 总 loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_orth_loss += orth_loss.item()

        # 每个 epoch 显示结果
        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"Total Loss: {total_loss:.4f} | "
              f"CLS: {total_cls_loss:.4f} | "
              f"ORTH: {total_orth_loss:.4f}")

    print("训练完成！")
