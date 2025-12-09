import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesMLP(nn.Module):
    """
    一个用于时间序列分类的简单多层感知器 (MLP) 模型。
    """

    def __init__(self, input_dim, num_classes, hidden_size=512, dropout_rate=0.5):
        super(TimeSeriesMLP, self).__init__()

        # 1. 展平层 (Flattening Layer)
        # 将输入数据 (Channels, Length) 展平为单个长向量 (Channels * Length)
        self.flatten = nn.Flatten()

        # 2. 线性层 (Linear Layers)
        # 第一层：从输入维度到隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_size)
        # 第二层：隐藏层到另一个隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        # 第三层（输出层）：从隐藏层到类别数
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

        # 3. Dropout 层 (用于正则化，防止过拟合)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x 形状: (Batch, Channels, Length)

        # 1. 展平: (Batch, Channels, Length) -> (Batch, Channels * Length)
        x = self.flatten(x)

        # 2. 第一层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 3. 第二层
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 4. 输出层（不加激活函数，因为 CrossEntropyLoss 会自动处理 softmax）
        logits = self.fc3(x)

        return logits




class DualFeatureMLP(nn.Module):
    """
    一个接收两个特征（振幅和相位）并融合的双分支 MLP 模型。
    """

    def __init__(self, input_dim_per_feature, num_classes, hidden_size=512, dropout_rate=0.5):
        """
        Args:
            input_dim_per_feature (int): 每个特征展平后的维度 (Channels * Length)。
            num_classes (int): 分类类别数。
            hidden_size (int): 隐藏层维度。
        """
        super(DualFeatureMLP, self).__init__()

        # 1. 展平层 (Flattening Layer) - 两个分支共享
        self.flatten = nn.Flatten()

        # 2. 定义单个 MLP 分支的结构
        self.mlp_branch = nn.Sequential(
            # FC1
            nn.Linear(input_dim_per_feature, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # FC2
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            # 注意：这里不使用 nn.Sequential 的原因是，我们需要将两个分支的FC2输出进行拼接
        )

        # 3. 为两个分支创建独立的 FC2 实例
        # 即使结构相同，也需要独立参数
        self.branch_amp_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.branch_pha_fc2 = nn.Linear(hidden_size, hidden_size // 2)

        # 4. 融合分类器 (Fusion Classifier)
        # 两个分支的输出维度都是 hidden_size // 2
        # 拼接后输入维度为 (hidden_size // 2) * 2 = hidden_size
        fusion_input_dim = hidden_size
        self.classifier = nn.Linear(fusion_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, amp, pha):
        # amp, pha 形状: (Batch, Channels, Length)

        # --- 1. 展平 ---
        # (Batch, Channels, Length) -> (Batch, Channels * Length)
        amp = self.flatten(amp)
        pha = self.flatten(pha)

        # --- 2. 振幅分支处理 ---
        # (Batch, input_dim) -> (Batch, hidden_size)
        f_amp = self.mlp_branch[:3](amp)  # 只执行到 FC1 (Linear, ReLU, Dropout)
        f_amp = self.branch_amp_fc2(f_amp)  # 执行 FC2
        f_amp = F.relu(f_amp)
        f_amp = self.dropout(f_amp)
        # f_amp 形状: (Batch, hidden_size // 2)

        # --- 3. 相位分支处理 ---
        # (Batch, input_dim) -> (Batch, hidden_size)
        f_pha = self.mlp_branch[:3](pha)  # 只执行到 FC1 (Linear, ReLU, Dropout)
        f_pha = self.branch_pha_fc2(f_pha)  # 执行 FC2
        f_pha = F.relu(f_pha)
        f_pha = self.dropout(f_pha)
        # f_pha 形状: (Batch, hidden_size // 2)

        # --- 4. 融合 (Concatenate) ---
        # (Batch, hidden_size // 2) 拼接 (Batch, hidden_size // 2) -> (Batch, hidden_size)
        f_fused = torch.cat([f_amp, f_pha], dim=1)

        # --- 5. 最终分类 ---
        logits = self.classifier(f_fused)

        return logits