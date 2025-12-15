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
        super(DualFeatureMLP, self).__init__()

        # ... (flattening layer omitted for brevity) ...

        # 2. 定义共享的 FC1 结构（仅包含 FC1 及其后的激活）
        # 这是两个分支共享参数的地方
        self.mlp_branch_shared_fc1 = nn.Sequential(
            nn.Linear(input_dim_per_feature, hidden_size),  # 共享 FC1
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 3. 定义独立的 FC2 结构（参数不共享）
        # 注意：这里我们把 ReLU 和 Dropout 也放在 FC2 后面，保持结构完整
        self.branch_amp_fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # Amp 独有的 FC2
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.branch_pha_fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # Pha 独有的 FC2
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 4. 融合分类器 (不变)
        fusion_input_dim = hidden_size
        self.classifier = nn.Linear(fusion_input_dim, num_classes)
        self.dropout_final = nn.Dropout(dropout_rate)  # 使用不同的名称以示区分

        # 确保 flatten 层也被定义
        self.flatten = nn.Flatten()

    def forward(self, amp, pha):
        # --- 1. 展平 ---
        amp = self.flatten(amp)
        pha = self.flatten(pha)

        # --- 2. 共享 FC1 处理 ---
        # 两个输入使用 self.mlp_branch_shared_fc1 的相同参数
        f_amp = self.mlp_branch_shared_fc1(amp)
        f_pha = self.mlp_branch_shared_fc1(pha)

        # --- 3. 独立 FC2 处理 ---
        # 各自使用独立的 FC2 参数
        f_amp = self.branch_amp_fc2(f_amp)
        f_pha = self.branch_pha_fc2(f_pha)
        # f_amp, f_pha 形状: (Batch, hidden_size // 2)

        # --- 4. 融合 (Concatenate) ---
        f_fused = torch.cat([f_amp, f_pha], dim=1)  # (Batch, hidden_size)

        # --- 5. 最终分类 ---
        logits = self.classifier(f_fused)
        logits = self.dropout_final(logits)  # 使用最终的 Dropout 层

        return logits