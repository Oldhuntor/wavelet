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

