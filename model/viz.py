import torch
import torch.nn as nn


# ============ 你的模型定义 ============
class Conv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class WaveletCNN(nn.Module):
    def __init__(self, input_channels=1, n_classes=2):
        super().__init__()
        self.low_pass_1 = Conv1DBlock(in_ch=input_channels, out_ch=3, kernel_size=2)
        self.high_pass_1 = Conv1DBlock(in_ch=input_channels, out_ch=3, kernel_size=2)
        self.low_pass_2 = Conv1DBlock(in_ch=3, out_ch=3, kernel_size=2)
        self.high_pass_2 = Conv1DBlock(in_ch=3, out_ch=3, kernel_size=2)
        self.fc = nn.LazyLinear(n_classes)

    def forward(self, x):
        B = x.size(0)
        L1 = self.low_pass_1(x)
        H1 = self.high_pass_1(x)
        L2 = self.low_pass_2(H1)
        H2 = self.high_pass_2(H1)

        feat_L1 = L1.view(B, -1)
        feat_L2 = L2.view(B, -1)
        feat_H2 = H2.view(B, -1)
        feat = torch.cat([feat_L1, feat_L2, feat_H2], dim=1)
        logits = self.fc(feat)
        return logits


# ============ 导出模型 ============
if __name__ == "__main__":
    print("开始导出WaveletCNN模型...")

    # 1. 创建模型实例
    model = WaveletCNN(input_channels=1, n_classes=2)
    model.eval()  # 重要：设置为评估模式

    # 2. 创建示例输入
    # 注意：由于使用 nn.LazyLinear，需要先进行一次前向传播来初始化它
    dummy_input = torch.randn(1, 1, 100)  # (batch=1, channels=1, length=100)

    # 初始化LazyLinear
    with torch.no_grad():
        _ = model(dummy_input)

    print(f"✅ 模型创建完成")
    print(f"📊 输入形状: {dummy_input.shape}")

    # 3. 导出为ONNX格式
    try:
        onnx_filename = "wavelet_cnn_model.onnx"

        torch.onnx.export(
            model,  # 要导出的模型
            dummy_input,  # 模型输入示例
            onnx_filename,  # 保存的文件名
            input_names=["input"],  # 输入节点名称
            output_names=["output"],  # 输出节点名称
            opset_version=14,  # ONNX版本
            dynamic_axes={  # 指定动态维度（支持变长输入）
                'input': {0: 'batch_size', 2: 'seq_length'},  # batch和序列长度可变
                'output': {0: 'batch_size'}
            },
            verbose=False,
            export_params=True  # 包含模型参数
        )

        import os

        print(f"✅ 导出成功: {onnx_filename}")
        print(f"📁 文件大小: {os.path.getsize(onnx_filename) / 1024:.1f} KB")

        # 4. 显示模型信息
        print(f"\n📋 模型结构摘要:")
        print(f"  输入: (batch_size, 1, seq_length)")
        print(f"  卷积核大小: 2")
        print(f"  第一级滤波器: 1 → 3 通道")
        print(f"  第二级滤波器: 3 → 3 通道")
        print(f"  特征拼接: 3个特征图 × 各 (3 × L_in) → 总维度 9 × L_in")
        print(f"  输出: (batch_size, 2)")

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback

        traceback.print_exc()