from ts_convertor import create_dataloader_from_arff
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type, Dict, Any, Callable
from loss.QMF import QMFLoss


def train_epoch(model, dataloader, criterion, optimizer, device):
    """在一个 epoch 上执行训练"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    """评估模型在测试集上的性能"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def train_epoch_LWT(model , dataloader, criterion, optimizer, device):
    """在一个 epoch 上执行训练"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        main_loss = criterion(outputs, labels)
        criterion_qmf = QMFLoss()
        qmf_loss = criterion_qmf(model.wavelet_layer.h, model.wavelet_layer.g)

        loss = main_loss + qmf_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model_LWT(model, dataloader, criterion, device):
    """评估模型在测试集上的性能"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            main_loss = criterion(outputs, labels)
            criterion_qmf = QMFLoss()
            qmf_loss = criterion_qmf(model.wavelet_layer.h, model.wavelet_layer.g)

            loss = main_loss + qmf_loss

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def main_train_and_test_generic(
    model_class: Type[nn.Module],         # 模型的类 (e.g., TimeSeriesMLP)
    model_params: Dict[str, Any],         # 模型初始化参数字典 (e.g., {'input_dim': 140, 'num_classes': 5})
    train_path: str,
    test_path: str,
    trainer: Callable,
    evaluator: Callable,
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001
) -> nn.Module:
    """
    通用主函数：加载数据、使用传入的模型类和参数初始化模型，并执行训练循环。
    """
    # 1. 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 2. 数据加载和标准化 (假设 create_dataloader_from_arff 已在外部定义)
    # 训练集：计算并返回标准化参数
    train_dataloader, train_mean, train_std = create_dataloader_from_arff(
        arff_file_path=train_path, batch_size=batch_size, shuffle=True
    )
    # 测试集：使用训练集的参数进行标准化
    test_dataloader, _, _ = create_dataloader_from_arff(
        arff_file_path=test_path, batch_size=batch_size, shuffle=False,
        mean=train_mean, std=train_std
    )

    # 3. 初始化模型、损失函数和优化器
    # 使用传入的模型类和参数字典进行初始化
    # **model_params 将字典展开为关键字参数
    try:
        model = model_class(**model_params).to(device)
    except TypeError as e:
        print(f"\n--- 模型初始化错误 ---")
        print(f"请检查模型 {model_class.__name__} 的 __init__ 函数签名是否与 model_params 匹配。")
        print(f"错误详情: {e}")
        raise

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_test_acc = 0.0

    # 4. 训练循环
    print(f"\n--- 开始训练 {model_class.__name__} 模型 ---")
    for epoch in range(num_epochs):

        # 训练
        train_loss, train_acc = trainer(model, train_dataloader, criterion, optimizer, device)

        # 评估
        test_loss, test_acc = evaluator(model, test_dataloader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.2f}%")

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # torch.save(model.state_dict(), f'best_{model_class.__name__}.pth')

    print("\n--- 训练完成 ---")
    print(f"最终最佳测试集准确率: {best_test_acc * 100:.2f}%")
    return model


DATA_PATH = '/Users/hxh/PycharmProjects/final_thesis/Dataset/'

# 您的路径示例 (请替换为您的实际路径):
# TRAIN_FILE = DATA_PATH +  'StarLightCurves/StarLightCurves_TRAIN.arff'
# TEST_FILE = DATA_PATH + 'StarLightCurves/StarLightCurves_TEST.arff'

# TRAIN_FILE = DATA_PATH +  'Earthquakes/Earthquakes_TRAIN.ts'
# TEST_FILE = DATA_PATH + 'Earthquakes/Earthquakes_TEST.ts'

# TRAIN_FILE = DATA_PATH +  'BinaryHeartbeat/BinaryHeartbeat_TRAIN.arff'
# TEST_FILE = DATA_PATH + 'BinaryHeartbeat/BinaryHeartbeat_TEST.arff'

# TRAIN_FILE = DATA_PATH +  'WormsTwoClass/WormsTwoClass_TRAIN.arff'
# TEST_FILE = DATA_PATH + 'WormsTwoClass/WormsTwoClass_TEST.arff'

# TRAIN_FILE = DATA_PATH +  'PowerCons/PowerCons_TRAIN.arff'
# TEST_FILE = DATA_PATH + 'PowerCons/PowerCons_TEST.arff'

TRAIN_FILE = DATA_PATH +  'Computers/Computers_TRAIN.arff'
TEST_FILE = DATA_PATH + 'Computers/Computers_TEST.arff'

# TRAIN_FILE = DATA_PATH +  'AbnormalHeartbeat/AbnormalHeartbeat_TRAIN.arff'
# TEST_FILE = DATA_PATH + 'AbnormalHeartbeat/AbnormalHeartbeat_TEST.arff'


# 根据您的参数设置:
C = 1         # 通道数
L = 720       # 序列长度
K = 2        # 类别数 (请注意，如果 ECG5000 实际上是 5 个类别，K 应该设置为 5)

# 模型和训练参数
INPUT_DIM = C * L # 140
NUM_C = K
EPOCHS = 50
LR = 0.001


if __name__ == '__main__':
    from model.wavelet_feature_extractor import DWT_MLP, TimeSeriesMLP
    from model.learnable_wavelet import LWT

    DWT_params = {
    'input_length' : L,
    'levels' : 3,
    'hidden_dim' : 32,
    'output_dim' : NUM_C,
    }

    MLP_params = {
        'input_dim' : INPUT_DIM,
        'hidden_size' : 64,
        'num_classes' : NUM_C,
    }

    final_model = main_train_and_test_generic(
        model_class=DWT_MLP,
        model_params=DWT_params,
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        trainer=train_epoch,
        evaluator=evaluate_model,
        num_epochs=EPOCHS,
        learning_rate=LR,
        batch_size=256,
    )

    final_model = main_train_and_test_generic(
        model_class=TimeSeriesMLP,
        model_params=MLP_params,
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        trainer=train_epoch,
        evaluator=evaluate_model,
        num_epochs=EPOCHS,
        learning_rate=LR,
        batch_size=256,
    )
