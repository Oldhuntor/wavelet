from utils.ts_convertor import create_dataloader_from_arff
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type, Dict, Any, Callable
from loss.QMF import QMFLoss
from wavelets.wavelet_cache import create_dataloader_from_npz
import neptune
import os

def train_epoch(model, dataloader, criterion, optimizer, device):
    """在一个 epoch 上执行训练"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    count = 0
    for inputs, labels in dataloader:
        # print(count)
        count = count + 1
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
        # print(f"total samples: {total_samples}, correct predictions: {correct_predictions}, loss: {running_loss}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    """评估模型在测试集上的性能"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    count = 0
    with torch.no_grad():

        for inputs, labels in dataloader:
            # print(count)
            count += 1
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            # print(f"total samples: {total_samples}, correct predictions: {correct_predictions}, loss: {running_loss}")

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
    run: neptune.Run,
    custom_dataloader:bool,
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001
) -> nn.Module:
    """
    通用主函数：加载数据、使用传入的模型类和参数初始化模型，并执行训练循环。
    """
    #  0 记录模型参数
    run["params"] = model_params
    run["model_name"] = model_class.__name__

    # 1. 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 2. 数据加载和标准化 (假设 create_dataloader_from_arff 已在外部定义)
    # 训练集：计算并返回标准化参数

    if custom_dataloader:
        train_dataloader, train_mean, train_std = create_dataloader_from_npz(
            npz_file_path=train_path,
            batch_size=batch_size,
            shuffle=True,
        )

        test_dataloader, _, _ = create_dataloader_from_npz(
            npz_file_path=test_path,
            batch_size=batch_size,
            shuffle=False,
            mean=train_mean,
            std=train_std
        )

    else:
        train_dataloader, train_mean, train_std = create_dataloader_from_arff(
            arff_file_path=train_path,
            batch_size=batch_size,
            shuffle=True
        )
        # 测试集：使用训练集的参数进行标准化
        test_dataloader, _, _ = create_dataloader_from_arff(
            arff_file_path=test_path,
            batch_size=batch_size,
            shuffle=False,
            mean=train_mean,
            std=train_std
        )

    # 记录数据

    # -------------------------
    # ⭐ 1. 记录数据文件名
    # -------------------------
    train_name = os.path.basename(train_path)
    test_name = os.path.basename(test_path)

    run["data/train/file_name"] = train_name
    run["data/test/file_name"] = test_name

    # -------------------------
    # ⭐ 2. 记录数据形状
    # -------------------------
    # 获取 train 数据形状
    train_batch, train_label = next(iter(train_dataloader))
    run["data/train/input_shape"] = str(train_batch.shape)
    run["data/train/label_shape"] = str(train_label.shape)

    # 获取 test 数据形状
    test_batch, test_label = next(iter(test_dataloader))
    run["data/test/input_shape"] = str(test_batch.shape)
    run["data/test/label_shape"] = str(test_label.shape)
    train_count = len(train_dataloader.dataset)
    test_count = len(test_dataloader.dataset)
    run["data/train/count"] = train_count
    run["data/test/count"] = test_count
    # -------------------------
    # ⭐ 3. 记录标准化参数
    # -------------------------
    run["data/train/mean"] = train_mean
    run["data/train/std"] = train_std

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

        run["train/loss"].append(train_loss)
        run["train/acc"].append(train_acc)
        run["test/loss"].append(test_loss)
        run["test/acc"].append(test_acc)

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

# TRAIN_FILE = DATA_PATH +  'Computers/Computers_TRAIN.arff'
# TEST_FILE = DATA_PATH + 'Computers/Computers_TEST.arff'

DATA_NAME = 'AbnormalHeartbeat'

DATA_TYPE = 'npz'  # npz or arff

TRAIN_FILE = DATA_PATH +  f'{DATA_NAME}/{DATA_NAME}_TRAIN.{DATA_TYPE}'
TEST_FILE = DATA_PATH + f'{DATA_NAME}/{DATA_NAME}_TEST.{DATA_TYPE}'



# 根据您的参数设置:
C = 1         # 通道数
L = 18530       # 序列长度
K = 2        # 类别数 (请注意，如果 ECG5000 实际上是 5 个类别，K 应该设置为 5)

# 模型和训练参数
INPUT_DIM = C * L # 140
NUM_C = K
EPOCHS = 50
LR = 0.001


if __name__ == '__main__':
    from model.wavelet_feature_extractor import TimeSeriesMLP
    from model.cwt_matrix import generate_adaptive_scales


    #
    os.environ["NEPTUNE_LOGGER_LEVEL"] = "DEBUG"

    run = neptune.init_run(
        project="casestudy",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        mode="debug"
    )


    scales = generate_adaptive_scales(L)
    # scales = np.arange(10, 128)
    fs = 1


    WMHC_params = {
        'scales' : scales,
        'fs' : fs,
        'trim_ratio' : 0.1,
        'num_heads' : 4,
        'head_dim' : 64,
        'num_classes' : NUM_C
    }

    DWT_params = {
    'input_length' : L,
    'levels' : 3,
    'hidden_dim' : 1024,
    'output_dim' : NUM_C,
    }

    MLP_params = {
        'input_dim' : 148240, # 148240,18530
        'hidden_size' : 512,
        'num_classes' : NUM_C,
    }

    # final_model = main_train_and_test_generic(
    #     model_class=DWT_MLP,
    #     model_params=DWT_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch,
    #     custom_dataloader=False,
    #     evaluator=evaluate_model,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    # )
    #
    final_model = main_train_and_test_generic(
        model_class=TimeSeriesMLP,
        model_params=MLP_params,
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        trainer=train_epoch,
        evaluator=evaluate_model,
        custom_dataloader=True,
        num_epochs=EPOCHS,
        learning_rate=LR,
        batch_size=32,
        run=run,
    )


    # final_model = main_train_and_test_generic(
    #     model_class=WaveletMultiHeadClassifier,
    #     model_params=WMHC_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch,
    #     evaluator=evaluate_model,
    #     custom_dataloader=True,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #

    run.stop()
