from torch.utils.data import DataLoader

from utils.ts_convertor import create_dataloader_from_arff
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type, Dict, Any, Callable
from loss.QMF import QMFLoss
from wavelets.wavelet_cache import create_dataloader_from_npz
import neptune
import os
from utils import get_data_path
from utils import DATA_PATH, DUAL_FEATURES_MODELS, DATA_NAMES, DATA_INFO

DATA_NAME = DATA_NAMES[11]
MARGIN = 0.5
DATA_TYPE = 'arff'  # pt, npz or arff

TRAIN_FILE, TEST_FILE = get_data_path(DATA_PATH, DATA_NAME, DATA_TYPE)

# 根据您的参数设置:
C = 1  # 通道数
L = DATA_INFO[DATA_NAME]['SEQUENCE_LENGTH']  # 序列长度
K = DATA_INFO[DATA_NAME]['NUM_CLASSES']  # 类别数 (请注意，如果 ECG5000 实际上是 5 个类别，K 应该设置为 5)

# 模型和训练参数
INPUT_DIM = C * L  # 140
NUM_C = K
EPOCHS = 2000
# EPOCHS = 100

LR = 0.001


def check_specific_module(model, module_name="classifier"):
    """检查特定模块的参数状态"""
    print(f"\n{'=' * 70}")
    print(f"Checking {module_name} parameters")
    print(f"{'=' * 70}")

    # 获取模块
    if hasattr(model, module_name):
        module = getattr(model, module_name)
    else:
        print(f"Module '{module_name}' not found!")
        return

    # 方法1: 使用 parameters()
    print(f"\n方法1: Using module.parameters()")
    for i, param in enumerate(module.parameters()):
        print(f"  Param {i}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}")

    # 方法2: 使用 named_parameters()
    print(f"\n方法2: Using module.named_parameters()")
    for name, param in module.named_parameters():
        print(f"  {name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}")

    # 方法3: 检查是否全部冻结
    print(f"\n方法3: Check if all frozen")
    all_frozen = all(not p.requires_grad for p in module.parameters())
    all_trainable = all(p.requires_grad for p in module.parameters())

    if all_frozen:
        print(f"  ✗ ALL parameters in {module_name} are FROZEN")
    elif all_trainable:
        print(f"  ✓ ALL parameters in {module_name} are TRAINABLE")
    else:
        print(f"  ⚠ MIXED: Some frozen, some trainable")

def create_optimizers_for_adaptive_filters(model, lr_wavelet=0.01, lr_classifier=0.0001):
    """为你的使用场景创建优化器

    Stage 1 (reconstruction_loss >= MARGIN):
        - 只训练 low_pass 和 high_pass filters
        - 不训练任何其他参数

    Stage 2 (reconstruction_loss < MARGIN):
        - filters 永远不再训练！
        - 训练除了filters之外的所有参数:
          * classifier 所有层
          * learnable activation (threshold_params)
    """

    # Optimizer 1: 只包含两个filters
    wavelet_params = [
        model.wavelet.low_pass,
        # model.wavelet.high_pass
    ]
    optimizer_wavelet = torch.optim.Adam(wavelet_params, lr=lr_wavelet)

    # Optimizer 2: 所有非filter的参数
    non_filter_params = []

    # 添加classifier的所有参数
    non_filter_params.extend(model.classifier.parameters())

    # 添加learnable activation参数
    if model.wavelet.use_learnable_activation:
        non_filter_params.extend(model.wavelet.threshold_params)

    optimizer_classifier = torch.optim.Adam(non_filter_params, lr=lr_classifier)

    print("\n" + "=" * 70)
    print("创建的优化器:")
    print("=" * 70)
    print(f"optimizer_wavelet (只在Stage 1使用):")
    print(f"  - low_pass filter (shape={tuple(model.wavelet.low_pass.shape)})")
    # print(f"  - high_pass filter (shape={tuple(model.wavelet.high_pass.shape)})")
    print(f"  总共 {sum(p.numel() for p in wavelet_params):,} 参数")

    classifier_param_count = sum(p.numel() for p in model.classifier.parameters())
    print(f"\noptimizer_classifier (只在Stage 2使用):")
    print(f"  - classifier层: {classifier_param_count:,} 参数")

    if model.wavelet.use_learnable_activation:
        activation_param_count = sum(p.numel() for p in model.wavelet.threshold_params)
        print(f"  - learnable activation: {activation_param_count:,} 参数")
        print(f"  总共 {classifier_param_count + activation_param_count:,} 参数")
    else:
        print(f"  总共 {classifier_param_count:,} 参数")

    print("\n重要: filters一旦在Stage 1训练好，进入Stage 2后永远不再更新！")
    print("=" * 70 + "\n")

    return optimizer_wavelet, optimizer_classifier

def train_epoch_ad(model, dataloader, criterion, optimizer_wavelet,
                          optimizer_classifier, device, filter_trained = False):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    stage1_batches = 0  # 统计Stage 1的batch数
    stage2_batches = 0  # 统计Stage 2的batch数

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.squeeze_()

        # 前向传播
        logits, coeffs = model(inputs)
        classification_loss = criterion(logits, labels)
        reconstruction_loss = model.compute_reconstruction_loss(inputs, coeffs)

        if not filter_trained:
            # ========== Stage 1: 训练filters ==========
            optimizer_wavelet.zero_grad()
            total_loss = reconstruction_loss
            total_loss.backward()
            optimizer_wavelet.step()
            stage1_batches += 1

        else:
            # ========== Stage 2: 训练classifier ==========
            optimizer_classifier.zero_grad()
            total_loss = classification_loss + reconstruction_loss
            total_loss.backward()
            optimizer_classifier.step()
            stage2_batches += 1

        # 统计
        running_loss += total_loss.item() * inputs.size(0)
        _, predicted = torch.max(logits.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    # 打印这个epoch的stage分布
    print(f"  Stage 1 batches: {stage1_batches}, Stage 2 batches: {stage2_batches}")

    return epoch_loss, epoch_acc, reconstruction_loss

def evaluate_model_ad(model, dataloader, criterion, device):
    """测试函数"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.squeeze_()

            logits, coeffs = model(inputs)
            classification_loss = criterion(logits, labels)
            reconstruction_loss = model.compute_reconstruction_loss(inputs, coeffs)
            total_loss = classification_loss + reconstruction_loss

            running_loss += total_loss.item() * inputs.size(0)
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def train_epoch_wcnn(model, dataloader, criterion, optimizer, device):
    """在一个 epoch 上执行训练"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        cls_loss = criterion(outputs['logits'], labels)
        orth_loss = outputs["orth_loss"]
        loss = cls_loss + orth_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs['logits'].data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model_wcnn(model, dataloader, criterion, device):
    """评估模型在测试集上的性能"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            cls_loss = criterion(outputs['logits'], labels)
            orth_loss = outputs["orth_loss"]
            loss =  cls_loss + orth_loss
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs['logits'].data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def train_epoch_DUAL(model, dataloader, criterion, optimizer, device):
    """在一个 epoch 上执行训练"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    count = 0
    for amp, pha, labels in dataloader:
        # print(count)
        count = count + 1

        optimizer.zero_grad()
        outputs = model(amp, pha)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * amp.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        # print(f"total samples: {total_samples}, correct predictions: {correct_predictions}, loss: {running_loss}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model_DUAL(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    count = 0
    with torch.no_grad():

        for amp, pha, labels in dataloader:
            # print(count)
            count += 1
            outputs = model(amp, pha)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * amp.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            # print(f"total samples: {total_samples}, correct predictions: {correct_predictions}, loss: {running_loss}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def train_epoch(model, dataloader, criterion, optimizer, device):
    """在一个 epoch 上执行训练"""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    count = 0
    for inputs, labels in dataloader:
        # print(count)
        labels.squeeze_()
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
    custom_dataloader: tuple = (),
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    verbose: bool = False,
):
    """
    通用主函数：加载数据、使用传入的模型类和参数初始化模型，并执行训练循环。
    """
    #  0 记录模型参数
    run["params"] = model_params
    run["model_name"] = model_class.__name__
    model_name = model_class.__name__

    # 1. 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 2. 数据加载和标准化 (假设 create_dataloader_from_arff 已在外部定义)
    # 训练集：计算并返回标准化参数
    train_mean, train_std = None, None
    if not custom_dataloader:
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
    else:
        train_dataloader, test_dataloader = custom_dataloader

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
    if model_name in DUAL_FEATURES_MODELS:
        feature1, feature2, train_label = next(iter(train_dataloader))
        run["data/train/input_shape"] = str(feature1.shape)
        run["data/train/label_shape"] = str(train_label.shape)

        feature1, feature2, test_label = next(iter(test_dataloader))
        run["data/test/input_shape"] = str(feature1.shape)
        run["data/test/label_shape"] = str(test_label.shape)
        train_count = len(train_dataloader.dataset)
        test_count = len(test_dataloader.dataset)
        run["data/train/count"] = train_count
        run["data/test/count"] = test_count

    else:
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
    if train_mean or train_std:
        run["data/train/mean"] = train_mean
        run["data/train/std"] = train_std

    # 3. 初始化模型、损失函数和优化器
    # 使用传入的模型类和参数字典进行初始化
    # **model_params 将字典展开为关键字参数
    try:
        model = model_class(**model_params).to(device)
        # print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
        run["total parameters"] = sum(p.numel() for p in model.parameters())
    except TypeError as e:
        print(f"\n--- 模型初始化错误 ---")
        print(f"请检查模型 {model_class.__name__} 的 __init__ 函数签名是否与 model_params 匹配。")
        print(f"错误详情: {e}")
        raise

    best_test_acc = 0.0

    if model_name in ['WaveletClassifier','QMFWaveletClassifier']:
        optimizer_wavelet, optimizer_classifier = create_optimizers_for_adaptive_filters(
            model, lr_wavelet=0.01, lr_classifier=0.0001
        )
        criterion = nn.CrossEntropyLoss()
        print(f"MARGIN = 0.1 (reconstruction_loss >= 0.1 时训练filters)\n")
        stage_1_trained = False
        # 模拟训练几个epoch
        recon_loss = 1

        for epoch in range(num_epochs):
            if recon_loss <= MARGIN:
                stage_1_trained = True

            train_loss, train_acc, recon_loss = train_epoch_ad(model, train_dataloader, criterion, optimizer_wavelet,
                               optimizer_classifier, device, stage_1_trained)

            test_loss, test_acc = evaluate_model_ad(model, test_dataloader, criterion, device)
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.2f}%")

            run["train/loss"].append(train_loss)
            run["train/acc"].append(train_acc)
            run["test/loss"].append(test_loss)
            run["test/acc"].append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc

        print("\n--- 训练完成 ---")
        print(f"最终最佳测试集准确率: {best_test_acc * 100:.2f}%")
        return model


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # 4. 训练循环
    print(f"\n--- 开始训练 {model_class.__name__} 模型 ---")
    for epoch in range(num_epochs):

        # 训练
        train_loss, train_acc = trainer(model, train_dataloader, criterion, optimizer, device)

        # 评估
        test_loss, test_acc = evaluator(model, test_dataloader, criterion, device)

        if verbose:
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
    print(f"model: {model_name}, best accuracy : {best_test_acc * 100:.2f}%")
    return model, best_test_acc


if __name__ == '__main__':
    from model.baseline import TimeSeriesMLP
    from model.CWT import generate_adaptive_scales
    from model.DWT import DWT_MLP
    from model.learnable_wavelet import LWT
    from model.baseline import DualFeatureMLP
    from model.MRA import MorletDataset, load_morlet_pt, DualFeatureMRAClassifier
    from model.wavelet_cnn import WaveletLikeClassifier
    from utils import load_mortlet_pt_dataloader
    from model.wavelet_cls import WaveletCNN
    from model.test import MRATimeSeriesClassifier
    from model.adaptive_filters import WaveletClassifier
    from model.qmf_wavelet import QMFWaveletClassifier
    from model.multi_channel_DWT import MultiWaveletClassifier

    os.environ["NEPTUNE_LOGGER_LEVEL"] = "DEBUG"

    run = neptune.init_run(
        project="casestudy",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        # mode="debug"
    )

    train_ds, test_ds = load_mortlet_pt_dataloader(DATA_NAME)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    seq_len = train_ds.phase.shape[-1]
    ch = train_ds.phase.shape[1]
    num_classes = len(set(train_ds.labels))

    scales = generate_adaptive_scales(L)
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
    'hidden_dim' : 64,
    'output_dim' : NUM_C,
    }

    MLP_params = {
        'input_dim' : L, # 148240,18530
        'hidden_size' : 64,
        'num_classes' : NUM_C,
    }


    LWT_params = {
        'input_length' : L,
        'levels' : 3,
        'hidden_dim': 64,
        'output_dim' : NUM_C
    }


    MRA_params = {
        "num_classes": num_classes,
    }


    DUAL_MLP_params = {
        'input_dim_per_feature' : seq_len*ch,
        'num_classes' : num_classes,
        'hidden_size' : 64
    }

    WAVELET_CNN_params= {
        'input_channels' : 1,
        'input_length' : L,
        'levels' : 2,
        'n_classes' : NUM_C,
        'orth_weight' : 1e-3
    }

    WaveletCNN2_params= {
        'input_channels' : 1,
        'n_classes' : NUM_C
    }

    test_params = {
        'input_channels' : 1,
        'num_classes' : NUM_C,
    }

    test_params2 = {
        'filter_length': 8,
        'levels': 3,
        'signal_length': L,
        'num_classes': NUM_C,
        'hidden_dim' : 64,
        'init_type' : 'random',
        'use_frequency_constraint' : True,
        'use_learnable_activation' : True
    }

    test_params3 = {
        'filter_length': 8,
        'levels': 3,
        'signal_length': L,
        'num_classes': NUM_C,
        'hidden_dim' : 64,
        'num_wavelets': 4,
        'init_type' : 'random',
        # 'use_frequency_constraint' : True,
        # 'use_learnable_activation' : True
    }

    final_model, _ = main_train_and_test_generic(
        model_class=MultiWaveletClassifier,
        model_params=test_params3,
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        trainer=train_epoch,
        custom_dataloader=(),
        evaluator=evaluate_model,
        num_epochs=EPOCHS,
        learning_rate=LR,
        batch_size=32,
        run=run,
    )

    # final_model = main_train_and_test_generic(
    #     model_class=QMFWaveletClassifier,
    #     model_params=test_params2,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch_ad,
    #     custom_dataloader=(),
    #     evaluator=evaluate_model_ad,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )


    # final_model = main_train_and_test_generic(
    #     model_class=WaveletClassifier,
    #     model_params=test_params2,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch_ad,
    #     custom_dataloader=(),
    #     evaluator=evaluate_model_ad,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )

    # final_model = main_train_and_test_generic(
    #     model_class=MRATimeSeriesClassifier,
    #     model_params=test_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch,
    #     custom_dataloader=(),
    #     evaluator=evaluate_model,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )

    # final_model = main_train_and_test_generic(
    #     model_class=DWT_MLP,
    #     model_params=DWT_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch,
    #     custom_dataloader=(),
    #     evaluator=evaluate_model,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )

    # final_model = main_train_and_test_generic(
    #     model_class=LWT,
    #     model_params=LWT_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch_LWT,
    #     custom_dataloader=(),
    #     evaluator=evaluate_model_LWT,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )


    # final_model = main_train_and_test_generic(
    #     model_class=TimeSeriesMLP,
    #     model_params=MLP_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch,
    #     evaluator=evaluate_model,
    #     custom_dataloader=(),
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )


    # final_model = main_train_and_test_generic(
    #     model_class=DualFeatureMRAClassifier,
    #     model_params=MRA_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch_DUAL,
    #     evaluator=evaluate_model_DUAL,
    #     custom_dataloader=(train_loader, test_loader),
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )


    # final_model = main_train_and_test_generic(
    #     model_class=DualFeatureMLP,
    #     model_params=DUAL_MLP_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch_DUAL,
    #     custom_dataloader=(train_loader, test_loader),
    #     evaluator=evaluate_model_DUAL,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )
    #
    # final_model = main_train_and_test_generic(
    #     model_class=WaveletCNN,
    #     model_params=WaveletCNN2_params,
    #     train_path=TRAIN_FILE,
    #     test_path=TEST_FILE,
    #     trainer=train_epoch,
    #     evaluator=evaluate_model,
    #     num_epochs=EPOCHS,
    #     learning_rate=LR,
    #     batch_size=32,
    #     run=run,
    # )



    run.stop()
