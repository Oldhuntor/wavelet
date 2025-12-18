from utils import create_dataloader_from_arff
from model.paralle_wave import SimpleWaveletClassifier

if __name__ == '__main__':

    train_dataloader, train_mean, train_std = create_dataloader_from_arff(
        arff_file_path='/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/Computers/Computers_TRAIN.arff',
        batch_size=32,
        shuffle=True
    )
    # 测试集：使用训练集的参数进行标准化
    test_dataloader, _, _ = create_dataloader_from_arff(
        arff_file_path='/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/Computers/Computers_TEST.arff',
        batch_size=32,
        shuffle=False,
        mean=train_mean,
        std=train_std
    )
    # 创建
    model = SimpleWaveletClassifier(
        filter_lengths=[4, 8, 16, 32],
        levels=3,
        signal_length=720,
        num_classes=2
    )

    # 训练
    model.train(train_dataloader)

    # 预测
    preds, labels = model.predict(test_dataloader)

    # 评估
    from sklearn.metrics import accuracy_score, confusion_matrix

    cm = confusion_matrix(labels, preds)
    print(cm)
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc:.3f}")