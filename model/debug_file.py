from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



if __name__ == '__main__':
    from utils import create_dataloader_from_arff
    from model.test import MultiModelParallelClassifier

    train_dataloader, train_mean, train_std = create_dataloader_from_arff(
        arff_file_path='/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/WormsTwoClass/WormsTwoClass_TRAIN.arff',
        batch_size=32,
        shuffle=True
    )
    # 测试集：使用训练集的参数进行标准化
    test_dataloader, _, _ = create_dataloader_from_arff(
        arff_file_path='/Users/hxh/PycharmProjects/wavelet/wavelet/Dataset/WormsTwoClass/WormsTwoClass_TEST.arff',
        batch_size=32,
        shuffle=False,
        mean=train_mean,
        std=train_std
    )

    model = MultiModelParallelClassifier(
        filter_lengths=[4, 8, 16, 32],
        levels=3,
        signal_length=900,
        num_classes=2,

        # ===== 所有特征开关 =====
        use_wavelet=True,  # ← 小波 on/off
        use_pca=True,  # ← PCA on/off
        use_stats=True,  # ← 统计特征 on/off
        use_fft=True,  # ← FFT on/off

        # ===== 训练参数 =====
        max_epochs=100,
        initial_lr=0.1,
        patience=10
    )
    model.train_models_parallel(train_dataloader, device='cpu')

    model.train_classifier(train_dataloader)

    # print('training data')
    # preds, labels = model.predict(train_dataloader)
    # cm = confusion_matrix(labels, preds)
    # # 方法 1: 打印数值
    # print("Confusion Matrix:")
    # print(cm)
    print('test')
    preds, labels = model.predict(test_dataloader)
    cm = confusion_matrix(labels, preds)

    # 方法 1: 打印数值
    print("Confusion Matrix:")
    print(cm)