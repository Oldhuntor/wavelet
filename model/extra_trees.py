from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


idx = 0
for inputs, labels in train_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    inputs.squeeze_()

    logits, coeffs = model(inputs)
    feat_batch = torch.cat(coeffs, dim=1)
    if idx == 0:
        all_features = feat_batch.detach().cpu().numpy()
        all_labels = labels.detach().cpu().numpy()
        idx += 1
        continue
    all_features = np.vstack([all_features, feat_batch.detach().cpu().numpy()])
    all_labels = np.concatenate([all_labels, labels.detach().cpu().numpy()])
    idx += 1

classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
classifier.fit(all_features, all_labels)
prediction = classifier.predict(all_features)

cm = confusion_matrix(all_labels, prediction)

# 方法 1: 打印数值
print("Confusion Matrix:")
print(cm)

# 方法 2: 可视化（推荐）
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot()
plt.show()


idx = 0
for inputs, labels in test_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    inputs.squeeze_()

    logits, coeffs = model(inputs)
    feat_batch = torch.cat(coeffs, dim=1)
    if idx == 0:
        all_features_test = feat_batch.detach().cpu().numpy()
        all_labels_test = labels.detach().cpu().numpy()
        idx += 1
        continue
    all_features_test = np.vstack([all_features_test, feat_batch.detach().cpu().numpy()])
    all_labels_test = np.concatenate([all_labels_test, labels.detach().cpu().numpy()])
    idx += 1



prediction = classifier.predict(all_features_test)

cm = confusion_matrix(all_labels_test, prediction)

# 方法 1: 打印数值
print("Confusion Matrix:")
print(cm)