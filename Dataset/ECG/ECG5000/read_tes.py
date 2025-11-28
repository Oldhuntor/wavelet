from sktime.datasets import load_from_arff_to_dataframe
import torch
import pandas as pd



df: pd.DataFrame = load_from_arff_to_dataframe('/Users/hxh/PycharmProjects/final_thesis/Dataset/ECG/ECG5000/ECG5000_TRAIN.arff')



# X, y = df
#
#
# print(X.shape)
# print(y.shape)
#
# X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
#
# print(X_tensor.shape)
# → (N, T, 1)