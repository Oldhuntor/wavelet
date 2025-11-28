from sktime.datasets import load_from_arff_to_dataframe
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

df: pd.DataFrame = load_from_arff_to_dataframe('/Users/hxh/PycharmProjects/final_thesis/Dataset/ECG/ECG5000/ECG5000_TRAIN.arff')

X, y = df

X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)

