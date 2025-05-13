import pandas as pd
import numpy as np


train_data = pd.read_csv('X_trainToronto.csv')
test_data = pd.read_csv('X_testToronto.csv')

print("Train shape:", train_data.shape)
train_data.head()
