import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src import config

df_train = pd.read_csv(config.TRAIN_DATA_PATH)
print(df_train.head())

df_train