import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sksel
import sklearn.preprocessing as skprepro
import sklearn.linear_model as skmod

df = pd.read_parquet(r"Data_streamers.parquet")
x = df.to_numpy().reshape(-1,1)
print(x)