import pandas as pd
import numpy as np
import sklearn.linear_model as skmod
import sklearn.preprocessing as skprepro
import sklearn.model_selection as sksel

df = pd.read_parquet('Data_to_analyse.parquet', engine='pyarrow')
print(df)
