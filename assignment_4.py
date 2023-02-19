'''
The objective of this assignment is to clean the csv file of the attendance.
The path to the csv file is "attendance_to_clean.csv"
You can find it in the instruction folder.
List of installed and authorized packages :
    - csv
    - pandas
    - datetime
    - numpy
You cannot use other packages than the listed ones (except built-in default package in python).
You can write you code after this comment :
'''

#Your code here:
import datetime
import pandas as pd
import csv
import numpy as np

a = ['error', '_']
df = pd.read_csv('attendance_to_clean.csv', na_values=a)
for i, l in df.iterrows():
    try:
        b = datetime.datetime.strptime(l['DATE'], '%Y-%m-%d')
    except:
        # print(l['DATE'])
        df.loc[i, 'DATE'] = np.nan
print(df.isnull().sum())
# print(df)