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

error = ['error', '_']
df = pd.read_csv('attendance_to_clean.csv', na_values=error)
for i, j in df.iterrows():
    try:
        a = float(j['BEGIN_HOUR'])
        if a > 18:
            df.loc[i, 'BEGIN_HOUR'] = np.NaN
    except:
        df.loc[i, 'BEGIN_HOUR'] = np.NaN
    try:
        a = datetime.datetime.strptime(j['DATE'], '%Y-%m-%d')
        if a < datetime.datetime(2022, 9, 1):
            df.loc[i, 'DATE'] = np.NaN
    except:
        df.loc[i, 'DATE'] = np.NaN
    try:
        a = float(j['TYPE'])
        df.loc[i, 'TYPE'] = np.NaN
    except:
        pass
    try:
        a = float(j['COUNT'])
        if a <= 2:
            df.loc[i, 'COUNT'] = float(j['COUNT'])  
        else:
            df.loc[i, 'COUNT'] = np.NaN
    except:
        df.loc[i, 'COUNT'] = np.NaN

    try:
        a = float(j['NAME_STUDENT'])
        df.loc[i, 'NAME_STUDENT'] = np.NaN
    except:
        pass
    

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df = df.sort_values(['NAME_STUDENT', 'DATE', 'BEGIN_HOUR', 'WEEK'])
df = df.reset_index(drop=True)
print(df)
#1
df = pd.read_parquet("Ex_1_price_to_clean_v3.parquet")  
  
for index, lines in df.iterrows():  
    try:  
        x = float(lines['Low'])  
    except:  
        df.loc[index, 'Low'] = np.nan  

for index, lines in df.iterrows():  
    df['Open'] = df['Open'].astype(float)  
    df['High'] = df['High'].astype(float)  
    df['Low'] = df['Low'].astype(float)  
    df['Close'] = df['Close'].astype(float)  
    df['Adj Close'] = df['Adj Close'].astype(float)  
    df['Volume'] = df['Volume'].astype(float)  

df.dropna(inplace = True)  
df = df.sort_values(by=['Date'])  
df.reset_index(drop = True, inplace = True)
#2
import numpy as np  
import sklearn.linear_model as skmod  
import matplotlib.pyplot as plt  
import pandas as pd  
import datetime  
  
df_origin = pd.read_parquet('Ex_2_price_to_plot_v2.parquet')  
  
df_select = df_origin[df_origin['Date'] > datetime.datetime(2009,8,18) and df_origin['Date'] > datetime.datetime(2009,8,6)]  
  
df = df_select[df_select['Date'] < datetime.datetime(2020,8,6)]  
  
df.reset_index(inplace=True, drop =True)  
     
df.plot(x = 'Low', y = 'Close')  
  
plt.show()
#4
features = [-7, -7, -9, -3, 8, -4, -5, 7, 8, -4, -4, -2, 4, -2, -4, 1, -10, -5, 1, -9] 
label = [-46.0, -75.0, -103.0, -46.0, 94.0, -50.0, -67.0, 56.0, 79.0, -46.0, -46.0, -24.0, 33.0, -16.0, -56.0, 3.0, -116.0, -48.0, 4.0, -113.0] 

input1 = np.array(features) 
input2 = np.array(label) 

input1 = input1.reshape(-1, 1) 
input2 = input2.reshape(-1, 1) 

#model = LinearRegression().fit(input1, input2) 

# w1 = model.coef_[0][0] 
# b = model.intercept_[0] 
# R2 = model.score(input1, input2)
#5
import numpy as np  
feature1 = [-16, -965, -813, -619, -334, 151, 74, 154, -269, -833, -976, -652, -713, -880, 37, -707, -352, 177, 179, 91, -702, -189, 308, -999, -660, 354, -209, 272, -73, -45]  
feature2 = [-3, -8, -7, 0, -9, 1, 0, -7, -9, -1, -6, 0, -9, -8, -3, -5, 2, 0, -7, -9, 3, 0, -5, -6, -5, -1, 3, 2, -3, -9]  
feature1 = np.array(feature1).reshape(-1,1)  
feature2 = np.array(feature2).reshape(-1,1)  
x = np.hstack([feature1, feature2])  
#x_train, x_test = sksel.train_test_split(x, train_size=.8, shuffle=False)
#
feature   = [-657, -254, -798, -237, -319, 170, -58, -651, 1, 29, -58, -738, 36, -98, 406, -842, -285, -963, -789, -326, -439, 413, 373, -512, -78, 110, -556, -386, -41, -112]
arr_f = np.array(feature).reshape(-1, 1)
#poly4 = skprepro.PolynomialFeatures(degree=4, include_bias=False)
#x_polymial = poly4.fit_transform(arr_f)   