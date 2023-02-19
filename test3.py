'''  
-------------------------ENDTERM EXAM-----------------  
DO NOT DELETE THE FOLLOWING CODE  
'''  
import sys  
try:  
    input1 = sys.argv[1]  
except:  
    pass  
'''  
In the following file, do not delete anything (comments, code, ...). Just add you code in every part (one per exercise).  
Use my variable for input (if there is any), use my printing for output (if there is any).  
You can upload your code to codepost.io to check the tests. A sucess in one test doesn't always mean than your exercise is correct,  
a fail doesn't always mean that your exercise is wrong. I will check all codes.  
At the end of exam, you should upload the last version of your code to codepost.io or to the online folder on Teams.  
The only authorized packages are:  
- pandas  
- pyarrow  
- fastparquet  
- numpy  
- sklearn  
- matplotlib  
- datetime  
  
'''  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import sklearn.model_selection as sksel  
import sklearn.preprocessing as skprepro  
import sklearn.linear_model as skmod  
import sklearn.preprocessing as skprepro  
import datetime  
if input1 == '4':  
# ----------------------EXERCISE 4 - Machine Learning II--------------------------------------  
# Instructions:  
# You have a model trained with one feature : feature1. Make a scatter plot with the feature as x axis and the label as y axis.  
# Add on the same plot, the predictions of the model (line plot).You can find the required plot in the file Ex_4_plot_V4.png  
 import sklearn.linear_model as skmod  
 import numpy as np  
 x = np.array([-1, -8, 8, -5, 9, 1, 1, 8, -3, 9, 6, 1, -3, -7, 5, -5, 3, -1, -8, -6]).reshape(-1,1)  
 y = np.array([9.0, 66.0, -35.0, 31.0, -75.0, -5.0, -7.0, -58.0, 23.0, -50.0, -34.0, -8.0, 23.0, 32.0, -48.0, 46.0, -30.0, 7.0, 60.0, 30.0]).reshape(-1,1)  
 model = skmod.LinearRegression().fit(x,y)  
 plt.scatter(x,y)  
 x = np.array([-7.5,7.5]).reshape(-1,1)  
 plt.plot(x, model.predict(x))  
 plt.show()  
# ----------------------End of EXERCISE 4 --------------------------------------  
  
  
elif input1 == '5':  
# ----------------------EXERCISE 5 - Machine Learning III--------------------------------------  
# Instructions:  
# You have two features : feature1 and feature2. You objective is to make a two columns matrix and  
# to separate them into a train and a test set. The size of train set is 80 % of the original matrix.  
# Remember to use shuffle = False in the train_test_split function of the scikit-learn package.  
# At the end print only the test arrays.   
# You can find the desired matrix in the document Ex_5_x_tests_matrix_V4.txt  
 import numpy as np  
 feature1 = [-16, -965, -813, -619, -334, 151, 74, 154, -269, -833, -976, -652, -713, -880, 37, -707, -352, 177, 179, 91, -702, -189, 308, -999, -660, 354, -209, 272, -73, -45]  
 feature2 = [-3, -8, -7, 0, -9, 1, 0, -7, -9, -1, -6, 0, -9, -8, -3, -5, 2, 0, -7, -9, 3, 0, -5, -6, -5, -1, 3, 2, -3, -9]  
 feature1 = np.array(feature1).reshape(-1,1)  
 feature2 = np.array(feature2).reshape(-1,1)  
 x = np.hstack([feature1, feature2])  
 x_train, x_test = sksel.train_test_split(x, train_size=.8, shuffle=False)  
  
# Here is the print instructions to print test arrays.  
 print(x_test)  
  
# ----------------------End of EXERCISE 5 --------------------------------------  
'''  
-------------------------ENDTERM EXAM-----------------  
DO NOT DELETE THE FOLLOWING CODE  
'''  
import sys  
try:  
    input1 = sys.argv[1]  
except:  
    pass  
'''  
In the following file, do not delete anything (comments, code, ...). Just add you code in every part (one per exercise).  
Use my variable for input (if there is any), use my printing for output (if there is any).  
You can upload your code to codepost.io to check the tests. A sucess in one test doesn't always mean than your exercise is correct,  
a fail doesn't always mean that your exercise is wrong. I will check all codes.  
At the end of exam, you should upload the last version of your code to codepost.io or to the online folder on Teams.  
The only authorized
Ernatt, [9 дек. 2022 в 11:33]
kages are:  
- pandas  
- pyarrow  
- fastparquet  
- numpy  
- sklearn  
- matplotlib  
- 
datetime  
  
'''  
if input1 == '2':  
# ----------------------EXERCISE 2 - Data plotting--------------------------------------  
# Instructions:  
# Open the dataframe (Ex_2_price_to_plot_v2.parquet) with the read_parquet() function of the pandas package.  
# If you upload your code to codepost use the the following path in the read_parquet() function : "Ex_2_price_to_plot_v2.parquet".  
# Display 1 scatter (dots only) plot. Only datas between the 2009-08-18 00:00:00 and the 2020-08-06 00:00:00 must be plotted.  
# The graph x axis is the "Low" column of the dataframe and the y axis is the "Close" column of the dataframe.  
# Its x limits are 13.6 and 83.5.  
# You can find the required plot in the file Ex_2_plot_V2.png  
  
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
  
     
  
elif input1 == '1':  
# ----------------------EXERCISE 1 - Data Cleaning--------------------------------------  
# Instructions:  
# Open the dataframe (Ex_1_price_to_clean_v3.parquet) with the read_parquet() function of the pandas package.  
# If you upload your code to codepost use the the following path in the read_parquet() function : "Ex_1_price_to_clean_v3.parquet".  
# Remove the line(s) with the data under the wrong format from the dataframe, reset the indexes, sort it by the date column and print it.  
# Keep the same format for the date column and be sure than the other columns are in float format.  
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
  
# Here is the printing instruction (where df is the final dataframe cleaned, sorted by date and with reset indexes)  
    print(df)  
# ---------------------End of EXERCISE 1 -------------------------------------- 
''' 
-------------------------ENDTERM EXAM----------------- 
DO NOT DELETE THE FOLLOWING CODE 
''' 
import sys 
try: 
    input1 = sys.argv[1] 
except: 
    pass 
''' 
In the following file, do not delete anything (comments, code, ...). Just add you code in every part (one per exercise). 
Use my variable for input (if there is any), use my printing for output (if there is any). 
You can upload your code to codepost.io to check the tests. A sucess in one test doesn't always mean than your exercise is correct, 
a fail doesn't always mean that your exercise is wrong. I will check all codes. 
At the end of exam, you should upload the last version of your code to codepost.io or to the online folder on Teams. 
The only authorized packages are: 
- pandas 
- pyarrow 
- fastparquet 
- numpy 
- sklearn 
- matplotlib 
- datetime 
 
''' 
 
if input1 == '1': 
# ----------------------EXERCISE 1 - Data Cleaning-------------------------------------- 
# Instructions: 
# Open the dataframe (Ex_1_price_to_clean_v8.parquet) with the read_parquet() function of the pandas package. 
# If you upload your code to codepost use the the following path in the read_parquet() function : "Ex
# Remove the duplicated line(s) from the dataframe, reset the indexes, sort it by the date column and print it. 
# Do not change the format of columns (for example, if one column has int values, don't change it to float) 
    import numpy as np 
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import sklearn.model_selection as sksel 
    import sklearn.preprocessing as skprepro 
    import sklearn.linear_model as skmod 
    import sklearn.preprocessing as skprepro 
    import datetime 
    from sklearn.linear_model import LinearRegression 
    df = pd.read_parquet("Ex_1_price_to_clean_v8.parquet") 
    df.drop_duplicates(inplace = True) 
    df = df.sort_values(by = [ 'Date']).reset_index(drop = True) 
 
 
 
# Here is the printing instruction (where df is the final dataframe cleaned, sorted by date and with reset indexes) 
    print(df) 
# ---------------------End of EXERCISE 1 -------------------------------------- 
 
elif input1 == '4': 
# ----------------------EXERCISE 4 - Machine Learning II-------------------------------------- 
# Instructions: 
# You have a list of feature named feature and a list of label named label. 
# Make a linear regression with the features and the label and display the equation and the accuracy of your model. 
# Do not use scaling or train/test sets. Choose the best printing instructions. 
 
    features = [-7, -7, -9, -3, 8, -4, -5, 7, 8, -4, -4, -2, 4, -2, -4, 1, -10, -5, 1, -9] 
    label = [-46.0, -75.0, -103.0, -46.0, 94.0, -50.0, -67.0, 56.0, 79.0, -46.0, -46.0, -24.0, 33.0, -16.0, -56.0, 3.0, -116.0, -48.0, 4.0, -113.0] 
 
    input1 = np.array(features) 
    input2 = np.array(label) 
 
    input1 = input1.reshape(-1, 1) 
    input2 = input2.reshape(-1, 1) 
 
    model = LinearRegression().fit(input1, input2) 
 
    w1 = model.coef_[0][0] 
    b = model.intercept_[0] 
    R2 = model.score(input1, input2) 
 
 
 
    # Here is several printings, choose the most appropriate one.   
    print("The most accurate linear regression has the following equation: y' = {:0.2f}*x + {:0.2f} and its accuracy is: {:0.3f}".format(w1, b, R2))   
    # print("The most accurate linear regression has the following equation: y' = {:0.2f}*x + {:0.2f} and its accuracy is: {:0.3f}".format(b, w1, R2))   
    # print("The most accurate linear regression has the following equation: y' = {:0.2f}*x + {:0.2f} and its accuracy is: {:0.3f}".format(R2, w1, b)) 
# ----------------------End of EXERCISE 4 --------------------------------------