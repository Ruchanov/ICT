'''
-------------------------ENDTERM EXAM-----------------
DO NOT DELETE THE FOLLOWING CODE
'''
import sys
try:
    # input1 = sys.argv[1]
    штзге
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
if input1 == '4':
# ----------------------EXERCISE 4 - Machine Learning II--------------------------------------
# Instructions:
# You have a model trained with one feature : feature1. Make a scatter plot with the feature as x axis and the label as y axis.
# Add on the same plot, the predictions of the model (line plot).You can find the required plot in the file Ex_4_plot_V18.png
    import sklearn.linear_model as skmod
    import numpy as np
    feature1 = np.array([4, 5, -8, -1, 5, 0, 1, -2, -7, 0, -3, 3, -4, -7, 1, 5, -1, 4, 9, 7])
    label = np.array([-34.0, -44.0, 123.0, 27.0, -36.0, 14.0, 2.0, 27.0, 109.0, 9.0, 54.0, -15.0, 45.0, 78.0, 4.0, -28.0, 16.0, -20.0, -89.0, -49.0])
    model = skmod.LinearRegression().fit(feature1.reshape(-1,1), label.reshape(-1,1))
    plt.scatter(feature1,label)  
    x = np.array([-7,7]).reshape(-1,1)  
    plt.plot(x, model.predict(x))  
    plt.show()  
# ----------------------End of EXERCISE 4 --------------------------------------


elif input1 == '5':
# ----------------------EXERCISE 5 - Machine Learning III--------------------------------------
# Instructions:
# You have two features : feature1 and feature2. You objective is to make a two columns matrix and to standardized
# them by using the standardization equation or the standard Scaler of the scikit-learn package.
# You can find the desired matrix in the document Ex_5_standard_matrix_V18.txt
    feature1 = [-770, -189, -523, -471, 107, -1000, -130, -91, -808, -405, -98, 497, -135, -112, -631, -683, -35, -901, -97, 347, -965, -919, -747, 7, -915, 196, -609, -641, -414, -159]
    feature2 = [-4, -5, 0, -3, -10, -8, -4, -4, -5, -3, 3, -9, -6, -6, -3, 1, -2, -7, -3, -9, 1, -5, -6, -7, 1, -4, -5, -7, -9, -2]  
    arr1 = np.array(feature1).reshape(-1,1)  
    arr2 = np.array(feature2).reshape(-1,1)  
    x = np.hstack([arr1, arr2])  
    x_train, x_standard = sksel.train_test_split(x, train_size=.8, shuffle=False)    

# Here is the print instructions to print the standardized matrix rounded.
    print(np.around(x_standard, 3))

# ----------------------End of EXERCISE 5 --------------------------------------