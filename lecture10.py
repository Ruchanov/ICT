import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skmod
# input1 = [int(i) for i in input1.split(',')]
# input2 = [int(i) for i in input2.split(',')]
input1 = [3.25, 0.816, 4.376, 1.314, 3.982, 2.957, 2.482, 3.7]
input2 = [21, 22, 13, 25, 17, 23, 23, 27]
arr_x = np.array(input1)
arr_x2 = arr_x.reshape(-1,1)
arr_y = np.array(input2)
arr_y2 = arr_y.reshape(-1,1)
plt.scatter(arr_x, arr_y)
model = skmod.LinearRegression()
model_trained = model.fit(arr_x, arr_y)

#use this printing (where "w1" is the weight and "b" the bias)
print("The most accurate linear regression has the following equation: y' = {:0.2f}*x + {:0.2f}".format(model_trained.coef_[0][0], model_trained.intercept_[0][0]))