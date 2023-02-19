import numpy as np
import sklearn.model_selection as sksel
import sklearn.preprocessing as skprepro
feature1 = [-770, -189, -523, -471, 107, -1000, -130, -91, -808, -405, -98, 497, -135, -112, -631, -683, -35, -901, -97, 347, -965, -919, -747, 7, -915, 196, -609, -641, -414, -159]
feature2 = [-4, -5, 0, -3, -10, -8, -4, -4, -5, -3, 3, -9, -6, -6, -3, 1, -2, -7, -3, -9, 1, -5, -6, -7, 1, -4, -5, -7, -9, -2]  
feature1 = np.array(feature1).reshape(-1,1)  
feature2 = np.array(feature2).reshape(-1,1)  
x = np.hstack([feature1, feature2])  
x_train, x_test = sksel.train_test_split(x, train_size=.8, shuffle=False)
feature   = [-657, -254, -798, -237, -319, 170, -58, -651, 1, 29, -58, -738, 36, -98, 406, -842, -285, -963, -789, -326, -439, 413, 373, -512, -78, 110, -556, -386, -41, -112]
arr_f = np.array(feature).reshape(-1, 1)
poly4 = skprepro.PolynomialFeatures(degree=4, include_bias=False)
x_polymial = poly4.fit_transform(arr_f)  
print(np.around(x_polymial, 3))