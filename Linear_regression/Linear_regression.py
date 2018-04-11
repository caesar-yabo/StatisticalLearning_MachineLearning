import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

df = open('D:\\My_python_Data\\Jupyter\\2016鸟\\白鹡鸰 .csv')
my_raw_data = pd.read_csv(df)
mydata = my_raw_data[['个体数', '低温', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']]
y = mydata.个体数
X = mydata[['低温', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

lr = LinearRegression()
lr.fit(X_train, y_train) 

#训练所得模型的系数
print (lr.intercept_)  #常数项
print (lr.coef_)

y_pred = lr.predict(X_test)

from sklearn import metrics  
# 用scikit-learn计算MSE
print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE  均方根差(Root Mean Squared Error, RMSE)
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#得到了MSE或者RMSE，如果我们用其他方法得到了不同的系数，需要选择模型时，就用MSE小的时候对应的参数。

#我们可以通过交叉验证来持续优化模型，代码如下，我们采用10折交叉验证，即cross_val_predict中的cv参数为10：
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(lr, X, y, cv=10)

#可以自动标准化l
lr2 = LinearRegression(normalize=True)
lr2.fit(X, Y)
predictions2 = lr2.predict(X)
#Y - predictions2

