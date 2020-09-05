from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.datasets import load_iris

iris = load_iris() #['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'],


x = iris.data         # iris['data']
y = iris.target       # iris['target']

print(iris['target_names']) #OUTPUT:  ['setosa' 'versicolor' 'virginica']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0) # Splitting the dataset into train and test data

L = LinearRegression()  

L.fit(x_train,y_train) # Fitting the dataset in LinearRegression()


print(L.coef_) # Here we find b1 of equation b0+b1*x   # OUTPUT -> [-0.17009418 -0.01856621  0.27900206  0.56061274]

print(L.intercept_) # intercept (b0)     # OUTPUT -> 0.35017224206863884

Pred_y = L.predict(x_test) # Predicting from test dataset what we have taken aside before and not used to test data.

acc = r2_score(y_test,Pred_y) #Calculating the accuracy of our score


print(acc)     # OUTPUT -> 0.8998261101639005  i.e. accuracy is 90 %


#mean_absolute_error measures the average magnitude of the errors in a set of predictions,without considering their direction.
#The Mean Absolute Error is the average of all absolute errors.
print(mean_absolute_error(y_test,Pred_y))  # OUTPUT -> 0.19781443339791988

#Root mean square error is a quadratic scoring rule that also measures the average magnitude of the error.
# The average squared difference between the estimated values and the actual value
print(mean_squared_error(y_test,Pred_y))   # OUTPUT -> 0.058867619212325153





