
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

data = pd.read_csv("student-mat.csv", sep=";")

data1 = data[["G1", "G2", "studytime", "failures", "absences"]]
predict = data["G3"]
X = np.array(data1)
y = np.array(predict)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)
print(f"Coefficient: {linear.coef_}")
print(f"Intercept: {linear.intercept_}")

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], y_test[x])






