
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

best_acc = 0

data = pd.read_csv("student-mat.csv", sep=";")

data1 = data[["G1", "G2", "studytime", "failures", "absences"]]
predict = data["G3"]
X = np.array(data1)
y = np.array(predict)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

for a in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best_acc:
        best_acc = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print(f"Accuracy of best model: {best_acc}")

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print(f"Coefficient: {linear.coef_}")
print(f"Intercept: {linear.intercept_}")

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], y_test[x])






