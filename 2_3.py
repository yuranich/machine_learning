__author__ = 'yuranich'

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train = np.genfromtxt('perceptron-train.csv', delimiter=',')
test = np.genfromtxt('perceptron-test.csv', delimiter=',')

clf = Perceptron(random_state=241)
clf.fit(train[:, [1, 2]], train[:, 0])
predictions = clf.predict(train[:, [1, 2]])

score = accuracy_score(train[:, 0], predictions)
print(score)

t_pred = clf.predict(test[:, [1, 2]])
t_score = accuracy_score(test[:, 0], t_pred)
print(t_score)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(train[:, [1, 2]])
x_test_scaled = scaler.transform(test[:, [1, 2]])
# print(train_scaled)

test_clf = Perceptron(random_state=241)
test_clf.fit(x_train_scaled, train[:, 0])
predictions = test_clf.predict(x_train_scaled)

score_scaled = accuracy_score(train[:, 0], predictions)
print(score_scaled)

test_predict = test_clf.predict(x_test_scaled)
test_score = accuracy_score(test[:, 0], test_predict)
print(test_score)

print(test_score - t_score)