__author__ = 'yuranich'

import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as met
import matplotlib.pyplot as plt

df = pandas.read_csv("gbm-data.csv")

vals = df.values

X_train, X_test, y_train, y_test = train_test_split(vals[:, 1:], vals[:, 0], test_size=0.8, random_state=241)

# for lr in [1, 0.5, 0.3, 0.2, 0.1]:
clf = GradientBoostingClassifier(learning_rate=1, n_estimators=250, verbose=False, random_state=241)
clf.fit(X_train, y_train)
sc_train = enumerate(clf.staged_decision_function(X_train))
sc_test  = enumerate(clf.staged_decision_function(X_test))
train_loss = {}
test_loss = {}
for i, y_predicted in sc_train:
    train_loss[i] = met.log_loss(y_train,1/(1+np.exp(-y_predicted)))

for i, y_predicted in sc_test:
    test_loss[i] = met.log_loss(y_test, 1/(1+np.exp(-y_predicted)))

plt.figure()
plt.plot(list(test_loss.values()), 'r', linewidth=2)
plt.plot(list(train_loss.values()), 'g', linewidth=2)
plt.legend(['test', 'train'])
plt.show()


# print(train_loss)
# print()
# print(test_loss)