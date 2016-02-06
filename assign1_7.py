__author__ = 'yuranich'

""" Solution to the second task of first week.
    Gettin importances of features.
"""

import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv')

objects = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

objects = objects[np.isnan(objects['Pclass']) == False]
objects = objects[np.isnan(objects['Fare']) == False]
objects = objects[np.isnan(objects['Age']) == False]

for x in objects.index:
    value = objects.get_value(x, 'Sex').strip()
    if value == 'male':
        objects.set_value(x, 'Sex', 1)
    else:
        objects.set_value(x, 'Sex', 0)


aim = objects['Survived']
objects = objects[['Pclass', 'Fare', 'Age', 'Sex']]

print(objects)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(objects, aim)

importances = clf.feature_importances_

print(importances)
