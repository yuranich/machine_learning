__author__ = 'yuranich'

"""
Working with kNN classification.
"""

from sklearn import cross_validation as cv
from sklearn import neighbors as nb
import pandas
from sklearn.preprocessing import scale

df = pandas.read_csv("wine.data")
classes = df['Class']
data = df.drop(labels='Class', axis=1)
data = scale(data)

kf = cv.KFold(178, n_folds=5, shuffle=True, random_state=42)

results = {}
for k in range(1, 51):
    classifier = nb.KNeighborsClassifier(n_neighbors=k)
    cvc = cv.cross_val_score(classifier, data, classes, cv=kf)
    results[k] = cvc.mean()

sorted_res = sorted(results, key=results.get, reverse=True)
for i in sorted_res:
    print("k=%s, max=%s" % (i, results[i]))
