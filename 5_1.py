__author__ = 'yuranich'

import pandas
from sklearn import cross_validation as cv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

abalone = pandas.read_csv("abalone.csv")
abalone['Sex'] = abalone['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

target = abalone['Rings']
objects = abalone.drop('Rings', axis=1)

kf = cv.KFold(target.size, n_folds=5, shuffle=True, random_state=1)
results = {}

for n in range(1, 51):
    regr = RandomForestRegressor(n_estimators=n, random_state=1)
    cvpred = cv.cross_val_predict(regr, objects, target, cv=kf)
    results[n] = r2_score(target, cvpred)

sorted_res = sorted(results, key=results.get, reverse=True)
for i in sorted_res:
    print("trees=%s, score=%s" % (i, results[i]))
