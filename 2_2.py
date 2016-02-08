__author__ = 'yuranich'

"""
Working with kNN regression.
"""

from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation as cv
import numpy as np

ds = load_boston()
# print(ds.data)

ds.data = scale(ds.data)
# print(ds.data)

kf = cv.KFold(len(ds.target), n_folds=5, shuffle=True, random_state=42)

m_range = np.linspace(1, 10, 200)
# print(space)
results = {}
for param in m_range:
    kr = KNeighborsRegressor(weights='distance', p=param)
    results[param] = cv.cross_val_score(kr, ds.data, ds.target, scoring='mean_squared_error', cv=kf).max()

sorted_res = sorted(results, key=results.get, reverse=True)
for i in sorted_res:
    print("p=%s, max=%s" % (i, results[i]))