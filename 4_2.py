__author__ = 'p3'

import pandas
from sklearn.decomposition import PCA
import numpy as np

objects = pandas.read_csv("close_prices.csv")
# print(objects[:2])

dates = objects['date']
objects = objects.drop('date', axis=1)
# print(objects[:3])
names = objects.axes[1]
pca = PCA(n_components=10)
pca.fit(objects)

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_[:4]))

tr = pca.transform(objects)
# print(tr[:, 0])

index = pandas.read_csv('djia_index.csv')
print(np.corrcoef(tr[:, 0], index['^DJI']))

d = dict(zip(pca.components_[0], names))
print(d[sorted(d, reverse=True)[0]])