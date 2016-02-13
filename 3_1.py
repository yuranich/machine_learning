__author__ = 'p3'

import numpy as np
from sklearn.svm import SVC

svcLin = SVC(100000, 'linear', random_state=241)

data = np.genfromtxt('svm-data.csv', delimiter=',')

svcLin.fit(data[:, [1, 2]], data[:, 0])

indexes = svcLin.support_
print(indexes)