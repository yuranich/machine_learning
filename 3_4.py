__author__ = 'yuranich'

import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as met

data = np.genfromtxt('data-logistic.csv', delimiter=',')

