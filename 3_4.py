__author__ = 'yuranich'

import numpy as np
import sklearn.metrics as met
import math

def getQ(data, target, w1, w2, l, C):
    Q = 0
    for i in range(0, l):
        Q += math.log2(1 + math.exp(-target[i]*(w1*data[i, 0] + w2*data[i, 1])))

    return 1/l*Q + 1/2*C*min(w1, w2)

def getW1(data, target, w1, w2, l, C, k):
    w = 0.0
    for i in range(0, l):
        w += target[i]*data[i, 0]*(1.0 - 1.0/(1.0 + math.exp(-target[i]*(w1*data[i, 0] + w2*data[i, 1]))))

    return w1 + k/l*w - k*C*w1

def getW2(data, target, w1, w2, l, C, k):
    w = 0.0
    for i in range(0, l):
        w += target[i]*data[i, 1]*(1.0 - 1.0/(1.0 + math.exp(-target[i]*(w1*data[i, 0] + w2*data[i, 1]))))

    return w2 + k/l*w - k*C*w2


def gradient_fall(data, target, C, k):
    w1 = [0.5]
    w2 = [0.5]
    l  = len(target)
    # Q = getQ(data, target, w1, w2, l, C)
    for i in range(1, 10000):
        w1.append(getW1(data, target, w1[i-1], w2[i-1], l, C, k))
        w2.append(getW2(data, target, w1[i-1], w2[i-1], l, C, k))
        if math.sqrt((w1[i]-w1[i-1])**2 + (w2[i]-w2[i-1])**2) <= 0.00001:
            break

    return np.array([w1, w2])

if __name__ == '__main__':
    data = np.genfromtxt('data-logistic.csv', delimiter=',')

    w = gradient_fall(data[:, [1, 2]], data[:, 0], 0, 0.1)
    print(w.shape)
    w = w.transpose()
    # print(w)
    w1 = w[w[:, 0].size - 1, 0]
    w2 = w[w[:, 0].size - 1, 1]
    # print(w1)
    a = 1.0/(1.0 + np.exp(-np.dot(data[:, 1], w1) - np.dot(data[:, 2], w2)))
    # print(a)
    first = met.roc_auc_score(data[:, 0], a)
    print(first)

    w = gradient_fall(data[:, [1, 2]], data[:, 0], 10, 0.1)
    print(w.shape)
    w = w.transpose()
    # print(w)
    w1 = w[w[:, 0].size - 1, 0]
    w2 = w[w[:, 0].size - 1, 1]
    # print(w1)
    a = 1.0/(1.0 + np.exp(-np.dot(data[:, 1], w1) - np.dot(data[:, 2], w2)))
    # print(a)
    first = met.roc_auc_score(data[:, 0], a)
    print(first)
