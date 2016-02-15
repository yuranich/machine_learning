__author__ = 'yuranich'

import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()

data_tr = vectorizer.fit_transform(newsgroups.data)
# target_tr = vectorizer.transform(newsgroups.target)
print(len(vectorizer.get_feature_names()))

# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# kf = cv.KFold(newsgroups.target.size, n_folds=5, shuffle=True, random_state=241)
# clf = SVC(kernel='linear', random_state=241)
# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kf)
# gs.fit(data_tr, newsgroups.target)
# print(gs.best_params_)
# print(gs.best_score_)

svcLin = SVC(1.0, 'linear', random_state=241)
svcLin.fit(data_tr, newsgroups.target)
indices = svcLin.coef_.indices
data = svcLin.coef_.data

d = {}
i = 0
for x in indices:
    d[x] = abs(data[i])
    i += 1

s_ind = sorted(d, key=d.get, reverse=True)

words = []
for x in s_ind[:10]:
    words.append(vectorizer.get_feature_names()[x])

for w in sorted(words):
    print(w, end=' ')