__author__ = 'p3'

import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse
from sklearn.linear_model import Ridge

data_train = pandas.read_csv("salary-train.csv")
data_test  = pandas.read_csv("salary-test-mini.csv")

data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_train['FullDescription'] = data_train['FullDescription'].apply(str.lower)

data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['FullDescription'] = data_test['FullDescription'].apply(str.lower)


vectorizer = TfidfVectorizer(min_df=5)
data_tr = vectorizer.fit_transform(data_train['FullDescription'])
dat_test = vectorizer.transform(data_test['FullDescription'])
# print(data_tr[:5])

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
# print(data_train[:5])
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
# print(X_train_categ[:5])

objects = sparse.hstack([data_tr, X_train_categ])
obj_test = sparse.hstack([dat_test, X_test_categ])
# print(objects)
target = data_train['SalaryNormalized']
ridge = Ridge()
ridge.fit(objects, target.as_matrix())
pred = ridge.predict(obj_test)
print(pred)