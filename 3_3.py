__author__ = 'yuranich'

import pandas
import sklearn.metrics as met

df = pandas.read_csv("classification.csv")

true = df['true']
pred = df['pred']

TP = df[true == 1][pred == 1]
TN = df[true == 0][pred == 0]

FP = df[true == 0][pred == 1]
FN = df[true == 1][pred == 0]

print(len(TP['true']), end=' ')
print(len(FP['true']), end=' ')
print(len(FN['true']), end=' ')
print(len(TN['true']))
print()

acc = met.accuracy_score(true, pred)
pr  = met.precision_score(true, pred)
rec = met.recall_score(true, pred)
f1  = met.f1_score(true, pred)

print(acc, end=' ')
print(pr, end=' ')
print(rec, end=' ')
print(f1)
print()

scores = pandas.read_csv("scores.csv")

accLogreg = met.roc_auc_score(scores['true'], scores['score_logreg'])
accSVM = met.roc_auc_score(scores['true'], scores['score_svm'])
accKNN = met.roc_auc_score(scores['true'], scores['score_knn'])
accTree = met.roc_auc_score(scores['true'], scores['score_tree'])

print(accLogreg)
print(accSVM)
print(accKNN)
print(accTree)
print()

curLogreg = met.precision_recall_curve(scores['true'], scores['score_logreg'])
curSVM = met.precision_recall_curve(scores['true'], scores['score_svm'])
curKNN = met.precision_recall_curve(scores['true'], scores['score_knn'])
curTree = met.precision_recall_curve(scores['true'], scores['score_tree'])

print(curLogreg[0][curLogreg[1] >= 0.7].max())
print(curSVM[0][curSVM[1] >= 0.7].max())
print(curKNN[0][curKNN[1] >= 0.7].max())
print(curTree[0][curTree[1] >= 0.7].max())
