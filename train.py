import sklearn.preprocessing
import sklearn.mixture
import bpmfeature
import otherfeature
import random
import collections
import numpy as np

def validate(trainset, testset):
    labels = collections.defaultdict(list)
    for name, fea in trainset:
        labels[name.split('/')[1]].append(fea)

    models = dict()
    for clas, feas in labels.items():
        X = np.array(feas)
        M = sklearn.mixture.BayesianGaussianMixture(1)
        M.fit(X)
        models[clas] = M

    corcnt = 0

    for name, fea in testset:
        ans = name.split('/')[1]
        maxscore = -np.Inf
        pred = None
        for clas, M in models.items():
            prob = M.score([fea])
            if prob > maxscore:
                maxscore = prob
                pred = clas

        if ans == pred:
            corcnt += 1

    return float(corcnt) / len(testset) * 100

datadict = dict()

for name, fea in otherfeature.data:
    datadict[name] = fea

for name, fea in bpmfeature.data:
    if name in datadict:
        datadict[name] += fea

data = list(datadict.items())
mat = np.zeros([len(data), len(data[0][1])])
for i, row in enumerate(data):
    mat[i] = row[1]
sklearn.preprocessing.scale(mat, 0, copy=False)
sdata = list()
for i, row in enumerate(data):
    sdata.append((row[0], mat[i]))
data = sdata
random.seed(23)
random.shuffle(data)

chunk = len(data) / 5
accs = list()
for st in range(0, len(data), chunk):
    trainset = data[0 : st] + data[st + chunk:]
    testset = data[st: st + chunk]
    accs.append(validate(trainset, testset))

print(np.mean(accs))
