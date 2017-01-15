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
    confus = collections.defaultdict(lambda : collections.defaultdict(int))

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

        confus[ans][pred] += 1

    return (float(corcnt) / len(testset), confus)

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
random.seed(31)
random.shuffle(data)

chunk = len(data) / 9 + 1
st = 0
trainset = data[0 : st] + data[st + chunk:]
testset = data[st: st + chunk]

acc, confus = validate(trainset, testset)

print('acc: %f'%acc)
print('confusion matrix (answer \ predict)')
accmap = dict((y,x) for x, y in enumerate(confus.keys()))
title = '\t' + '\t'.join(x[0][:7] for x in sorted(list(accmap.items()), key=lambda x: x[1]))
print(title)
for name, row in confus.items():
    lis = list([0] * len(confus))
    s = sum(row.values())
    for n, val in row.items():
        lis[accmap[n]] = '%.2f'%(float(val) / s)
    print(name[:7] + '\t' + '\t'.join(str(x) for x in lis))
