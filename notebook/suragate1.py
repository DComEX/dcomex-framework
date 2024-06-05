from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import sklearn.pipeline
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
import functools
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
import pickle

D = [utils.read(path) for path in glob.glob("1/*/MSolveInput.xml")]
nt = 20
param, time, volume = D[0]
tend = time[-1]
tp = [i * tend / nt for i in range(1, nt + 1)]
train, test = train_test_split(range(len(D)), test_size=0.5, random_state=1234)
ftrain = functools.partial(np.take, indices=train, axis=0)
ftest = functools.partial(np.take, indices=test, axis=0)
X = []
y = []
for (k1, mu, sv, *rest), time, volume in D:
    vol = np.interp(tp, time, volume)
    X.append((k1, mu, sv))
    y.append(vol)
X_train, y_train = map(ftrain, [X, y])
X_test, y_test = map(ftest, [X, y])
pipe = sklearn.pipeline.make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=[80] * 20, max_iter=100000))
model = TransformedTargetRegressor(regressor=pipe, transformer=MinMaxScaler())
model.fit(X_train, y_train)
print("score: train, test:", model.score(X_train, y_train),
      model.score(X_test, y_test))
with open('model.pickle', 'wb') as handle:
    pickle.dump((tp, model, test), handle)

points, line = plt.plot([], [], 'or', [], [], '-k')
vmax = max(v for (param, time, volume) in D for v in volume)
plt.axis((0, 1.1 * tend, 0, 1.1 * vmax))
for idx in test[:5]:
    params, time, volume = D[idx]
    vp = model.predict([params])
    points.set_data(tp, vp)
    line.set_data(time, volume)
    plt.savefig("suragate.%05d.png" % idx)
