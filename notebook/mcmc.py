import sys

sys.path.append("..")
import glob
import graph
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn.preprocessing
import statistics
import utils
import random


def func(x):
    sigma, *params = x
    params = scaler.inverse_transform([params])
    vp = np.squeeze(model.predict(params) / volume_max)
    sq = np.sum(np.subtract(vp, v0)**2)
    ssq = sigma * sigma
    return -0.5 * len(vp) * math.log(2 * math.pi * ssq) - 0.5 * sq / ssq


with open("model.pickle", "rb") as file:
    tp, model, test = pickle.load(file)

D = [utils.read(path) for path in glob.glob("1/*/MSolveInput.xml")]

scaler = sklearn.preprocessing.StandardScaler()
volume_max = max(v for params, time, volume in D for v in volume)
scaler.fit([params for (params, *rest) in D])

params0, time0, volume0 = D[test[3]]
v0 = np.squeeze(model.predict([params0]) / volume_max)
#v0 = np.interp(tp, time0, volume0) / volume_max

lo = 0, -2.5, -2.5, -2.5
hi = 0.5, 2.5, 2.5, 2.5
samples = list(
    zip(*graph.tmcmc(func, 1000, lo, hi, random=random.Random(12345))))
names = "sigma", "k1", "mu", "sv"
refs = 1, *scaler.transform([params0])[0]
for i in range(1, len(samples)):
    for j in range(i + 1, len(samples)):
        H, xe, ye = np.histogram2d(samples[i],
                                   samples[j],
                                   10,
                                   range=((lo[i], hi[i]), (lo[j], hi[j])),
                                   density=True)
        plt.gca().set_aspect("equal", "box")
        plt.imshow(H.T,
                   interpolation="spline16",
                   origin="lower",
                   extent=(lo[i], hi[i], lo[j], hi[i]),
                   cmap=plt.get_cmap("jet"))
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        plt.plot([refs[i]], [refs[j]], "xr")
        plt.savefig("hist.%s.%s.png" % (names[i], names[j]))
        plt.close()
