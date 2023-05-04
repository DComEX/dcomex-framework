import math
import statistics
import graph
import random
import sys


def gauss(x):
    return -statistics.fsum(e**2 for e in x) / 2


random.seed(123456)
sampler = graph.tmcmc
beta = 0.1
N = 2000
M = 50
d = 3
mean = []
var0 = []
var1 = []
var2 = []
lo = -5
hi = 5
for t in range(M):
    x = sampler(gauss, N, d * [lo], d * [hi], beta=beta)
    mean.append(statistics.fmean(e[0] for e in x))
    var0.append(statistics.variance(e[0] for e in x))
    var1.append(statistics.variance(e[1] for e in x))
    var2.append(statistics.variance(e[2] for e in x))
print("%.3f %.4f %.4f %.4f" % (statistics.fmean(mean), statistics.fmean(var0),
                               statistics.fmean(var1), statistics.fmean(var2)))
