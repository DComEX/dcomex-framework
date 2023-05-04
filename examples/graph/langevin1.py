import math
import random
import matplotlib.pylab as plt
import graph
import numpy as np


def fun(x):
    return -1 / 2 * (x[0]**2 + x[1]**2)


def dfun(x):
    return [-x[0], -x[1]]


random.seed(123456)
draws = 10000
sigma = math.sqrt(3)
S0 = graph.langevin(fun, draws, [0, 0], dfun, sigma, log=True)
plt.scatter(*zip(*S0), alpha=0.1)
plt.savefig("langevin1.png")
