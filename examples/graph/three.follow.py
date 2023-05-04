import scipy.stats
import graph
import numpy as np
import random
import math
import follow


def prior(psi):
    return 1 if 1 < psi[0] < 40 else 0


seed = 123456
np.random.seed(seed)
random.seed(seed)
data = [0.28, 1.51, 1.14]
l1_given_l0 = follow.follow("l1 ~ l0")(
    lambda theta, psi: scipy.stats.halfnorm.pdf(theta[0], scale=psi[0]))
l2_given_l1 = follow.follow("l2 ~ l1")(
    lambda theta, psi: scipy.stats.halfnorm.pdf(theta[0], scale=psi[0]))
data_given_l2 = follow.follow("y ~ l2")(lambda theta: math.prod(
    scipy.stats.halfnorm.pdf(e, scale=theta[0]) for e in data))
l1 = graph.Integral(data_given_l2,
                    l2_given_l1,
                    draws=10,
                    init=[1],
                    scale=[1.5])
l0 = graph.Integral(l1, l1_given_l0, draws=1000, init=[50], scale=[20])
samples = graph.metropolis(lambda psi: l0(psi) * prior(psi),
                           draws=500,
                           init=[50],
                           scale=[20])
print("has loop:", follow.loop())
with open("three.follow.gv", "w") as file:
    follow.graphviz(file)
