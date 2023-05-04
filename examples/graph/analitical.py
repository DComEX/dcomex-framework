import math
import matplotlib.pylab as plt
import numpy as np
import random
import statistics
import graph

random.seed(123456)
y = [2, 3]
sigma = 0.1
theta_given_psi = [
    lambda theta, psi: statistics.NormalDist(psi, 1).pdf(theta[0]),
    lambda theta, psi: statistics.NormalDist(psi, 5).pdf(theta[0]),
]

data_given_theta = [
    lambda theta: -(theta[0] - y[0])**2 / sigma**2 / 2,
    lambda theta: -(theta[0] - y[1])**2 / sigma**2 / 2,
]

integral = [
    graph.Integral(data_given_theta[0],
                   theta_given_psi[0],
                   draws=1000,
                   init=[0],
                   scale=[0.1],
                   log=True),
    graph.Integral(data_given_theta[1],
                   theta_given_psi[1],
                   draws=1000,
                   init=[0],
                   scale=[0.1],
                   log=True),
]


def likelihood(psi):
    return math.prod(fun(psi[0]) for fun in integral)


def prior(psi):
    return 1 if -4 <= psi[0] <= 4 else 0


def fpost(psi):
    return 0.4225878520215124 * math.exp(-0.5150415081492156 * psi**2 +
                                         2.100150038994303 * psi -
                                         2.160126048590465)


samples = graph.metropolis(lambda psi: likelihood(psi) * prior(psi),
                           draws=5000,
                           init=[0],
                           scale=[1.0])
psi = np.linspace(-4, 4, 100)
post = [fpost(e) for e in psi]
plt.yticks([])
plt.xlim(-2, 5)
plt.hist([e[0] for e in samples],
         50,
         density=True,
         histtype='step',
         linewidth=2)
plt.plot(psi, post)
plt.savefig("analitical.vis.png")
