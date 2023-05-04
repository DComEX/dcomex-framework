import math
import numpy as np
import random
import statistics
import follow
import graph

random.seed(123456)
y = [2, 3]
sigma = 0.1
theta_given_psi = [
    follow.follow("theta[0] ~ psi")(
        lambda theta, psi: statistics.NormalDist(psi, 1).pdf(theta[0])),
    follow.follow("theta[1] ~ psi")(
        lambda theta, psi: statistics.NormalDist(psi, 5).pdf(theta[0])),
]

data_given_theta = [
    follow.follow("y[0] ~ theta[0]")(
        lambda theta: -(theta[0] - y[0])**2 / sigma**2 / 2),
    follow.follow("y[1] ~ theta[1]")(
        lambda theta: -(theta[0] - y[1])**2 / sigma**2 / 2),
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


@follow.follow(label="psi ~ uniform")
def likelihood(psi):
    return math.prod(fun(psi[0]) for fun in integral)


def prior(psi):
    return 1 if -4 <= psi[0] <= 4 else 0


samples = graph.metropolis(lambda psi: likelihood(psi) * prior(psi),
                           draws=100,
                           init=[0],
                           scale=[1.0])
for s in samples:
    break
print("has loop:", follow.loop())
with open("analitical.follow.gv", "w") as file:
    follow.graphviz(file)
