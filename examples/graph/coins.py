import graph
import math
import numpy as np
import pickle
import random
import scipy.stats
import statistics
import sys

random.seed(123456)
n = 250
y = [140, 110]
theta_given_psi = [
    lambda theta, psi: scipy.stats.beta.pdf(theta[0], a=psi[0], b=psi[1]),
    lambda theta, psi: scipy.stats.beta.pdf(theta[0], a=psi[0], b=psi[1]),
]

data_given_theta = [
    lambda theta: scipy.stats.binom.pmf(y[0], n, theta[0]),
    lambda theta: scipy.stats.binom.pmf(y[1], n, theta[0]),
]

init = [y[0] / n, y[1] / n]
scale = 0.05
draws = 100
integral = [
    graph.Integral(data_given_theta[0],
                   theta_given_psi[0],
                   draws=draws,
                   init=[init[0]],
                   scale=[scale]),
    graph.Integral(data_given_theta[1],
                   theta_given_psi[1],
                   draws=draws,
                   init=[init[1]],
                   scale=[scale]),
]


def prior(psi):
    return scipy.stats.gamma.pdf(psi[0], 4, scale=2) * scipy.stats.gamma.pdf(
        psi[1], 4, scale=2)


def likelihood(psi):
    return integral[0](psi) * integral[1](psi)


samples = graph.metropolis(lambda psi: likelihood(psi) * prior(psi),
                           draws=50000,
                           init=[0.1, 0.1],
                           scale=[1.5, 1.5])
with open("coins.samples.pkl", "wb") as f:
    pickle.dump(list(samples), f)
'''
https://allendowney.github.io/BayesianInferencePyMC/04_hierarchical.html#going-hierarchical
'''
