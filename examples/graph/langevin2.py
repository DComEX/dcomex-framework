import graph
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


def lprob(x):
    return target.log_prob(x)


def fun(x):
    return -1 / 2 * x[0]**2


def dfun(x):
    return [-x[0]]


tfd = tfp.distributions
dtype = np.float32
step_size = 0.75
sigma = math.sqrt(2 * step_size)
target = tfd.Normal(loc=dtype(0), scale=dtype(1))
draws = 1000
S0 = tfp.mcmc.sample_chain(num_results=draws,
                           current_state=dtype(1),
                           kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
                               lprob, step_size=step_size),
                           trace_fn=None,
                           seed=42)
S1 = graph.langevin(fun, draws, [0], dfun, sigma, log=True)
x = np.linspace(-4, 4, 100)
y = [math.exp(-x**2 / 2) / math.sqrt(2 * math.pi) for x in x]
plt.hist(S0, 100, histtype='step', density=True, color='red')
plt.hist([e for (e, ) in S1], 100, histtype='step', density=True, color='blue')
plt.plot(x, y, '-g')
plt.savefig('langevin2.png')
