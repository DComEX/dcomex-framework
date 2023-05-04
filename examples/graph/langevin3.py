import graph
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.mcmc import sample
from tensorflow_probability.python.mcmc import langevin


def target_log_prob(z):
    return target.log_prob(z)


dtype = np.float32
true_mean = dtype([1, 2, 7])
true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
num_results = 500
num_chains = 500
chol = np.linalg.cholesky(true_cov)
target = mvn_tril.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)
init_state = [
    np.ones([num_chains, 3], dtype=dtype),
]
states = sample.sample_chain(
    num_results=num_results,
    current_state=init_state,
    kernel=langevin.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=target_log_prob, step_size=.1),
    num_burnin_steps=200,
    num_steps_between_results=1,
    trace_fn=None,
    seed=123456)
states = tf.concat(states, axis=-1)
sample_mean = tf.reduce_mean(states, axis=[0, 1])
x = (states - sample_mean)[..., tf.newaxis]
sample_cov = tf.reduce_mean(tf.matmul(x, tf.transpose(a=x, perm=[0, 1, 3, 2])),
                            axis=[0, 1])
print(sample_mean.numpy())
print(sample_cov.numpy())
