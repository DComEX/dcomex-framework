.. _graph:

Graphs
======

The code is a Python package containing classes and functions for
estimating integrals using various sampling algorithms. The Integral
class in the package caches samples and evaluates the integral given a
hyperparameter psi. It takes two callable arguments: data_given_theta,
which is the joint probability of the observed data viewed as a
function of parameter, and theta_given_psi, which is the conditional
probability of parameters theta given hyperparameter psi. The method
parameter specifies the type of sampling algorithm to use, and the
options parameter is a dictionary of options for the sampling
algorithm.

The package contains several sampling algorithms implemented as
functions, including metropolis and langevin. metropolis is a
Metropolis sampler, which takes as arguments a function that
calculates the unnormalized density or log unnormalized probability,
the number of samples to draw, the initial point, the scale of the
proposal distribution, and a boolean indicating whether to assume
log-probability. langevin is a Metropolis-adjusted Langevin (MALA)
sampler, which takes as arguments a function that calculates the
unnormalized density or log unnormalized probability, the number of
samples to draw, the initial point, the gradient of the log
unnormalized probability, the standard deviation of the proposal
distribution, and a boolean indicating whether to assume
log-probability.


Summary
-------

.. autosummary::
   graph.cmaes
   graph.Integral
   graph.korali
   graph.metropolis
   graph.tmcmc

Functions
---------

.. autoclass:: graph.Integral
	       :special-members: __call__
.. autofunction:: graph.cmaes
.. autofunction:: graph.korali
.. autofunction:: graph.metropolis
.. autofunction:: graph.tmcmc
