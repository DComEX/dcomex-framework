import graph
import math
import random
import scipy.stats
import statistics
import sys


def fun(x, a, b, w, dev):
    coeff = -1 / (2 * dev**2)
    u = coeff * statistics.fsum((e - d)**2 for e, d in zip(x, a))
    v = coeff * statistics.fsum((e - d)**2 for e, d in zip(x, b))
    return scipy.special.logsumexp((u + math.log(w), v + math.log(1 - w)))


sampler = graph.tmcmc
# sampler = graph.korali

random.seed(12345)
D = (
    ("I", 2, 0.5, 0.5),
    ("II", 2, 0.1, 0.9),
    ("III", 4, 0.5, 0.5),
    ("IV", 4, 0.1, 0.9),
    ("V", 6, 0.5, 0.5),
    ("VI", 6, 0.3, 0.5),
    ("VII", 6, 0.1, 0.5),
    ("VIII", 6, 0.1, 0.9),
)
N = 1000
M = 50
beta = 1.0
print("beta = %g" % beta)
for name, d, dev, w in D:
    a = [0.5] * d
    b = [-0.5] * d
    first_peak = []
    smax = []
    logev = []
    for t in range(M):
        x, S = sampler(lambda x: fun(x, a, b, w, dev),
                       N, [-2] * d, [2] * d,
                       beta=beta,
                       return_evidence=True)
        cnt = 0
        for e in x:
            da = statistics.fsum((u - v)**2 for u, v in zip(e, a))
            db = statistics.fsum((u - v)**2 for u, v in zip(e, b))
            if da < db:
                cnt += 1
        first_peak.append(cnt / N)
        smax.append(statistics.fmean(max(e) for e in x))
        logev.append(S)
    cv = lambda a: (statistics.mean(a), 100 * abs(scipy.stats.variation(a)))
    print("%4s %.2f (%.1f%%) %4.2f (%.1f%%) %4.2f (%.1f%%)" %
          (name, *cv(first_peak), *cv(smax), *cv(logev)))
'''
Example 2: Mixture of Two Gaussians
Table 3.Summary of the Analysis Results for Example 2

[0] Ching, J., & Chen, Y. C. (2007). Transitional Markov chain Monte Carlo
method for Bayesian model updating, model class selection, and model
averaging. Journal of engineering mechanics, 133(7), 816-832.

'''
