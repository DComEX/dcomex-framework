import sys
import statistics
import random
import math
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions


def kahan_cumsum(a):
    ans = []
    s = 0.0
    c = 0.0
    for e in a:
        y = e - c
        t = s + y
        c = (t - s) - y
        s = t
        ans.append(s)
    return ans


def kahan_sum(a):
    s = 0.0
    c = 0.0
    for e in a:
        y = e - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def inside(x):
    for l, h, e in zip(lo, hi, x):
        if e < l or e > h:
            return False
    return True


def fun0(theta, x, y):
    alpha, beta, sigma = theta
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    M = len(x)
    dif = [y - alpha * x - beta for x, y in zip(x, y)]
    sumsq = statistics.fsum(dif**2 for dif in dif)
    res = -0.5 * M * (math.log(2 * math.pi) +
                      2 * math.log(sigma)) - 0.5 * sumsq / sigma2
    sx = statistics.fsum(x)
    xx = statistics.fsum(x * x for x in x)
    FIM = [[xx, sx, 0], [sx, M, 0], [0, 0, 2 * M]]
    inv_FIM = np.linalg.inv(FIM) * sigma2
    D, V = np.linalg.eig(inv_FIM)
    dd = statistics.fsum(dif)
    dx = statistics.fsum(dif * x for dif, x in zip(dif, x))
    gradient = [dx / sigma2, dd / sigma2, -M / sigma + sumsq / sigma3]
    return res, gradient, inv_FIM, V, D


def fun(theta):
    return fun0(theta, xd, yd)


seed = 123456
np.random.seed(seed)
random.seed(seed)
alpha = 2
beta = -2
sigma = 2
xd = np.linspace(1, 10, 40)
yd = [alpha * xd + beta + random.gauss(0, sigma) for xd in xd]
N = 100
eps = 0.04
d = 3
lo = (-5, -5, 0)
hi = (5, 5, 10)
conf = 0.68
chi = scipy.stats.distributions.chi2.ppf(conf, d)
x = [[random.uniform(l, h) for l, h in zip(lo, hi)] for i in range(N)]
f = [None] * N
out = [None] * N
for i in range(N):
    f[i], *out[i] = fun(x[i])
x2 = [[None] * d for i in range(N)]
f2 = [None] * N
out2 = [None] * N
gen = 1
p = 0
End = False
cov = [[None] * d for i in range(d)]
while True:
    old_p, plo, phi = p, p, 2
    while phi - plo > eps:
        p = (plo + phi) / 2
        temp = [(p - old_p) * f for f in f]
        M1 = scipy.special.logsumexp(temp) - math.log(N)
        M2 = scipy.special.logsumexp(2 * temp) - math.log(N)
        if M2 - 2 * M1 > math.log(2):
            phi = p
        else:
            plo = p
    if p > 1:
        p = 1
        End = True
    dp = p - old_p
    weight = scipy.special.softmax([dp * f for f in f])
    mu = [kahan_sum(w * e[k] for w, e in zip(weight, x)) for k in range(d)]
    x0 = [[a - b for a, b in zip(e, mu)] for e in x]
    for l in range(d):
        for k in range(l, d):
            cov[k][l] = cov[l][k] = beta * beta * kahan_sum(
                w * e[k] * e[l] for w, e in zip(weight, x0))
    ind = random.choices(range(N), cum_weights=kahan_cumsum(weight), k=N)
    ind.sort()
    delta = np.random.multivariate_normal([0] * d, cov=cov, size=N)
    for i, j in enumerate(ind):
        xp = [a + b for a, b in zip(x[j], delta[i])]
        if inside(xp):
            fp, *rest = fun(xp)
            if fp > f[j] or p * fp > p * f[j] + math.log(random.uniform(0, 1)):
                x[j] = xp[:]
                f[j] = fp
        x2[i] = x[j][:]
        f2[i] = f[j]
    if End:
        break
    x2, x, f2, f, out2, out = x, x2, f, f2, out, out2

xmean = [statistics.fmean(x[i] for x in x2) for i in range(d)]
print(xmean)
# print(fun((1.8577, -1.7009, 1.8435), xd, yd))
