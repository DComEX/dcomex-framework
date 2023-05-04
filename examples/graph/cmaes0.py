import math
import graph
import matplotlib.pyplot as plt
import random
import sys
import scipy.linalg


def print0(l):
    for i in l:
        sys.stdout.write("%+7.2e " % i)
    sys.stdout.write("\n")


def frandom(x):
    return random.uniform(0, 1)


def sphere(x):
    return math.fsum(e**2 for e in x)


def flower(x):
    a, b, c = 1, 1, 4
    return a * math.sqrt(x[0]**2 + x[1]**2) + b * math.sin(
        c * math.atan2(x[1], x[0]))


def elli(x):
    n = len(x)
    return math.fsum(1e6**(i / (n - 1)) * x[i]**2 for i in range(n))


def rosen(x):
    alpha = 100
    return sum(alpha * (x[:-1]**2 - x[1:])**2 + (1 - x[:-1])**2)


def cigar(x):
    return x[0]**2 + 1e6 * math.fsum(e**2 for e in x[1:])


random.seed(1234)
print0(graph.cmaes(elli, (2, 2, 2, 2), 1, 167))
print0(graph.cmaes(sphere, 8 * [1], 1, 100))
print0(graph.cmaes(cigar, 8 * [1], 1, 300))
print0(graph.cmaes(rosen, 8 * [0], 0.5, 439))
print0(graph.cmaes(flower, (1, 1), 1, 200))

trace = graph.cmaes(elli, 10 * [0.1], 0.1, 600, trace=True)
nfev, fmin, xmin, sigma, C, *rest = zip(*trace)
plt.figure()
plt.yscale("log")
plt.xlabel("number of function evaluations")
plt.ylabel("fmin")
plt.plot(nfev, fmin)

plt.figure()
plt.xlabel("number of function evaluations")
plt.ylabel("object variables")
for x in zip(*xmin):
    plt.plot(nfev, x)

plt.figure()
plt.xlabel("number of function evaluations")
plt.ylabel("sigma")
plt.yscale("log")
plt.plot(nfev, sigma)

ratio, scale = [], []
for c in C:
    w = [math.sqrt(e) for e in scipy.linalg.eigvalsh(c)]
    scale.append(w)
    ratio.append(w[-1] / w[0])

plt.figure()
plt.xlabel("number of function evaluations")
plt.ylabel("axis ratio")
plt.yscale("log")
plt.plot(nfev, ratio)

plt.figure()
plt.xlabel("number of function evaluations")
plt.ylabel("scales")
plt.yscale("log")
for s in zip(*scale):
    plt.plot(nfev, s)
plt.savefig("cmaes0.png")
