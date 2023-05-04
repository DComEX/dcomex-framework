import graph
import math
import matplotlib.pylab as plt
import random
import kahan


def fun(x):
    a, b = x
    return -1 / 2 * (a**2 * 5 / 4 + b**2 * 5 / 4 + a * b / 2 - (a + b) * y)


def dfun(x):
    a, b = x
    return (2 * y - b - 5 * a) / 4, (2 * y - 5 * b - a) / 4


def direct(draws):
    for i in range(draws):
        a = random.gauss(y / 3, math.sqrt(5 / 6))
        yield [random.gauss((2 * y - a) / 5, math.sqrt(4 / 5))]


random.seed(123456)
y = 4.3
init = (y / 3, y / 3)
sigma = 1.46
scale = (1, 1)
draws = 1000
for samples, label in ((graph.metropolis(fun, draws, init, scale, log=True),
                        "metropolis"), (graph.langevin(fun,
                                                       draws,
                                                       init,
                                                       dfun,
                                                       sigma,
                                                       log=True), "langevin"),
                       (direct(draws), "direct")):
    mean = kahan.cummean(a for (a, *rest) in samples)
    plt.plot(list(mean), label=label)
    plt.ylim(1.15, 1.65)
plt.axhline(init[0], color='k')
plt.legend()
plt.savefig("langevin4.png")
