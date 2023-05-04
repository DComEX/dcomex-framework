import graph
import math
import random
import scipy.stats
import statistics
import sys
import matplotlib.pylab as plt


def fun(x, a, b, w, dev):
    coeff = -1 / (2 * dev**2)
    u = coeff * statistics.fsum((e - d)**2 for e, d in zip(x, a))
    v = coeff * statistics.fsum((e - d)**2 for e, d in zip(x, b))
    return scipy.special.logsumexp((u + math.log(w), v + math.log(1 - w)))


random.seed(12345)
beta = float(sys.argv[1])
dev = 0.1
w = 0.9
draws = 500
trace = graph.tmcmc(lambda x: fun(x, [0.5, 0.5], [-0.5, -0.5], 0.9, 0.1),
                    draws, [-2, -2], [2, 2],
                    beta=beta,
                    trace=True)
for i, (x, accept) in enumerate(trace):
    plt.xlim(-2.1, 2.1)
    plt.ylim(-2.1, 2.1)
    plt.gca().set_ymargin(0)
    plt.gca().set_xmargin(0)
    plt.gca().set_axis_off()
    plt.gca().set_aspect('equal')
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.scatter(*zip(*x), alpha=0.1, edgecolor='none', color='k')
    plt.scatter((0.5, -0.5), (0.5, -0.5), marker='x', color='r')
    plt.title("accept ratio: %6.2f" % (accept / draws))
    plt.savefig("%03d.png" % i)
    plt.close()
