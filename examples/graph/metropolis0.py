import graph
import kahan
import scipy.stats
import statistics


def fun(x):
    return scipy.stats.multivariate_normal.logpdf(x, mean, cov)


mean = (1, 2, 7)
cov = ((1, 0.25, 0.25), (0.25, 1, 0.25), (0.25, 0.25, 1))
init = mean
scale = (0.75, 0.75, 0.75)
draws = 40000
S0 = list(graph.metropolis(fun, draws, init, scale, log=True))
x0 = kahan.mean(x for x, y, z in S0)
y0 = kahan.mean(y for x, y, z in S0)
z0 = kahan.mean(z for x, y, z in S0)

xx = kahan.mean((x - x0) * (x - x0) for x, y, z in S0)
xy = kahan.mean((x - x0) * (y - y0) for x, y, z in S0)
xz = kahan.mean((x - x0) * (z - z0) for x, y, z in S0)
yy = kahan.mean((y - y0) * (y - y0) for x, y, z in S0)
yz = kahan.mean((y - y0) * (z - z0) for x, y, z in S0)
zz = kahan.mean((z - z0) * (z - z0) for x, y, z in S0)

print("%.2f %.2f %.2f" % (x0, y0, z0))
print("%.2f %.2f %.2f" % (xx, xy, xz))
print("     %.2f %.2f" % (yy, yz))
print("          %.2f" % zz)
