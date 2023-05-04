import graph
import matplotlib.pylab as plt


def fun(x):
    a, b = x
    return -a**2 - (b / 2)**2


samples, S = graph.korali(fun, 1000, [-5, -4], [5, 4], return_evidence=True)
print("log evidence: ", S)
plt.plot(*zip(*samples), 'o', alpha=0.5)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel("a")
plt.ylabel("b")
plt.savefig("korali0.png")
