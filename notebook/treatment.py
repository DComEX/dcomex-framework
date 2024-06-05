import glob
import utils
import numpy as np
import matplotlib.pyplot as plt


def treatment(time, volume, start, ammount):

    def h(x):
        return x**4 / (1 / ammount + x**4)

    def t(x):
        return np.heaviside(time - start * tm, 0.5) * h(x - start)

    tm = max(time)
    V, = np.interp([start * tm], time, volume)
    time = np.asarray(time)
    return volume * (1 - t(time / tm)) + t(time / tm) * V


D = [utils.read(path) for path in glob.glob("1/*/MSolveInput.xml")]
params, time, volume = D[20]
for ammount in 4, 8, 16, 32, 64, 128:
    plt.plot(time, treatment(time, volume, start=0.5, ammount=ammount), 'g', label=f"{ammount}")
plt.xlabel("time, seconds")
plt.ylabel("tumor volume, mm^3")
plt.legend()
plt.plot(time, volume, "r")
plt.savefig("treatment.png", bbox_inches="tight")
