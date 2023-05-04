import pickle
import numpy as np
import scipy.integrate
import matplotlib.pylab as plt

with open("samples.pkl", "rb") as f:
    samples = pickle.load(f)
D = np.loadtxt("three.dat")
x = D[:, 0]
y0 = D[:, 1]
I0 = scipy.integrate.trapz(y0, x)
plt.yticks([])
plt.hist([e[0] for e in samples],
         40,
         density=True,
         histtype='step',
         linewidth=2)
plt.plot(x, y0 / I0, '-')
plt.savefig("three.vis.png")
