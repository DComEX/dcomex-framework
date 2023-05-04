import pickle
import numpy as np
import scipy.integrate
import matplotlib.pylab as plt
import statistics
with open("coins.samples.pkl", "rb") as f:
    samples = pickle.load(f)
print(statistics.fmean(e[0] for e in samples))
print(statistics.fmean(e[1] for e in samples))
plt.ylim((0, 0.15))
plt.yticks([])
plt.hist([e[0] for e in samples],
         100,
         density=True,
         histtype='step',
         linewidth=2)
plt.hist([e[1] for e in samples],
         100,
         density=True,
         histtype='step',
         linewidth=2)
plt.savefig("coins.post.png")
