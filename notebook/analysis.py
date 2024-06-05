import glob
import matplotlib.pyplot as plt
import numpy as np
import utils
import os

D1 = [utils.read(path) for path in glob.glob("1/*/MSolveInput.xml")]
D2 = [utils.read(path) for path in glob.glob("2/*/MSolveInput.xml")]
for (k1, mu, sv, *rest), time, volume in D2:
    path = "k1:%.3e/mu:%.3e/sv:%.3e" % (k1, mu, sv)
    time = np.divide(time, time[-1])
    volume = np.divide(volume, volume[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for t0, t1, v0, v1 in zip(time, time[1:], volume, volume[1:]):
            f.write("%.16e %.16e %.16e\n" % (t0, v0, (v1 - v0) / (t1 - t0)))
