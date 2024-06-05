import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sec_in_day = 24 * 60 * 60

for path in sys.argv[1:]:
    with open(path, "r") as file:
        f, Time, P, D = json.load(file)
    os.makedirs("%s" % f, exist_ok=True)
    for (k1, mu, sv, *rest), d in zip(P, D):
        path = "%s/%.2e,%.2e,%.2e,txt" % (f,k1, mu, sv)
        time, volume = zip(*d)
        time = time[:-1]
        volume = volume[1:]
        with open(path, "w") as file:
            t0, v0 = d[0]
            for t, v in zip(time, volume):
                file.write("%.16e %.16e\n" % (t / sec_in_day, v))
