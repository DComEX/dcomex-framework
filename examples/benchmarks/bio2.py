#!/bin/env python3

import itertools
import json
import multiprocessing
import numpy as np
import os
import signal
import subprocess
import sys
import pathlib


class BioException(Exception):
    pass


def bio(k1, mu, sv, time, Surrogate=False, Verbose=False, Label=None):
    cmd = ["bio"]
    if Surrogate:
        cmd.append("-s")
    cmd.append("-r")
    cmd.append(Label if Label != None else str(os.getpid()))
    cmd.append("--")
    cmd.append("%.16e" % k1)
    cmd.append("%.16e" % mu)
    cmd.append("%.16e" % sv)
    for nstep, dt in zip(time[::2], time[1::2]):
        cmd.append("%d" % nstep)
        cmd.append("%.16e" % dt)
    if Verbose:
        sys.stderr.write("%d: %s\n" % (os.getpid(), " ".join(cmd)))
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise BioException(cmd, stderr.decode())
    output = stdout.decode()
    try:
        output = [[float(t) for t in l.split()] for l in output.split("\n")
                  if len(l)]
    except ValueError:
        raise BioException(cmd, output)
    return output


f = float(sys.argv[1])
Time = int(200 * f), 2e-1 / f, int(99 * f), 40.0 / f, int(439 * f), 2000.0 / f
Surrogate = "-s" in sys.argv[1:]
Verbose = True

P = [
    (k1, mu, sv, Time, Surrogate, Verbose, "%s/%08d" % (sys.argv[1], i))
    for i, (k1, mu, sv) in enumerate(
        itertools.product(
            np.geomspace(7.52e-12, 7.52e-10, 8),  #
            np.geomspace(5, 50, 8),  #
            np.geomspace(5000, 20000, 8)))
]
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    try:
        D = pool.starmap(bio, P)
    except BioException as e:
        cmd, stdout = e.args
        sys.stderr.write("bio: error: command '%s' failed\n" % " ".join(cmd))
        sys.stderr.write("%s" % stdout)
        sys.exit(1)
    except KeyboardInterrupt as e:
        sys.stderr.write("bio: KeyboardInterrupt\n")
        sys.exit(1)

with open("bio2,%g.json" % f, "w") as file:
    json.dump([f, Time, P, D], file, indent=4)

import matplotlib.pyplot as plt

sec_in_day = 24 * 60 * 60
for (k1, mu, sv, *rest), d in zip(P, D):
    time, volume = zip(*d)
    plt.plot([time / sec_in_day for time in time],
             [(v - volume[0]) for v in volume], "-")
plt.savefig("bio2,%g.png" % f)
