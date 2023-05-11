#!/bin/env python3

import graph
import subprocess
import sys
import timeit


def fun(x):
    time = 1
    k1, mu = x
    if Verbose:
        sys.stderr.write("%s\n" % x)
    command = ["bio"]
    if Surrogate:
        command.append("-s")
    command.append("%.16e" % k1)
    command.append("%.16e" % mu)
    command.append("%d" % time)
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
    except FileNotFoundError as e:
        sys.stderr.write("bio1.py: error: '%s' command not found\n" % e.filename)
        sys.exit(e.errno)
    except subprocess.CalledProcessError as e:
        import pprint
        sys.stderr.write("bio1.py: error: command '%s' failed\n" % e.cmd)
        sys.exit(e.returncode)
    output = output.decode()
    try:
        volume = float(output)
    except ValueError:
        sys.stderr.write("bio1.py: error: not a float '%s'\n" % output)
        sys.exit(1)
    sigma = 0.5
    scale = 1e-11
    if Verbose:
        sys.stderr.write("%.16e\n" % volume)
    return -((volume / scale - 5.0)**2 / sigma**2)


Surrogate = False
Verbose = False
draws = None
Samples = False
while True:
    sys.argv.pop(0)
    if not sys.argv or sys.argv[0][0] != "-" or len(sys.argv[0]) < 2:
        break
    if sys.argv[0][1] == "h":
        sys.stderr.write("""\
usage bio1.py [-s] [-v] [-o] -d draws

Benchmark simulation with Korali and MSolve. By default, it prints the
number of the MPI ranks and the time in seconds it takes to draw the
samples.

Options:
  -s                   Use surrogate, do not run MSolve simulation
  -m                   Dumps samples
  -v                   Verbose output
  -h                   Display this help message and exit.

Arguments:
  -d int               The number of samples to draw
""")
        sys.exit(2)
    elif sys.argv[0][1] == "d":
        sys.argv.pop(0)
        if not sys.argv:
            sys.stderr.write(
                "bio1.py: error: option -d requires an argument\n")
            sys.exit(2)
        try:
            draws = int(sys.argv[0])
        except ValueError:
            sys.stderr.write(
                "bio1.py: error: invalid value '%s' for -d option. Provide an integer value\n"
                % sys.argv[0])
            sys.exit(2)
    elif sys.argv[0][1] == "s":
        Surrogate = True
    elif sys.argv[0][1] == "m":
        Samples = True
    elif sys.argv[0][1] == "v":
        Verbose = True
    else:
        sys.stderr.write("bio1.py: error: unknown option '%s'\n" % sys.argv[0])
        sys.exit(2)
if sys.argv:
    sys.stderr.write("bio1.py: error: unknown arguments %s\n" % sys.argv)
    sys.exit(2)
sys.argv.append('')
if draws is None:
    sys.stderr.write(
        "bio1.py: error: -d option is not set. Specify the number of samples to draw\n"
    )
    sys.exit(2)

try:
    import mpi4py.MPI
except ModuleNotFoundError:
    sys.stderr.write("bio1.py: error: no python module mpi4py.MPI\n")
    sys.exit(2)
if mpi4py.MPI.COMM_WORLD.Get_size() < 2:
    sys.stderr.write(
        "bio1.py: error: distributed korali requires at least two MPI ranks\n")
    sys.exit(2)

lo = (0.1, 1)
hi = (0.5, 5)
start = timeit.default_timer()
try:
    samples, S = graph.korali(fun,
                              draws=draws,
                              lo=lo,
                              hi=hi,
                              return_evidence=True,
                              comm=mpi4py.MPI.COMM_WORLD)
except graph.KoraliNotFound:
    if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
        sys.stderr.write("bio1.py: error: graph.korali failed to import korali module\n")
    mpi4py.MPI.Finalize()
    sys.exit(2)
end = timeit.default_timer()
if mpi4py.MPI.COMM_WORLD.Get_rank() == mpi4py.MPI.COMM_WORLD.Get_size() - 1:
    if Samples:
        for k1, mu in samples:
            print("%.16e %.16e" % (k1, mu))
    else:
        print(mpi4py.MPI.COMM_WORLD.Get_size(), end - start)
