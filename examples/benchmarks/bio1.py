import graph
import matplotlib.pylab as plt
import subprocess
import sys
import mpi4py.MPI
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
    except subprocess.CalledProcessError:
        sys.stderr.write("bio1.py: command '%s' failed\n" % command)
        exit(1)
    output = output.decode()
    try:
        volume = float(output)
    except ValueError:
        sys.stderr.write("bio1.py: not a float '%s'\n" % output)
        exit(1)
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
        sys.stderr.write("usage bio1.py [-s] [-v] [-o] -d draws\n")
        sys.exit(2)
    elif sys.argv[0][1] == "d":
        sys.argv.pop(0)
        if not sys.argv:
            sys.stderr.write("bio1.py: error: option -d needs an argument\n")
            sys.exit(2)
        try:
            draws = int(sys.argv[0])
        except ValueError:
            sys.stderr.write("bio1: not an integer '%s'\n" % sys.argv[0])
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
sys.argv.append('')
if draws == None:
    sys.stderr.write("bio0.py: -d is not set\n")
    sys.exit(2)

lo = (0.1, 1)
hi = (0.5, 5)
start = timeit.default_timer()
samples, S = graph.korali(fun,
                          draws=draws,
                          lo=lo,
                          hi=hi,
                          return_evidence=True,
                          comm=mpi4py.MPI.COMM_WORLD)
end = timeit.default_timer()
if mpi4py.MPI.COMM_WORLD.Get_rank() == mpi4py.MPI.COMM_WORLD.Get_size() - 1:
    if Samples:
        for k1, mu in samples:
            print("%.16e %.16e" % (k1, mu))
    else:
        print(mpi4py.MPI.COMM_WORLD.Get_size(), end - start)
