from xml.dom.minidom import parse
import graph
import itertools
import matplotlib.pylab as plt
import multiprocessing
import os
import random
import subprocess
import sys


def fun(args):
    k1, mu = args
    mphtxt = os.path.join(os.environ["HOME"], ".local", "share",
                          "MeshCyprusTM.mphtxt")
    with open("config.xml", "w") as file:
        file.write('''<MSolve4Korali version="1.0">
        <Mesh>
                <File>%s</File>
        </Mesh>
        <Physics type="TumorGrowth">
                <Time>20</Time>
                <Timestep>1</Timestep>
        </Physics>
        <Output>
                <TumorVolume/>
        </Output>
        <Parameters>
                <k1>%.16e</k1>
                <mu>%.16e</mu>
        </Parameters>
</MSolve4Korali>
''' % (mphtxt, k1, mu))
    with open("stdout", "w") as stdout, open("stderr", "w") as stderr:
        rc = subprocess.call(
            ["ISAAR.MSolve.MSolve4Korali", "config.xml", "result.xml"],
            stdout=stdout,
            stderr=stderr)
    if rc != 0:
        sys.stderr.write("bio.py: msolve_bio failed for parameters %s" %
                         str(args))
        return -float("inf")
    document = parse("result.xml")
    SolutionMsgTag = document.getElementsByTagName("SolutionMsg")
    if not SolutionMsgTag:
        raise Exception(
            "bio.py: result.xml does not have SolutionMsg for parameters %s" %
            str(args))
    SolutionMsg = SolutionMsgTag[0].childNodes[0].nodeValue
    if SolutionMsg == "Fail":
        return -float("inf")
    elif SolutionMsg == "Success":
        Vtag = document.getElementsByTagName("Volume")
        if not Vtag:
            raise Exception(
                "bio.py: result.xml does not have Volume for parameters %s" %
                str(args))
        V = float(Vtag[0].childNodes[0].nodeValue)
        return -(V - 1e-6)**2
    else:
        raise Exception("bio.py: unknown SolutionMsg: '%s'\n" % SolutionMsg)


def worker(i):
    random.seed(os.getpid())
    path = "%05d.out" % i
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    return list(
        graph.metropolis(fun,
                         draws,
                         init=[0.5, 0.3],
                         scale=[0.01, 0.01],
                         log=True))


draws = int(sys.argv[1])
np = multiprocessing.cpu_count() if len(sys.argv) < 3 else int(sys.argv[2])
if np == 1:
    samples = [worker(0)]
else:
    pool = multiprocessing.Pool(np)
    samples = pool.map(worker, range(np))
plt.plot(list(itertools.chain(*samples)), 'o', alpha=0.1)
plt.savefig("bio.png")
