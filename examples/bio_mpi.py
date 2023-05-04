from xml.dom.minidom import parse
import graph
import mpi4py.MPI
import os
import pickle
import random
import subprocess


def fun(args):
    k1, mu = args
    with open("config.xml", "w") as file:
        file.write('''<MSolve4Korali version="1.0">
        <Mesh>
                <File>MeshCyprusTM.mphtxt</File>
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
''' % (k1, mu))
    rc = subprocess.call(["msolve_bio"])
    if rc != 0:
        raise Exception("bio_mpi.py: msolve_bio failed for parameters %s" %
                        str(args))
    document = parse("result.xml")
    Vtag = document.getElementsByTagName("Volume")
    if not Vtag:
        raise Exception(
            "bio_mpi.py: result.xml does not have Volume for parameters %s" %
            str(args))
    V = float(Vtag[0].childNodes[0].nodeValue)
    return -(V - 1.0)**2


size = mpi4py.MPI.COMM_WORLD.Get_size()
rank = mpi4py.MPI.COMM_WORLD.Get_rank()
random.seed(os.getpid())
path = "%05d.out" % rank
os.makedirs(path, exist_ok=True)
os.chdir(path)
samples = list(
    graph.metropolis(fun, 500, init=[0, 0], scale=[0.5, 0.5], log=True))
with open("samples.pkl", "wb") as file:
    pickle.dump(samples, file)
mpi4py.MPI.COMM_WORLD.Barrier()
if rank == 0:
    os.chdir("..")
    for i in range(1, size):
        path = "%05d.out/samples.pkl" % i
        with open(path, 'rb') as file:
            samples.extend(pickle.load(file))
    print(statistics.fmean(mu + 2 * k1
                           for k1, mu in itertools.chain(*samples)))
