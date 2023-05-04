import korali
import multiprocessing
import os
import statistics
import subprocess
import sys
import xml.dom.minidom


def model(ks):
    k1, mu, sigma = ks["Parameters"]
    path = "%05d.out" % int(ks["Sample Id"])
    os.makedirs(path, exist_ok=True)
    wd = os.getcwd()
    os.chdir(path)
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
        raise Exception("bio_korali: msolve_bio failed for parameters %s" %
                        str(ks))
    document = xml.dom.minidom.parse("result.xml")
    os.chdir(wd)
    Vtag = document.getElementsByTagName("Volume")
    if not Vtag:
        raise Exception(
            "bio_korali: result.xml does not have Volume for parameters %s" %
            str(args))
    V = float(Vtag[0].childNodes[0].nodeValue)
    ks["Reference Evaluations"] = [V]
    ks["Standard Deviation"] = [sigma]


num_cores = multiprocessing.cpu_count()
e = korali.Experiment()
e["Random Seed"] = 12345
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = [1.0]
e["Problem"]["Computational Model"] = model
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 500 * num_cores
e["Distributions"][0]["Name"] = "Uniform01"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.0
e["Distributions"][0]["Maximum"] = 1.0
e["Distributions"][1]["Name"] = "Prior sigma"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.0
e["Distributions"][1]["Maximum"] = 1e-1
e["Variables"][0]["Name"] = "k1"
e["Variables"][0]["Prior Distribution"] = "Uniform01"
e["Variables"][1]["Name"] = "mu"
e["Variables"][1]["Prior Distribution"] = "Uniform01"
e["Variables"][2]["Name"] = "sigma"
e["Variables"][2]["Prior Distribution"] = "Prior sigma"
e["Console Output"]["Verbosity"] = "Detailed"
k = korali.Engine()
if num_cores > 1:
    k["Conduit"]["Type"] = "Concurrent"
    k["Conduit"]["Concurrent Jobs"] = num_cores

k.run(e)
samples = e["Results"]["Posterior Sample Database"]
print(statistics.fmean(mu + 2 * k1 for k1, mu, sigma in samples))
