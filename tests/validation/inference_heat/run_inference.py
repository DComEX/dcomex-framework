#!/usr/bin/python3

import argparse
import korali
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.join('..', '..', '..'))

from integration.bridge import run_msolve


def load_data(filename: str):
    df = pd.read_csv(filename)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    T = df["T"].to_numpy()
    return x, y, T


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cores",
                        type=int,
                        default=1,
                        help="number of cores used by korali.")
    parser.add_argument("--num-samples",
                        type=int,
                        default=1000,
                        help="number of TMCMC samples per generation.")
    args = parser.parse_args()
    num_cores = args.num_cores

    xcoords, ycoords, Tref = load_data("data.csv")

    def model(ks):
        theta1, theta2, sigma = ks["Parameters"]
        generation = int(ks["Current Generation"])
        sample_id = int(ks["Sample Id"])

        parameters = [theta1, theta2]

        x, y, T = run_msolve(xcoords=xcoords,
                             ycoords=ycoords,
                             generation=generation,
                             sample_id=sample_id,
                             parameters=parameters)

        ks["Reference Evaluations"] = T
        ks["Standard Deviation"] = [sigma] * len(T)

    e = korali.Experiment()

    e["Random Seed"] = 12345
    e["Problem"]["Type"] = "Bayesian/Reference"

    e["Problem"]["Type"] = "Bayesian/Reference"
    e["Problem"]["Likelihood Model"] = "Normal"
    e["Problem"]["Reference Data"] = Tref.tolist()
    e["Problem"]["Computational Model"] = model

    e["Solver"]["Type"] = "Sampler/TMCMC"
    e["Solver"]["Population Size"] = args.num_samples

    e["Distributions"][0]["Name"] = "Uniform01"
    e["Distributions"][0]["Type"] = "Univariate/Uniform"
    e["Distributions"][0]["Minimum"] = 0.0
    e["Distributions"][0]["Maximum"] = 1.0

    e["Distributions"][1]["Name"] = "Prior sigma"
    e["Distributions"][1]["Type"] = "Univariate/Uniform"
    e["Distributions"][1]["Minimum"] = 0.0
    e["Distributions"][1]["Maximum"] = 1e-1

    e["Variables"][0]["Name"] = "theta1"
    e["Variables"][0]["Prior Distribution"] = "Uniform01"

    e["Variables"][1]["Name"] = "theta2"
    e["Variables"][1]["Prior Distribution"] = "Uniform01"

    e["Variables"][2]["Name"] = "sigma"
    e["Variables"][2]["Prior Distribution"] = "Prior sigma"

    e["Console Output"]["Verbosity"] = "Detailed"

    k = korali.Engine()

    if num_cores > 1:
        k["Conduit"]["Type"] = "Concurrent"
        k["Conduit"]["Concurrent Jobs"] = num_cores

    k.run(e)

    # check the results
    samples = np.array(e["Results"]["Posterior Sample Database"])

    mean_theta1 = np.mean(samples[:, 0])
    mean_theta2 = np.mean(samples[:, 1])

    ref_theta1 = 0.2
    ref_theta2 = 0.7

    tolerance = 1e-1

    print("Inferred parameters:")
    print(f"theta1 = {mean_theta1}, theta2 = {mean_theta2}")

    assert abs(ref_theta1 - mean_theta1) < tolerance
    assert abs(ref_theta2 - mean_theta2) < tolerance
