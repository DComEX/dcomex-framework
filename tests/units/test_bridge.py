import korali
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.join('..', '..'))

from integration.bridge import (run_msolve_mock, run_msolve)


def read_temperature(korali_file: str):
    import json

    with open(korali_file, "r") as f:
        doc = json.load(f)

    T = []
    for sample in doc["Samples"]:
        T.append(list(sample["Evaluations"]))

    return np.array(T)


class TestBridge(unittest.TestCase):

    def test_korali_executes_msolve_mock(self):

        xcoords = np.array([0.1, 0.4, 0.9])
        ycoords = np.array([0.2, 0.5, 0.6])

        theta1 = np.array([1.0, 0.2])
        theta2 = np.array([0.2, 0.1])

        def model(ksample):
            generation = int(ksample["Current Generation"])
            sample_id = int(ksample["Sample Id"])
            theta1, theta2 = ksample["Parameters"]
            parameters = [theta1, theta2]

            x, y, T = run_msolve_mock(xcoords=xcoords,
                                      ycoords=ycoords,
                                      generation=generation,
                                      sample_id=sample_id,
                                      parameters=parameters)

            ksample["Evaluations"] = T

        e = korali.Experiment()

        e["Problem"]["Type"] = "Propagation"
        e["Problem"]["Execution Model"] = model

        e['Solver']['Type'] = 'Executor'
        e['Solver']['Executions Per Generation'] = 2

        e["Variables"][0]["Name"] = "theta1"
        e["Variables"][0]["Precomputed Values"] = theta1.tolist()
        e["Variables"][1]["Name"] = "theta2"
        e["Variables"][1]["Precomputed Values"] = theta2.tolist()

        e['Store Sample Information'] = True

        k = korali.Engine()
        k.run(e)

        Tref = theta1[:, np.newaxis] * xcoords[
            np.newaxis, :] + theta2[:, np.newaxis] * ycoords[np.newaxis, :]
        T = read_temperature(os.path.join("_korali_result", "latest"))

        np.testing.assert_array_equal(T, Tref)

    def test_korali_executes_msolve(self):

        xcoords = np.array([0.1, 0.4, 0.9])
        ycoords = np.array([0.2, 0.5, 0.6])

        theta1 = np.array([1.0, 0.2])
        theta2 = np.array([0.2, 0.1])

        def model(ksample):
            generation = int(ksample["Current Generation"])
            sample_id = int(ksample["Sample Id"])
            theta1, theta2 = ksample["Parameters"]
            parameters = [theta1, theta2]

            x, y, T = run_msolve(xcoords=xcoords,
                                 ycoords=ycoords,
                                 generation=generation,
                                 sample_id=sample_id,
                                 parameters=parameters)

            ksample["Evaluations"] = T

        e = korali.Experiment()

        e["Problem"]["Type"] = "Propagation"
        e["Problem"]["Execution Model"] = model

        e['Solver']['Type'] = 'Executor'
        e['Solver']['Executions Per Generation'] = 2

        e["Variables"][0]["Name"] = "theta1"
        e["Variables"][0]["Precomputed Values"] = theta1.tolist()
        e["Variables"][1]["Name"] = "theta2"
        e["Variables"][1]["Precomputed Values"] = theta2.tolist()

        e['Store Sample Information'] = True

        k = korali.Engine()
        k.run(e)

        # The reference was generated manually by calling msolve with
        # the above parameters.
        Tref = np.array([[0.000167, 0.001144, 0.000864],
                         [0.026306, 0.008453, 0.000962]])
        T = read_temperature(os.path.join("_korali_result", "latest"))

        np.testing.assert_array_almost_equal(T, Tref, decimal=5)
