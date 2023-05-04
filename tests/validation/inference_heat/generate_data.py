#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.join('..', '..', '..'))

from integration.bridge import run_msolve


def generate_data(theta1: float = 0.2,
                  theta2: float = 0.7,
                  num_measurements: int = 50,
                  sigma: float = 1e-4,
                  seed: int = 12345):

    np.random.seed(seed)
    x = np.random.uniform(0, 1, num_measurements)
    y = np.random.uniform(0, 1, num_measurements)

    x, y, T = run_msolve(xcoords=x,
                         ycoords=y,
                         generation=0,
                         sample_id=0,
                         parameters=[theta1, theta2])

    T = np.random.normal(loc=T, scale=sigma)
    return x, y, T


if __name__ == '__main__':

    x, y, T = generate_data()

    df = pd.DataFrame({"x": x, "y": y, "T": T})
    df.to_csv("data.csv", index=False)
