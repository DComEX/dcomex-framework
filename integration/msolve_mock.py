#!/usr/bin/env python3
"""
A standalone program that mimicks msolve interface.
"""

import argparse
import numpy as np
from xml.dom.minidom import parse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    document = parse(input_file)

    x = []
    y = []
    for node in document.getElementsByTagName("Temperature"):
        x.append(float(node.getAttribute("X")))
        y.append(float(node.getAttribute("Y")))

    x = np.array(x)
    y = np.array(y)

    theta1 = float(
        document.getElementsByTagName("Theta1")[0].childNodes[0].nodeValue)
    theta2 = float(
        document.getElementsByTagName("Theta2")[0].childNodes[0].nodeValue)

    # just a linear model
    T = theta1 * x + theta2 * y

    # write the results

    with open(output_file, "w") as f:

        print("""<?xml version="1.0" encoding="utf-8"?>
<MSolve4Korali_output version="1.0">
    <Temperatures>""",
              file=f)

        for x_, y_, T_ in zip(x, y, T):
            print(f'        <Temperature X="{x_}" Y="{y_}">{T_}</Temperature>',
                  file=f)

        print("""    </Temperatures>
</MSolve4Korali_output>""", file=f)
