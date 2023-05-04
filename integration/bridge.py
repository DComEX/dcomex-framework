from xml.dom.minidom import parse
import os
import subprocess

script_dir = os.path.abspath(os.path.dirname(__file__))


def write_config_file(xcoords, ycoords, parameters: list, filename: str):
    """
    Write an xml configuration file readable by MSolve.
    The configuration describes the problem of a 2D plate
    heated by a heat source, with temperature measurements
    at given locations.

    :param list xcoords: x coordinates of the temperature measurements
    :param list ycoords: y coordinates of the temperature measurements
    :param list parameters: position of the heat source
    :param str filename: name of the configuration file to write
    """

    theta1, theta2 = parameters

    with open(filename, "w") as f:
        print("""<MSolve4Korali version="1.0">
	<Mesh>
		<LengthX>1</LengthX>
		<LengthY>1</LengthY>
		<ElementsX>10</ElementsX>
		<ElementsY>10</ElementsY>
	</Mesh>
	<Physics type="Thermal">
		<CommonThickness>1</CommonThickness>
		<Density>1</Density>
		<SpecialHeatCoefficient>1</SpecialHeatCoefficient>
		<Conductivity>1</Conductivity>
		<TemperatureAtBoundaries>0</TemperatureAtBoundaries>
		<HeatSourceMagnitude>10</HeatSourceMagnitude>
		<HeatSourceSpread>0.01</HeatSourceSpread>
	</Physics>
	<Output>""",
              file=f)

        for x, y in zip(xcoords, ycoords):
            print(f'        <Temperature X="{x}" Y="{y}"/>', file=f)

        print(f"""	</Output>
	<Parameters>
		<Theta1>{theta1}</Theta1>
		<Theta2>{theta2}</Theta2>
	</Parameters>
</MSolve4Korali>""",
              file=f)


def parse_msolve_results(filename: str):
    document = parse(filename)
    x = []
    y = []
    T = []
    for node in document.getElementsByTagName("Temperature"):
        x.append(float(node.getAttribute("X")))
        y.append(float(node.getAttribute("Y")))
        T.append(float(node.childNodes[0].nodeValue))
    return x, y, T


def run_msolve_mock(xcoords, ycoords, generation: int, sample_id: int,
                    parameters: list):
    """
    Run a mock version of msolve.
    For debugging and testing purpose.
    """

    basedir = os.getcwd()
    run_dir = os.path.join(basedir, f"gen_{str(generation).zfill(6)}",
                           f"sample_{str(sample_id).zfill(6)}")

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    input_fname = os.path.join(run_dir, "config.xml")
    output_fname = os.path.join(run_dir, "result.xml")
    stdout_fname = os.path.join(run_dir, "stdout.txt")
    stderr_fname = os.path.join(run_dir, "stderr.txt")

    write_config_file(xcoords, ycoords, parameters, input_fname)

    with open(stdout_fname, "w") as stdout_file, \
         open(stderr_fname, "w") as stderr_file:

        subprocess.call([
            'python3',
            os.path.join(script_dir, 'msolve_mock.py'), '--input-file',
            input_fname, '--output-file', output_fname
        ],
                        stdout=stdout_file,
                        stderr=stderr_file)

    x, y, T = parse_msolve_results(output_fname)

    os.chdir(basedir)
    return x, y, T


def run_msolve(xcoords, ycoords, generation: int, sample_id: int,
               parameters: list):
    """
    Run an instance of Msolve for given parameters.
    This function performs the following steps:
    - Create a directory unique to the sample
    - Create an xml configuration file readable by Msolve
    - Run Msolve
    - Parse the xml file produced by MSolve
    - Return the parsed results

    :param list xcoords: x coordinates of the temperature measurements
    :param list ycoords: y coordinates of the temperature measurements
    :param int generation: the generation index of the korali experiment (used to create a unique directory)
    :param int sample_id: the index of the korali sample (used to create a unique directory)
    :param list parameters: position of the heat source
    :return: the coordinates and values of the temperature measurements (x, y, T)
    """

    basedir = os.getcwd()
    run_dir = os.path.join(basedir, f"gen_{str(generation).zfill(6)}",
                           f"sample_{str(sample_id).zfill(6)}")

    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    input_fname = os.path.join(run_dir, "config.xml")
    output_fname = os.path.join(run_dir, "result.xml")
    stdout_fname = os.path.join(run_dir, "stdout.txt")
    stderr_fname = os.path.join(run_dir, "stderr.txt")

    write_config_file(xcoords, ycoords, parameters, input_fname)

    msolve_path = os.path.join('ISAAR.MSolve.MSolve4Korali')
    with open(stdout_fname, "w") as stdout_file, \
         open(stderr_fname, "w") as stderr_file:

        subprocess.call([msolve_path, input_fname, output_fname],
                        stdout=stdout_file,
                        stderr=stderr_file)

    x, y, T = parse_msolve_results(output_fname)

    os.chdir(basedir)
    return x, y, T
