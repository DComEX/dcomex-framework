.. _integration:

Integration
===========

This section describes the integration of MSolve with Korali.

Approach
########

The integration of MSolve into Korali is performed through files.
To execute one MSolve simulation, corresponding to a Korali sample, Korali performs the following:

1. Create a directory unique to the sample.
2. Create an xml configuration file that contains the parameters values. This file is readable by MSolve and describes the simulation.
3. Execute MSolve. Msolve produces an xml output file containing the results of the simulations.
4. Parse the output file and extract the quantities of interest needed by the optimization or sampling process.


Python module
#############

Here we describe the bridge python module and its functions.

Summary
-------

.. autosummary::
   integration.bridge.run_msolve
   integration.bridge.write_config_file
   integration.bridge.run_msolve_mock


Interface with MSolve
---------------------

.. autofunction:: integration.bridge.run_msolve

Utilities
---------

|

.. autofunction:: integration.bridge.write_config_file

|

.. autofunction:: integration.bridge.run_msolve_mock
