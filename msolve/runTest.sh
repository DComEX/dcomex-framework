#!/bin/bash
while getopts i:o: flag
do
    case "${flag}" in
        i) inputFile=${OPTARG};;
        o) outputFile=${OPTARG};;
    esac
done
# This script assumes that it has been called from an MPI execution, e.g. `mpirun -np 2 runTest.sh` or `srun -n2 runTest.sh`
# dotnet test /msolve/MSolveApp/Msolve.One.MPI
/msolve/MSolveApp/ISAAR.MSolve.MSolve4Korali/bin/Debug/net6.0/ISAAR.MSolve.MSolve4Korali $inputFile $outputFile
