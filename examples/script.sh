#!/bin/sh
#SBATCH --constraint gpu
#SBATCH --ntasks 2
#SBATCH --time 10
#SBATCH --cpus-per-task 2

. /etc/profile
i='jfrog.svc.cscs.ch/contbuild/testing/anfink/3810120997072523/public/base/dcomex-framework:72387487716b188e'
d=24
module load sarus
sarus pull $i
srun sarus run --mpi $i python3 /src/examples/benchmarks/bio1.py -d $d
