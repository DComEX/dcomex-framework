# One node scaling efficency

We assume that the tumor volume on day $1$ is $V_0$, and we want to
sample the parameters $k_1$ and $\mu$ based on the likelihood function
$N(V_{\text{mosolve}} - V_0,, \sigma = 1/2)$. Here, $N$ denotes the
normal distribution and $\sigma$ is the standard deviation. We use the
TMCMC algorithm with the default settings in Korali.

For strong scaling, we fix the total number of draws to be 48 and vary
the number of logical ranks from 1 to 24. The code used for this
experiment is shown below:

```
for n in 1 2 4 8 12 16 24; do python3 bio0.py -d 48 -n $n; done
```

<p align="center"><img src="strong.png" alt="Strong scaling plot"/></p>

For weak scaling, we fix the number of draws per logical rank to be 24
and increase the total number of logical ranks:

```
for n in 1 2 4 8 12 16 24
do d=`echo $n | awk '{print 24 * $n}'`
   python3 bio0.py -d $d -n $n; done
```

<p align="center"><img src="weak.png" alt="Weaks scaling plot"/></p>

# Multi node scaling efficency

```
$ ssh daint srun -C gpu -A d124 -n 1 lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
Address sizes:       46 bits physical, 48 bits virtual
CPU(s):              24
On-line CPU(s) list: 0-23
Thread(s) per core:  2
Core(s) per socket:  12
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               63
Model name:          Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz
Stepping:            2
CPU MHz:             3110.175
CPU max MHz:         2601.0000
CPU min MHz:         1200.0000
BogoMIPS:            5200.30
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            30720K
NUMA node0 CPU(s):   0-23
```

```
$ cat msolve/run
i='jfrog.svc.cscs.ch/contbuild/testing/anfink/3810120997072523/public/dcomex-framework:08fe8d6c'
module load sarus
sarus pull $i
for n in 2 4 8 16 32 64 128 256 512 1024 2048
do d=`echo $n | awk '{print 24 * $n}'`
      srun -J $n.msolve -o out.$n.log -e err.$n.log -C gpu --cpus-per-task 2 -A d124 -n $n sarus run --mpi $i python3 /src/examples/benchmarks/bio1.py -d $d -v
done
```
