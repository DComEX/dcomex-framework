from distutils.core import setup, Extension
import subprocess
import sys
import re
import os
try:
    ret = subprocess.run(("gsl-config", "--libs"),
                         capture_output=True, check=True,
                         timeout=100)
except (FileNotFoundError, subprocess.CalledProcessError) as e:
    sys.stderr.write("%s: error: %s\n" % (sys.argv[0], e))
    sys.exit(2)
if len(ret.stdout) < 2 or ret.stdout[0] != b"-" or ret.stdout[1] != b"L":
    i = ret.stdout.find(b" ")
    lib_dir = ret.stdout[2:i]
    libs = ret.stdout[i:]
else:
    sys.stderr.write("%s: error: wrong replay from gsl-config: %s\n" %
                     (sys.argv[0], ret.stdout))
    sys.exit(2)
lib_pathes = [
    os.path.join(lib_dir, b"lib" + e[2:] + b".a") for e in libs.split()
    if e[:2] == b"-l" and e != b"-lm"
]
for lib in lib_pathes:
    if not os.path.isfile(lib):
        sys.stdout.write("%s: error: no static library '%s'\n" %
                         (sys.argv[0], lib))
        sys.exit(2)
setup(ext_modules=[
    Extension(
        name="libkorali",
        include_dirs=["source", "."],
        extra_objects=[e.decode() for e in lib_pathes],
        sources=[
            "source/auxiliar/fs.cpp",
            "source/auxiliar/jsonInterface.cpp",
            "source/auxiliar/koraliJson.cpp",
            "source/auxiliar/kstring.cpp",
            "source/auxiliar/libco/libco.c",
            "source/auxiliar/logger.cpp",
            "source/auxiliar/math.cpp",
            "source/auxiliar/MPIUtils.cpp",
            "source/auxiliar/reactionParser.cpp",
            "source/auxiliar/rtnorm/rtnorm.cpp",
            "source/engine.cpp",
            "source/modules/conduit/concurrent/concurrent.cpp",
            "source/modules/conduit/conduit.cpp",
            "source/modules/conduit/distributed/distributed.cpp",
            "source/modules/conduit/sequential/sequential.cpp",
            "source/modules/distribution/distribution.cpp",
            "source/modules/distribution/multivariate/multivariate.cpp",
            "source/modules/distribution/multivariate/normal/normal.cpp",
            "source/modules/distribution/specific/multinomial/multinomial.cpp",
            "source/modules/distribution/specific/specific.cpp",
            "source/modules/distribution/univariate/beta/beta.cpp",
            "source/modules/distribution/univariate/cauchy/cauchy.cpp",
            "source/modules/distribution/univariate/exponential/exponential.cpp",
            "source/modules/distribution/univariate/gamma/gamma.cpp",
            "source/modules/distribution/univariate/geometric/geometric.cpp",
            "source/modules/distribution/univariate/igamma/igamma.cpp",
            "source/modules/distribution/univariate/laplace/laplace.cpp",
            "source/modules/distribution/univariate/logNormal/logNormal.cpp",
            "source/modules/distribution/univariate/normal/normal.cpp",
            "source/modules/distribution/univariate/poisson/poisson.cpp",
            "source/modules/distribution/univariate/truncatedNormal/truncatedNormal.cpp",
            "source/modules/distribution/univariate/uniformratio/uniformratio.cpp",
            "source/modules/distribution/univariate/uniform/uniform.cpp",
            "source/modules/distribution/univariate/univariate.cpp",
            "source/modules/distribution/univariate/weibull/weibull.cpp",
            "source/modules/experiment/experiment.cpp",
            "source/modules/module.cpp",
            "source/modules/neuralNetwork/layer/activation/activation.cpp",
            "source/modules/neuralNetwork/layer/convolution/convolution.cpp",
            "source/modules/neuralNetwork/layer/deconvolution/deconvolution.cpp",
            "source/modules/neuralNetwork/layer/input/input.cpp",
            "source/modules/neuralNetwork/layer/layer.cpp",
            "source/modules/neuralNetwork/layer/linear/linear.cpp",
            "source/modules/neuralNetwork/layer/output/output.cpp",
            "source/modules/neuralNetwork/layer/pooling/pooling.cpp",
            "source/modules/neuralNetwork/layer/recurrent/gru/gru.cpp",
            "source/modules/neuralNetwork/layer/recurrent/lstm/lstm.cpp",
            "source/modules/neuralNetwork/layer/recurrent/recurrent.cpp",
            "source/modules/neuralNetwork/neuralNetwork.cpp",
            "source/modules/problem/bayesian/bayesian.cpp",
            "source/modules/problem/bayesian/custom/custom.cpp",
            "source/modules/problem/bayesian/reference/reference.cpp",
            "source/modules/problem/design/design.cpp",
            "source/modules/problem/hierarchical/hierarchical.cpp",
            "source/modules/problem/hierarchical/psi/psi.cpp",
            "source/modules/problem/hierarchical/thetaNew/thetaNew.cpp",
            "source/modules/problem/hierarchical/theta/theta.cpp",
            "source/modules/problem/integration/integration.cpp",
            "source/modules/problem/optimization/optimization.cpp",
            "source/modules/problem/problem.cpp",
            "source/modules/problem/propagation/propagation.cpp",
            "source/modules/problem/reaction/reaction.cpp",
            "source/modules/problem/reinforcementLearning/continuous/continuous.cpp",
            "source/modules/problem/reinforcementLearning/discrete/discrete.cpp",
            "source/modules/problem/reinforcementLearning/reinforcementLearning.cpp",
            "source/modules/problem/sampling/sampling.cpp",
            "source/modules/problem/supervisedLearning/supervisedLearning.cpp",
            "source/modules/solver/agent/agent.cpp",
            "source/modules/solver/agent/continuous/continuous.cpp",
            "source/modules/solver/agent/continuous/VRACER/VRACER.cpp",
            "source/modules/solver/agent/discrete/discrete.cpp",
            "source/modules/solver/agent/discrete/dVRACER/dVRACER.cpp",
            "source/modules/solver/deepSupervisor/deepSupervisor.cpp",
            "source/modules/solver/deepSupervisor/optimizers/fAdaBelief/fAdaBelief.cpp",
            "source/modules/solver/deepSupervisor/optimizers/fAdaGrad/fAdaGrad.cpp",
            "source/modules/solver/deepSupervisor/optimizers/fAdam/fAdam.cpp",
            "source/modules/solver/deepSupervisor/optimizers/fGradientBasedOptimizer.cpp",
            "source/modules/solver/deepSupervisor/optimizers/fMadGrad/fMadGrad.cpp",
            "source/modules/solver/designer/designer.cpp",
            "source/modules/solver/executor/executor.cpp",
            "source/modules/solver/integrator/integrator.cpp",
            "source/modules/solver/integrator/montecarlo/MonteCarlo.cpp",
            "source/modules/solver/integrator/quadrature/Quadrature.cpp",
            "source/modules/solver/optimizer/AdaBelief/AdaBelief.cpp",
            "source/modules/solver/optimizer/Adam/Adam.cpp",
            "source/modules/solver/optimizer/CMAES/CMAES.cpp",
            "source/modules/solver/optimizer/DEA/DEA.cpp",
            "source/modules/solver/optimizer/gridSearch/gridSearch.cpp",
            "source/modules/solver/optimizer/MADGRAD/MADGRAD.cpp",
            "source/modules/solver/optimizer/MOCMAES/MOCMAES.cpp",
            "source/modules/solver/optimizer/optimizer.cpp",
            "source/modules/solver/optimizer/Rprop/Rprop.cpp",
            "source/modules/solver/sampler/HMC/HMC.cpp",
            "source/modules/solver/sampler/MCMC/MCMC.cpp",
            "source/modules/solver/sampler/Nested/Nested.cpp",
            "source/modules/solver/sampler/sampler.cpp",
            "source/modules/solver/sampler/TMCMC/TMCMC.cpp",
            "source/modules/solver/solver.cpp",
            "source/modules/solver/SSM/SSA/SSA.cpp",
            "source/modules/solver/SSM/SSM.cpp",
            "source/modules/solver/SSM/TauLeaping/TauLeaping.cpp",
            "source/sample/sample.cpp",
        ]),
])
