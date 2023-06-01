#include "module.hpp"

#include "conduit/concurrent/concurrent.hpp"
#include "conduit/distributed/distributed.hpp"
#include "conduit/sequential/sequential.hpp"
#include "distribution/distribution.hpp"
#include "distribution/multivariate/normal/normal.hpp"
#include "distribution/specific/multinomial/multinomial.hpp"
#include "distribution/specific/specific.hpp"
#include "distribution/univariate/beta/beta.hpp"
#include "distribution/univariate/cauchy/cauchy.hpp"
#include "distribution/univariate/exponential/exponential.hpp"
#include "distribution/univariate/gamma/gamma.hpp"
#include "distribution/univariate/geometric/geometric.hpp"
#include "distribution/univariate/igamma/igamma.hpp"
#include "distribution/univariate/laplace/laplace.hpp"
#include "distribution/univariate/logNormal/logNormal.hpp"
#include "distribution/univariate/normal/normal.hpp"
#include "distribution/univariate/poisson/poisson.hpp"
#include "distribution/univariate/truncatedNormal/truncatedNormal.hpp"
#include "distribution/univariate/uniform/uniform.hpp"
#include "distribution/univariate/uniformratio/uniformratio.hpp"
#include "distribution/univariate/weibull/weibull.hpp"
#include "experiment/experiment.hpp"
#include "neuralNetwork/layer/activation/activation.hpp"
#include "neuralNetwork/layer/convolution/convolution.hpp"
#include "neuralNetwork/layer/deconvolution/deconvolution.hpp"
#include "neuralNetwork/layer/pooling/pooling.hpp"
#include "neuralNetwork/layer/input/input.hpp"
#include "neuralNetwork/layer/layer.hpp"
#include "neuralNetwork/layer/linear/linear.hpp"
#include "neuralNetwork/layer/output/output.hpp"
#include "neuralNetwork/layer/recurrent/gru/gru.hpp"
#include "neuralNetwork/layer/recurrent/lstm/lstm.hpp"
#include "neuralNetwork/neuralNetwork.hpp"
#include "problem/bayesian/custom/custom.hpp"
#include "problem/bayesian/reference/reference.hpp"
#include "problem/design/design.hpp"
#include "problem/hierarchical/psi/psi.hpp"
#include "problem/hierarchical/theta/theta.hpp"
#include "problem/hierarchical/thetaNew/thetaNew.hpp"
#include "problem/integration/integration.hpp"
#include "problem/optimization/optimization.hpp"
#include "problem/problem.hpp"
#include "problem/propagation/propagation.hpp"
#include "problem/reinforcementLearning/continuous/continuous.hpp"
#include "problem/reinforcementLearning/discrete/discrete.hpp"
#include "problem/sampling/sampling.hpp"
#include "problem/supervisedLearning/supervisedLearning.hpp"
#include "problem/reaction/reaction.hpp"
#include "solver/agent/continuous/VRACER/VRACER.hpp"
#include "solver/agent/continuous/continuous.hpp"
#include "solver/agent/discrete/dVRACER/dVRACER.hpp"
#include "solver/agent/discrete/discrete.hpp"
#include "solver/designer/designer.hpp"
#include "solver/executor/executor.hpp"
#include "solver/integrator/integrator.hpp"
#include "solver/integrator/montecarlo/MonteCarlo.hpp"
#include "solver/integrator/quadrature/Quadrature.hpp"
#include "solver/deepSupervisor/deepSupervisor.hpp"
#include "solver/deepSupervisor/optimizers/fAdam/fAdam.hpp"
#include "solver/deepSupervisor/optimizers/fAdaBelief/fAdaBelief.hpp"
#include "solver/deepSupervisor/optimizers/fMadGrad/fMadGrad.hpp"
#include "solver/deepSupervisor/optimizers/fAdaGrad/fAdaGrad.hpp"
#include "solver/optimizer/AdaBelief/AdaBelief.hpp"
#include "solver/optimizer/Adam/Adam.hpp"
#include "solver/optimizer/CMAES/CMAES.hpp"
#include "solver/optimizer/DEA/DEA.hpp"
#include "solver/optimizer/MADGRAD/MADGRAD.hpp"
#include "solver/optimizer/MOCMAES/MOCMAES.hpp"
#include "solver/optimizer/Rprop/Rprop.hpp"
#include "solver/optimizer/gridSearch/gridSearch.hpp"
#include "solver/optimizer/optimizer.hpp"
#include "solver/sampler/HMC/HMC.hpp"
#include "solver/sampler/MCMC/MCMC.hpp"
#include "solver/sampler/Nested/Nested.hpp"
#include "solver/sampler/TMCMC/TMCMC.hpp"
#include "solver/sampler/sampler.hpp"
#include "solver/SSM/SSA/SSA.hpp"
#include "solver/SSM/TauLeaping/TauLeaping.hpp"
#include "solver/SSM/SSM.hpp"

namespace korali
{
knlohmann::json __profiler;
std::chrono::time_point<std::chrono::high_resolution_clock> _startTime;
std::chrono::time_point<std::chrono::high_resolution_clock> _endTime;
double _cumulativeTime;

void Module::initialize() {};
void Module::setEngine(korali::Engine* engine) {_k->_engine = engine;};
void Module::finalize() {};
std::string Module::getType() { return _type; };
bool Module::checkTermination() { return false; };
void Module::getConfiguration(knlohmann::json &js){};
void Module::setConfiguration(knlohmann::json &js){};
void Module::applyModuleDefaults(knlohmann::json &js){};
void Module::applyVariableDefaults(){};
bool Module::runOperation(std::string operation, korali::Sample &sample) { return false; };

Module *Module::getModule(knlohmann::json &js, Experiment *e)
{
  std::string moduleType = "Undefined";

  if (!isDefined(js, "Type"))
    KORALI_LOG_ERROR(" + No module type provided in:\n %s\n", js.dump(2).c_str());

  try
  {
    moduleType = js["Type"].get<std::string>();
  }
  catch (const std::exception &ex)
  {
    KORALI_LOG_ERROR(" + Could not parse module type: '%s'.\n%s", js["Type"].dump(2).c_str(), ex.what());
  }

  moduleType.erase(remove_if(moduleType.begin(), moduleType.end(), isspace), moduleType.end());

  bool isExperiment = false;
  if (js["Type"] == "Experiment") isExperiment = true;

  // Once we've read the module type, we delete this information, because  it is not parsed by the module itself
  eraseValue(js, "Type");

  // Creating module pointer from it's type.
  Module *module = nullptr;
  if (iCompare(moduleType, "Experiment")) module = new korali::Experiment();

  // Conduits
  if (iCompare(moduleType, "Concurrent")) module = new korali::conduit::Concurrent();
  if (iCompare(moduleType, "Distributed")) module = new korali::conduit::Distributed();
  if (iCompare(moduleType, "Sequential")) module = new korali::conduit::Sequential();
  
  // Distributions
  if (iCompare(moduleType, "Multivariate/Normal")) module = new korali::distribution::multivariate::Normal();
  if (iCompare(moduleType, "Specific/Multinomial")) module = new korali::distribution::specific::Multinomial();
  if (iCompare(moduleType, "Univariate/Beta")) module = new korali::distribution::univariate::Beta();
  if (iCompare(moduleType, "Univariate/Cauchy")) module = new korali::distribution::univariate::Cauchy();
  if (iCompare(moduleType, "Univariate/Exponential")) module = new korali::distribution::univariate::Exponential();
  if (iCompare(moduleType, "Univariate/Gamma")) module = new korali::distribution::univariate::Gamma();
  if (iCompare(moduleType, "Univariate/Geometric")) module = new korali::distribution::univariate::Geometric();
  if (iCompare(moduleType, "Univariate/Igamma")) module = new korali::distribution::univariate::Igamma();
  if (iCompare(moduleType, "Univariate/Laplace")) module = new korali::distribution::univariate::Laplace();
  if (iCompare(moduleType, "Univariate/LogNormal")) module = new korali::distribution::univariate::LogNormal();
  if (iCompare(moduleType, "Univariate/Normal")) module = new korali::distribution::univariate::Normal();
  if (iCompare(moduleType, "Univariate/Poisson")) module = new korali::distribution::univariate::Poisson();
  if (iCompare(moduleType, "Univariate/TruncatedNormal")) module = new korali::distribution::univariate::TruncatedNormal();
  if (iCompare(moduleType, "Univariate/Uniform")) module = new korali::distribution::univariate::Uniform();
  if (iCompare(moduleType, "Univariate/UniformRatio")) module = new korali::distribution::univariate::UniformRatio();
  if (iCompare(moduleType, "Univariate/Weibull")) module = new korali::distribution::univariate::Weibull();

  // Problem types
  if (iCompare(moduleType, "Bayesian/Custom")) module = new korali::problem::bayesian::Custom();
  if (iCompare(moduleType, "Bayesian/Reference")) module = new korali::problem::bayesian::Reference();
  if (iCompare(moduleType, "Design")) module = new korali::problem::Design();
  if (iCompare(moduleType, "Hierarchical/Psi")) module = new korali::problem::hierarchical::Psi();
  if (iCompare(moduleType, "Hierarchical/Theta")) module = new korali::problem::hierarchical::Theta();
  if (iCompare(moduleType, "Hierarchical/ThetaNew")) module = new korali::problem::hierarchical::ThetaNew();
  if (iCompare(moduleType, "Integration")) module = new korali::problem::Integration();
  if (iCompare(moduleType, "Optimization")) module = new korali::problem::Optimization();
  if (iCompare(moduleType, "Propagation")) module = new korali::problem::Propagation();
  if (iCompare(moduleType, "Reaction")) module = new korali::problem::Reaction();
  if (iCompare(moduleType, "ReinforcementLearning/Continuous")) module = new korali::problem::reinforcementLearning::Continuous();
  if (iCompare(moduleType, "ReinforcementLearning/Discrete")) module = new korali::problem::reinforcementLearning::Discrete();
  if (iCompare(moduleType, "Sampling")) module = new korali::problem::Sampling();
  if (iCompare(moduleType, "SupervisedLearning")) module = new korali::problem::SupervisedLearning();
  
  // Solver modules
  if (iCompare(moduleType, "Designer")) module = new korali::solver::Designer();
  if (iCompare(moduleType, "Executor")) module = new korali::solver::Executor();
  if (iCompare(moduleType, "Integrator/MonteCarlo")) module = new korali::solver::integrator::MonteCarlo();
  if (iCompare(moduleType, "Integrator/Quadrature")) module = new korali::solver::integrator::Quadrature();
  if (iCompare(moduleType, "DeepSupervisor")) module = new korali::solver::DeepSupervisor();
  if (iCompare(moduleType, "DeepSupervisor/optimizers/fAdam")) module = new korali::fAdam();
  if (iCompare(moduleType, "DeepSupervisor/optimizers/fAdaBelief")) module = new korali::fAdaBelief();
  if (iCompare(moduleType, "DeepSupervisor/optimizers/fMadGrad")) module = new korali::fMadGrad();
  if (iCompare(moduleType, "DeepSupervisor/optimizers/fAdaGrad")) module = new korali::fAdaGrad();
  if (iCompare(moduleType, "Agent/Continuous/VRACER")) module = new korali::solver::agent::continuous::VRACER();
  if (iCompare(moduleType, "Agent/Discrete/dVRACER")) module = new korali::solver::agent::discrete::dVRACER();
  if (iCompare(moduleType, "Optimizer/CMAES")) module = new korali::solver::optimizer::CMAES();
  if (iCompare(moduleType, "Optimizer/DEA")) module = new korali::solver::optimizer::DEA();
  if (iCompare(moduleType, "Optimizer/Rprop")) module = new korali::solver::optimizer::Rprop();
  if (iCompare(moduleType, "Optimizer/Adam")) module = new korali::solver::optimizer::Adam();
  if (iCompare(moduleType, "Optimizer/AdaBelief")) module = new korali::solver::optimizer::AdaBelief();
  if (iCompare(moduleType, "Optimizer/MADGRAD")) module = new korali::solver::optimizer::MADGRAD();
  if (iCompare(moduleType, "Optimizer/MOCMAES")) module = new korali::solver::optimizer::MOCMAES();
  if (iCompare(moduleType, "Optimizer/GridSearch")) module = new korali::solver::optimizer::GridSearch();
  if (iCompare(moduleType, "Sampler/Nested")) module = new korali::solver::sampler::Nested();
  if (iCompare(moduleType, "Sampler/MCMC")) module = new korali::solver::sampler::MCMC();
  if (iCompare(moduleType, "Sampler/HMC")) module = new korali::solver::sampler::HMC();
  if (iCompare(moduleType, "Sampler/TMCMC")) module = new korali::solver::sampler::TMCMC();
  if (iCompare(moduleType, "SSM/SSA")) module = new korali::solver::ssm::SSA();
  if (iCompare(moduleType, "SSM/TauLeaping")) module = new korali::solver::ssm::TauLeaping();

  // Neural Network modules
  if (iCompare(moduleType, "NeuralNetwork")) module = new korali::NeuralNetwork();
  if (iCompare(moduleType, "Layer/Linear")) module = new korali::neuralNetwork::layer::Linear();
  if (iCompare(moduleType, "Layer/Convolution")) module = new korali::neuralNetwork::layer::Convolution();
  if (iCompare(moduleType, "Layer/Deconvolution")) module = new korali::neuralNetwork::layer::Deconvolution();
  if (iCompare(moduleType, "Layer/Pooling")) module = new korali::neuralNetwork::layer::Pooling();
  if (iCompare(moduleType, "Layer/Recurrent/GRU")) module = new korali::neuralNetwork::layer::recurrent::GRU();
  if (iCompare(moduleType, "Layer/Recurrent/LSTM")) module = new korali::neuralNetwork::layer::recurrent::LSTM();
  if (iCompare(moduleType, "Layer/Input")) module = new korali::neuralNetwork::layer::Input();
  if (iCompare(moduleType, "Layer/Output")) module = new korali::neuralNetwork::layer::Output();
  if (iCompare(moduleType, "Layer/Activation")) module = new korali::neuralNetwork::layer::Activation();

  if (module == nullptr) KORALI_LOG_ERROR(" + Unrecognized module: %s.\n", moduleType.c_str());

  // If this is a new experiment, we should assign it its own configuration
  if (isExperiment == true) dynamic_cast<Experiment *>(module)->_js.getJson() = js;

  // If this is a module inside an experiment, it needs to be properly configured
  if (isExperiment == false) module->_k = e;

  return module;
}

} // namespace korali
