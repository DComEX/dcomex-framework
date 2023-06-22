#include "modules/problem/bayesian/bayesian.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Bayesian::initialize()
{
  for (size_t i = 0; i < _k->_variables.size(); i++)
  {
    bool foundDistribution = false;

    for (size_t j = 0; j < _k->_distributions.size(); j++)
    {
      _k->_distributions[j]->updateDistribution();
      if (_k->_variables[i]->_priorDistribution == _k->_distributions[j]->_name)
      {
        foundDistribution = true;
        _k->_variables[i]->_distributionIndex = j;
      }
    }

    if (foundDistribution == false)
      KORALI_LOG_ERROR("Did not find distribution %s, specified by variable %s\n", _k->_variables[i]->_priorDistribution.c_str(), _k->_variables[i]->_name.c_str());
  }
}

void Bayesian::evaluateLogPrior(Sample &sample)
{
  double logPrior = 0.0;
  const auto params = KORALI_GET(std::vector<double>, sample, "Parameters");

  for (size_t i = 0; i < params.size(); i++)
    logPrior += _k->_distributions[_k->_variables[i]->_distributionIndex]->getLogDensity(params[i]);

  sample["logPrior"] = logPrior;
}

void Bayesian::evaluateLogPriorGradient(Sample &sample)
{
  const auto params = KORALI_GET(std::vector<double>, sample, "Parameters");
  std::vector<double> logPriorGradient(params.size(), 0.);

  for (size_t i = 0; i < params.size(); i++)
    logPriorGradient[i] = _k->_distributions[_k->_variables[i]->_distributionIndex]->getLogDensityGradient(params[i]);

  sample["logPrior Gradient"] = logPriorGradient;
}

void Bayesian::evaluateLogPriorHessian(Sample &sample)
{
  const auto params = KORALI_GET(std::vector<double>, sample, "Parameters");
  const size_t numParam = params.size();
  std::vector<double> logPriorHessian(numParam * numParam, 0.);

  for (size_t i = 0; i < numParam; i++)
    logPriorHessian[i * numParam + i] = _k->_distributions[_k->_variables[i]->_distributionIndex]->getLogDensityHessian(params[i]);

  sample["logPrior Hessian"] = logPriorHessian;
}

void Bayesian::evaluateLogPosterior(Sample &sample)
{
  const int sampleId = sample["Sample Id"];
  evaluateLogPrior(sample);

  const double logPrior = KORALI_GET(double, sample, "logPrior");

  if (logPrior == -Inf)
  {
    sample["logLikelihood"] = -Inf;
    sample["logPosterior"] = -Inf;
  }
  else
  {
    evaluateLoglikelihood(sample);
    const double logLikelihood = KORALI_GET(double, sample, "logLikelihood");
    const double logPosterior = logPrior + logLikelihood;

    if (std::isnan(logLikelihood) == true) KORALI_LOG_ERROR("Sample %d returned NaN logLikelihood evaluation.\n", sampleId);

    sample["logPosterior"] = logPosterior;
  }
}

void Bayesian::evaluate(Sample &sample)
{
  evaluateLogPosterior(sample);
  sample["F(x)"] = sample["logPosterior"];
  sample["logP(x)"] = sample["logPosterior"];
}

void Bayesian::evaluateGradient(Sample &sample)
{
  evaluateLogPriorGradient(sample);
  evaluateLoglikelihoodGradient(sample);
  const auto logPriorGrad = KORALI_GET(std::vector<double>, sample, "logPrior Gradient");
  auto logLikGrad = KORALI_GET(std::vector<double>, sample, "logLikelihood Gradient");

  for (size_t i = 0; i < logPriorGrad.size(); ++i)
    logLikGrad[i] += logPriorGrad[i];
  sample["grad(logP(x))"] = logLikGrad;
}

void Bayesian::evaluateHessian(Sample &sample)
{
  evaluateLogPriorHessian(sample);
  evaluateLogLikelihoodHessian(sample);
  const auto logPriorHessian = KORALI_GET(std::vector<double>, sample, "logPrior Hessian");
  auto logLikHessian = KORALI_GET(std::vector<double>, sample, "logLikelihood Hessian");

  for (size_t i = 0; i < logPriorHessian.size(); i++)
    logLikHessian[i] += logPriorHessian[i];

  sample["H(logP(x))"] = logLikHessian;
}

void Bayesian::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Prior Distribution"))
 {
 try { _k->_variables[i]->_priorDistribution = _k->_js["Variables"][i]["Prior Distribution"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ bayesian ] \n + Key:    ['Prior Distribution']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Prior Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Prior Distribution'] required by bayesian.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Distribution Index"))
 {
 try { _k->_variables[i]->_distributionIndex = _k->_js["Variables"][i]["Distribution Index"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ bayesian ] \n + Key:    ['Distribution Index']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Distribution Index");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Distribution Index'] required by bayesian.\n"); 

 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Sampler"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
   candidateSolverName = toLower("Optimizer"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: bayesian\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "bayesian";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: bayesian: \n%s\n", js.dump(2).c_str());
} 

void Bayesian::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Prior Distribution"] = _k->_variables[i]->_priorDistribution;
   _k->_js["Variables"][i]["Distribution Index"] = _k->_variables[i]->_distributionIndex;
 } 
 Problem::getConfiguration(js);
} 

void Bayesian::applyModuleDefaults(knlohmann::json& js) 
{

 Problem::applyModuleDefaults(js);
} 

void Bayesian::applyVariableDefaults() 
{

 std::string defaultString = "{\"Distribution Index\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

bool Bayesian::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Evaluate")
 {
  evaluate(sample);
  return true;
 }

 if (operation == "Evaluate logPrior")
 {
  evaluateLogPrior(sample);
  return true;
 }

 if (operation == "Evaluate logLikelihood")
 {
  evaluateLoglikelihood(sample);
  return true;
 }

 if (operation == "Evaluate logPosterior")
 {
  evaluateLogPosterior(sample);
  return true;
 }

 if (operation == "Evaluate Gradient")
 {
  evaluateGradient(sample);
  return true;
 }

 if (operation == "Evaluate Hessian")
 {
  evaluateHessian(sample);
  return true;
 }

 if (operation == "Evaluate Fisher Information")
 {
  evaluateFisherInformation(sample);
  return true;
 }

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Bayesian.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;
