#include "modules/conduit/conduit.hpp"
#include "modules/problem/hierarchical/theta/theta.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

void Theta::initialize()
{
  // Psi Experiment

  // Setting experiment configurations to actual korali experiments
  _psiExperimentObject._js.getJson() = _psiExperiment;

  // Running initialization to verify that the configuration is correct
  _psiExperimentObject.initialize();

  _psiProblem = dynamic_cast<Psi *>(_psiExperimentObject._problem);
  if (_psiProblem == NULL) KORALI_LOG_ERROR("Psi experiment passed is not of type Hierarchical/Psi\n");

  // Now inheriting Sub problem's variables
  _k->_distributions = _subExperimentObject._distributions;
  _k->_variables = _subExperimentObject._variables;
  _psiVariableCount = _psiExperimentObject._variables.size();

  // Loading Psi problem results
  _psiProblemSampleLogLikelihoods = _psiExperiment["Results"]["Posterior Sample LogLikelihood Database"].get<std::vector<double>>();
  _psiProblemSampleLogPriors = _psiExperiment["Results"]["Posterior Sample LogPrior Database"].get<std::vector<double>>();
  _psiProblemSampleCoordinates = _psiExperiment["Results"]["Posterior Sample Database"].get<std::vector<std::vector<double>>>();
  _psiProblemSampleCount = _psiProblemSampleCoordinates.size();

  for (size_t i = 0; i < _psiProblemSampleLogPriors.size(); i++)
  {
    double expPrior = exp(_psiProblemSampleLogPriors[i]);
    if (std::isfinite(expPrior) == false)
      KORALI_LOG_ERROR("Non finite (%lf) prior has been detected at sample %zu in Psi problem.\n", expPrior, i);
  }

  /// Sub Experiment
  if (_subExperiment["Is Finished"] == false)
    KORALI_LOG_ERROR("The Hierarchical Bayesian (Theta) requires that the sub problem has run completely, but this one has not.\n");

  _subExperimentObject._js.getJson() = _subExperiment;
  _subExperimentObject.initialize();
  _subProblemVariableCount = _subExperimentObject._variables.size();

  // Loading Theta problem results
  _subProblemSampleLogLikelihoods = _subExperiment["Results"]["Posterior Sample LogLikelihood Database"].get<std::vector<double>>();
  _subProblemSampleLogPriors = _subExperiment["Results"]["Posterior Sample LogPrior Database"].get<std::vector<double>>();
  _subProblemSampleCoordinates = _subExperiment["Results"]["Posterior Sample Database"].get<std::vector<std::vector<double>>>();
  _subProblemSampleCount = _subProblemSampleCoordinates.size();

  for (size_t i = 0; i < _subProblemSampleLogPriors.size(); i++)
  {
    double expPrior = exp(_subProblemSampleLogPriors[i]);
    if (std::isfinite(expPrior) == false)
      KORALI_LOG_ERROR("Non finite (%lf) prior has been detected at sample %zu in sub problem.\n", expPrior, i);
  }

  std::vector<double> logValues;
  logValues.resize(_subProblemSampleCount);

  _psiProblem = dynamic_cast<Psi *>(_psiProblem);

  for (size_t i = 0; i < _psiProblemSampleCount; i++)
  {
    Sample psiSample;
    psiSample["Parameters"] = _psiProblemSampleCoordinates[i];

    _psiProblem->updateConditionalPriors(psiSample);

    for (size_t j = 0; j < _subProblemSampleCount; j++)
    {
      double logConditionalPrior = 0;
      for (size_t k = 0; k < _subProblemVariableCount; k++)
        logConditionalPrior += _psiExperimentObject._distributions[_psiProblem->_conditionalPriorIndexes[k]]->getLogDensity(_subProblemSampleCoordinates[j][k]);

      logValues[j] = logConditionalPrior - _subProblemSampleLogPriors[j];
    }

    double localSum = -log(_subProblemSampleCount) + logSumExp(logValues);

    _precomputedLogDenominator.push_back(localSum);
  }

  // Now inheriting Sub problem's variables and distributions
  _k->_variables.clear();
  for (size_t i = 0; i < _subExperimentObject._variables.size(); i++)
    _k->_variables.push_back(_subExperimentObject._variables[i]);

  _k->_distributions.clear();
  for (size_t i = 0; i < _subExperimentObject._distributions.size(); i++)
    _k->_distributions.push_back(_subExperimentObject._distributions[i]);

  Hierarchical::initialize();
}

void Theta::evaluateLogLikelihood(Sample &sample)
{
  dynamic_cast<problem::Bayesian *>(_subExperimentObject._problem)->evaluateLoglikelihood(sample);

  double logLikelihood = sample["logLikelihood"].get<double>();

  std::vector<double> psiSample;
  psiSample.resize(_psiVariableCount);

  std::vector<double> logValues;
  logValues.resize(_psiProblemSampleCount);

  for (size_t i = 0; i < _psiProblemSampleCount; i++)
  {
    Sample psiSample;
    psiSample["Parameters"] = _psiProblemSampleCoordinates[i];

    _psiProblem->updateConditionalPriors(psiSample);

    double logConditionalPrior = 0.;
    for (size_t k = 0; k < _subProblemVariableCount; k++)
      logConditionalPrior += _psiExperimentObject._distributions[_psiProblem->_conditionalPriorIndexes[k]]->getLogDensity(sample["Parameters"][k]);

    logValues[i] = logConditionalPrior - _precomputedLogDenominator[i];
  }

  sample["logLikelihood"] = logLikelihood - log(_psiProblemSampleCount) + logSumExp(logValues);
}

void Theta::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Sub Experiment"))
 {
 _subExperiment = js["Sub Experiment"].get<knlohmann::json>();

   eraseValue(js, "Sub Experiment");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Sub Experiment'] required by theta.\n"); 

 if (isDefined(js, "Psi Experiment"))
 {
 _psiExperiment = js["Psi Experiment"].get<knlohmann::json>();

   eraseValue(js, "Psi Experiment");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Psi Experiment'] required by theta.\n"); 

 Hierarchical::setConfiguration(js);
 _type = "hierarchical/theta";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: theta: \n%s\n", js.dump(2).c_str());
} 

void Theta::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Sub Experiment"] = _subExperiment;
   js["Psi Experiment"] = _psiExperiment;
 Hierarchical::getConfiguration(js);
} 

void Theta::applyModuleDefaults(knlohmann::json& js) 
{

 Hierarchical::applyModuleDefaults(js);
} 

void Theta::applyVariableDefaults() 
{

 Hierarchical::applyVariableDefaults();
} 

;

} //hierarchical
} //problem
} //korali
;
