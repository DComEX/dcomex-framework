#include "modules/conduit/conduit.hpp"
#include "modules/problem/hierarchical/thetaNew/thetaNew.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

void ThetaNew::initialize()
{
  Hierarchical::initialize();

  // Setting experiment configurations to actual korali experiments
  _psiExperimentObject._js.getJson() = _psiExperiment;

  // Running initialization to verify that the configuration is correct
  _psiExperimentObject.initialize();

  // Psi-problem correctness checks
  if (_psiExperiment["Is Finished"] == false)
    KORALI_LOG_ERROR("The Hierarchical Bayesian (Theta New) requires that the psi-problem has run completely, but it has not.\n");

  // Loading Psi problem results
  _psiProblemSampleLogLikelihoods = _psiExperiment["Results"]["Posterior Sample LogLikelihood Database"].get<std::vector<double>>();
  _psiProblemSampleLogPriors = _psiExperiment["Results"]["Posterior Sample LogPrior Database"].get<std::vector<double>>();
  _psiProblemSampleCoordinates = _psiExperiment["Results"]["Posterior Sample Database"].get<std::vector<std::vector<double>>>();
  _psiProblemSampleCount = _psiProblemSampleCoordinates.size();

  for (size_t i = 0; i < _psiProblemSampleLogPriors.size(); i++)
  {
    double expPrior = exp(_psiProblemSampleLogPriors[i]);
    if (std::isfinite(expPrior) == false)
      KORALI_LOG_ERROR("Non finite (%lf) prior has been detected at sample %zu in subproblem.\n", expPrior, i);
  }
}

void ThetaNew::evaluateLogLikelihood(Sample &sample)
{
  auto _psiProblem = dynamic_cast<Psi *>(_psiExperimentObject._problem);

  size_t Ntheta = _k->_variables.size();
  std::vector<double> logValues(_psiProblemSampleCount, 0.0);

  for (size_t i = 0; i < _psiProblemSampleCount; i++)
  {
    Sample psiSample;
    psiSample["Parameters"] = _psiProblemSampleCoordinates[i];
    _psiProblem->updateConditionalPriors(psiSample);

    logValues[i] = 0.;
    for (size_t k = 0; k < Ntheta; k++)
      logValues[i] += _psiExperimentObject._distributions[_psiProblem->_conditionalPriorIndexes[k]]->getLogDensity(sample["Parameters"][k]);
  }

  sample["logLikelihood"] = -log(_psiProblemSampleCount) + logSumExp(logValues);
}

void ThetaNew::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Psi Experiment"))
 {
 _psiExperiment = js["Psi Experiment"].get<knlohmann::json>();

   eraseValue(js, "Psi Experiment");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Psi Experiment'] required by thetaNew.\n"); 

 Hierarchical::setConfiguration(js);
 _type = "hierarchical/thetaNew";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: thetaNew: \n%s\n", js.dump(2).c_str());
} 

void ThetaNew::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Psi Experiment"] = _psiExperiment;
 Hierarchical::getConfiguration(js);
} 

void ThetaNew::applyModuleDefaults(knlohmann::json& js) 
{

 Hierarchical::applyModuleDefaults(js);
} 

void ThetaNew::applyVariableDefaults() 
{

 Hierarchical::applyVariableDefaults();
} 

;

} //hierarchical
} //problem
} //korali
;
