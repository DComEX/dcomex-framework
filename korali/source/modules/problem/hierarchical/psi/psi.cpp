#include "modules/conduit/conduit.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/hierarchical/psi/psi.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

void Psi::initialize()
{
  Hierarchical::initialize();

  _conditionalPriorIndexes.resize(_conditionalPriors.size());

  if (_conditionalPriors.size() == 0) KORALI_LOG_ERROR("Hierarchical Bayesian (Psi) problems require at least one conditional prior\n");

  for (size_t i = 0; i < _conditionalPriors.size(); i++)
  {
    bool foundDistribution = false;

    for (size_t j = 0; j < _k->_distributions.size(); j++)
      if (_conditionalPriors[i] == _k->_distributions[j]->_name)
      {
        foundDistribution = true;
        _conditionalPriorIndexes[i] = j;
      }

    if (foundDistribution == false)
      KORALI_LOG_ERROR("Did not find conditional prior distribution %s\n", _conditionalPriors[i].c_str());
  }

  if (_subExperiments.size() < 2) KORALI_LOG_ERROR("Hierarchical Bayesian (Psi) problem requires defining at least two executed sub-problems.\n");

  // Obtaining sub problem count and variable counts
  _subProblemsCount = _subExperiments.size();
  _subProblemsVariablesCount = _conditionalPriors.size();

  // Sub-problem correctness checks
  for (size_t i = 0; i < _subProblemsCount; i++)
  {
    if (_conditionalPriors.size() != _subExperiments[i]["Variables"].size())
      KORALI_LOG_ERROR("Sub-problem %lu contains a different number of variables (%lu) than conditional priors in the Hierarchical Bayesian (Psi) problem (%lu).\n", i, _subExperiments[i]["Problem"]["Variables"].size(), _conditionalPriors.size());

    if (_subExperiments[i]["Is Finished"] == false)
      KORALI_LOG_ERROR("The Hierarchical Bayesian (Psi) requires that all problems have run completely, but Problem %lu has not.\n", i);
  }

  _subProblemsSampleCoordinates.resize(_subProblemsCount);
  _subProblemsSampleLogLikelihoods.resize(_subProblemsCount);
  _subProblemsSampleLogPriors.resize(_subProblemsCount);

  for (size_t i = 0; i < _subProblemsCount; i++)
  {
    try
    {
      _subProblemsSampleLogPriors[i] = _subExperiments[i]["Results"]["Posterior Sample LogPrior Database"].get<std::vector<double>>();
      _subProblemsSampleLogLikelihoods[i] = _subExperiments[i]["Results"]["Posterior Sample LogLikelihood Database"].get<std::vector<double>>();
      _subProblemsSampleCoordinates[i] = _subExperiments[i]["Results"]["Posterior Sample Database"].get<std::vector<std::vector<double>>>();
    }
    catch (std::exception &e)
    {
      KORALI_LOG_ERROR("Error reading the sample database from sub-problem: %lu. Was it a sampling experiment?\n", i);
    }

    for (size_t j = 0; j < _subProblemsSampleLogPriors.size(); j++)
    {
      double expPrior = exp(_subProblemsSampleLogPriors[i][j]);
      if (std::isfinite(expPrior) == false)
        KORALI_LOG_ERROR("Non finite (%lf) prior has been detected at sample %zu in subproblem %zu.\n", expPrior, j, i);
    }
  }

  // Configuring conditional priors given hyperparameters
  _conditionalPriorInfos.resize(_conditionalPriors.size());

  for (size_t i = 0; i < _conditionalPriors.size(); i++)
  {
    auto distributionJs = knlohmann::json();
    _k->_distributions[_conditionalPriorIndexes[i]]->getConfiguration(distributionJs);

    for (auto it = distributionJs.begin(); it != distributionJs.end(); ++it)
      if (it.value().is_string())
      {
        std::string key(it.key());
        std::string value(it.value().get<std::string>());
        size_t position = 0;
        double *pointer;

        if (key == "Name") continue;
        if (key == "Type") continue;
        if (key == "Range") continue;
        if (key == "Random Seed") continue;

        bool foundValue = false;
        for (size_t k = 0; k < _k->_variables.size(); k++)
          if (_k->_variables[k]->_name == value)
          {
            position = k;
            pointer = _k->_distributions[_conditionalPriorIndexes[i]]->getPropertyPointer(key);
            foundValue = true;
          }
        if (foundValue == false) KORALI_LOG_ERROR("No variable name specified that satisfies conditional prior property \"%s\" with key: \"%s\".\n", key.c_str(), value.c_str());

        _conditionalPriorInfos[i]._samplePointers.push_back(pointer);
        _conditionalPriorInfos[i]._samplePositions.push_back(position);
      }
  }
}

void Psi::updateConditionalPriors(Sample &sample)
{
  for (size_t i = 0; i < _conditionalPriors.size(); i++)
  {
    for (size_t j = 0; j < _conditionalPriorInfos[i]._samplePositions.size(); j++)
      *(_conditionalPriorInfos[i]._samplePointers[j]) = sample["Parameters"][_conditionalPriorInfos[i]._samplePositions[j]];
    _k->_distributions[_conditionalPriorIndexes[i]]->updateDistribution();
  }
}

void Psi::evaluateLogLikelihood(Sample &sample)
{
  try
  {
    updateConditionalPriors(sample);

    double logLikelihood = 0.0;

    for (size_t i = 0; i < _subProblemsCount; i++)
    {
      std::vector<double> logValues(_subProblemsSampleLogPriors[i].size());

      for (size_t j = 0; j < _subProblemsSampleLogPriors[i].size(); j++)
      {
        logValues[j] = -_subProblemsSampleLogPriors[i][j];
        for (size_t k = 0; k < _conditionalPriors.size(); k++)
          logValues[j] += _k->_distributions[_conditionalPriorIndexes[k]]->getLogDensity(_subProblemsSampleCoordinates[i][j][k]);
      }

      logLikelihood += logSumExp(logValues);
    }

    sample["logLikelihood"] = logLikelihood;
  }
  catch (std::exception &e)
  {
    sample["logLikelihood"] = -Inf;
  }
}

void Psi::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Sub Experiments"))
 {
 _subExperiments = js["Sub Experiments"].get<std::vector<knlohmann::json>>();

   eraseValue(js, "Sub Experiments");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Sub Experiments'] required by psi.\n"); 

 if (isDefined(js, "Conditional Priors"))
 {
 try { _conditionalPriors = js["Conditional Priors"].get<std::vector<std::string>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ psi ] \n + Key:    ['Conditional Priors']\n%s", e.what()); } 
   eraseValue(js, "Conditional Priors");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Conditional Priors'] required by psi.\n"); 

 Hierarchical::setConfiguration(js);
 _type = "hierarchical/psi";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: psi: \n%s\n", js.dump(2).c_str());
} 

void Psi::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Sub Experiments"] = _subExperiments;
   js["Conditional Priors"] = _conditionalPriors;
 Hierarchical::getConfiguration(js);
} 

void Psi::applyModuleDefaults(knlohmann::json& js) 
{

 Hierarchical::applyModuleDefaults(js);
} 

void Psi::applyVariableDefaults() 
{

 Hierarchical::applyVariableDefaults();
} 

;

} //hierarchical
} //problem
} //korali
;
