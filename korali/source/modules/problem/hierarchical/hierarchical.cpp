#include "modules/problem/hierarchical/hierarchical.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Hierarchical::initialize()
{
  for (size_t i = 0; i < _k->_variables.size(); i++)
  {
    bool foundDistribution = false;

    for (size_t j = 0; j < _k->_distributions.size(); j++)
      if (_k->_variables[i]->_priorDistribution == _k->_distributions[j]->_name)
      {
        foundDistribution = true;
        _k->_variables[i]->_distributionIndex = j;
      }

    if (foundDistribution == false)
      KORALI_LOG_ERROR("Did not find distribution %s, specified by variable %s\n", _k->_variables[i]->_priorDistribution.c_str(), _k->_variables[i]->_name.c_str());
  }
}

void Hierarchical::evaluateLogPrior(Sample &sample)
{
  double logPrior = 0.0;

  for (size_t i = 0; i < sample["Parameters"].size(); i++)
    logPrior += _k->_distributions[_k->_variables[i]->_distributionIndex]->getLogDensity(sample["Parameters"][i]);

  sample["logPrior"] = logPrior;
}

void Hierarchical::evaluateLogPosterior(Sample &sample)
{
  int sampleId = sample["Sample Id"];
  evaluateLogPrior(sample);

  if (sample["logPrior"] == -Inf)
  {
    sample["logLikelihood"] = -Inf;
    sample["logPosterior"] = -Inf;
  }
  else
  {
    evaluateLogLikelihood(sample);
    double logPrior = sample["logPrior"];
    double logLikelihood = sample["logLikelihood"];
    double logPosterior = logPrior + logLikelihood;

    if (std::isnan(logPosterior) == true) KORALI_LOG_ERROR("Sample %d returned NaN logPosterior evaluation.\n", sampleId);

    sample["logPosterior"] = logPrior + logLikelihood;
  }
}

bool Hierarchical::isSampleFeasible(Sample &sample)
{
  for (size_t i = 0; i < sample["Parameters"].size(); i++)
    if (isfinite(_k->_distributions[_k->_variables[i]->_distributionIndex]->getLogDensity(sample["Parameters"][i])) == false)
    {
      sample["Is Feasible"] = false;
      return false;
    }
  sample["Is Feasible"] = true;
  return true;
}

void Hierarchical::evaluate(Sample &sample)
{
  evaluateLogPosterior(sample);
  sample["P(x)"] = sample["logPosterior"];
  sample["F(x)"] = sample["logPosterior"];
}

void Hierarchical::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Prior Distribution"))
 {
 try { _k->_variables[i]->_priorDistribution = _k->_js["Variables"][i]["Prior Distribution"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ hierarchical ] \n + Key:    ['Prior Distribution']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Prior Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Prior Distribution'] required by hierarchical.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Distribution Index"))
 {
 try { _k->_variables[i]->_distributionIndex = _k->_js["Variables"][i]["Distribution Index"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ hierarchical ] \n + Key:    ['Distribution Index']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Distribution Index");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Distribution Index'] required by hierarchical.\n"); 

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
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: hierarchical\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "hierarchical";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: hierarchical: \n%s\n", js.dump(2).c_str());
} 

void Hierarchical::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Prior Distribution"] = _k->_variables[i]->_priorDistribution;
   _k->_js["Variables"][i]["Distribution Index"] = _k->_variables[i]->_distributionIndex;
 } 
 Problem::getConfiguration(js);
} 

void Hierarchical::applyModuleDefaults(knlohmann::json& js) 
{

 Problem::applyModuleDefaults(js);
} 

void Hierarchical::applyVariableDefaults() 
{

 std::string defaultString = "{\"Distribution Index\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

bool Hierarchical::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Evaluate")
 {
  evaluate(sample);
  return true;
 }

 if (operation == "Check Feasibility")
 {
  isSampleFeasible(sample);
  return true;
 }

 if (operation == "Evaluate logPrior")
 {
  evaluateLogPrior(sample);
  return true;
 }

 if (operation == "Evaluate logLikelihood")
 {
  evaluateLogLikelihood(sample);
  return true;
 }

 if (operation == "Evaluate logPosterior")
 {
  evaluateLogPosterior(sample);
  return true;
 }

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Hierarchical.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;