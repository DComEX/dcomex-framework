#include "modules/problem/propagation/propagation.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Propagation::initialize()
{
  if (_k->_variables.size() == 0) KORALI_LOG_ERROR("Execution problems require at least one variable.\n");

  // Validate the given _priorDistribution and _precomputedValues
  size_t Ns = _k->_variables[0]->_precomputedValues.size();
  for (size_t i = 0; i < _k->_variables.size(); i++)
    if (_k->_variables[i]->_precomputedValues.size() != Ns) KORALI_LOG_ERROR("All 'Precomputed Values' must have the same length ");

  for (size_t i = 0; i < _k->_variables.size(); i++)
  {
    bool foundDistribution = false;
    if (Ns == 0)
    {
      // Validate the _priorDistribution names
      for (size_t j = 0; j < _k->_distributions.size(); j++)
        if (_k->_variables[i]->_priorDistribution == _k->_distributions[j]->_name)
        {
          foundDistribution = true;
          _k->_variables[i]->_distributionIndex = j;
        }

      if (foundDistribution == false) KORALI_LOG_ERROR("Did not find distribution %s, specified by variable %s\n", _k->_variables[i]->_priorDistribution.c_str(), _k->_variables[i]->_name.c_str());
      if (_numberOfSamples == 0) KORALI_LOG_ERROR("Number of Samples must be larger than 0");
      _k->_variables[i]->_sampledValues.resize(_numberOfSamples);
    }
    else
    {
      for (size_t j = 0; j < _k->_distributions.size(); j++)
        if (_k->_variables[i]->_priorDistribution == _k->_distributions[j]->_name)
        {
          foundDistribution = true;
          _k->_variables[i]->_distributionIndex = j;
        }

      if (foundDistribution == true) KORALI_LOG_ERROR("Found distribution %s in variable %s\n, although using precomputed Values", _k->_variables[i]->_priorDistribution.c_str(), _k->_variables[i]->_name.c_str());
      if (_numberOfSamples > 0) KORALI_LOG_ERROR("Number of Samples set although using precomputed Values");
    }
  }
}

void Propagation::execute(Sample &sample)
{
  sample.run(_executionModel);
}

void Propagation::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Execution Model"))
 {
 try { _executionModel = js["Execution Model"].get<std::uint64_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ propagation ] \n + Key:    ['Execution Model']\n%s", e.what()); } 
   eraseValue(js, "Execution Model");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Execution Model'] required by propagation.\n"); 

 if (isDefined(js, "Number Of Samples"))
 {
 try { _numberOfSamples = js["Number Of Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ propagation ] \n + Key:    ['Number Of Samples']\n%s", e.what()); } 
   eraseValue(js, "Number Of Samples");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Number Of Samples'] required by propagation.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Precomputed Values"))
 {
 try { _k->_variables[i]->_precomputedValues = _k->_js["Variables"][i]["Precomputed Values"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ propagation ] \n + Key:    ['Precomputed Values']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Precomputed Values");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Precomputed Values'] required by propagation.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Prior Distribution"))
 {
 try { _k->_variables[i]->_priorDistribution = _k->_js["Variables"][i]["Prior Distribution"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ propagation ] \n + Key:    ['Prior Distribution']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Prior Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Prior Distribution'] required by propagation.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Distribution Index"))
 {
 try { _k->_variables[i]->_distributionIndex = _k->_js["Variables"][i]["Distribution Index"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ propagation ] \n + Key:    ['Distribution Index']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Distribution Index");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Distribution Index'] required by propagation.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Sampled Values"))
 {
 try { _k->_variables[i]->_sampledValues = _k->_js["Variables"][i]["Sampled Values"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ propagation ] \n + Key:    ['Sampled Values']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Sampled Values");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Sampled Values'] required by propagation.\n"); 

 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Executor"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: propagation\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "propagation";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: propagation: \n%s\n", js.dump(2).c_str());
} 

void Propagation::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Execution Model"] = _executionModel;
   js["Number Of Samples"] = _numberOfSamples;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Precomputed Values"] = _k->_variables[i]->_precomputedValues;
   _k->_js["Variables"][i]["Prior Distribution"] = _k->_variables[i]->_priorDistribution;
   _k->_js["Variables"][i]["Distribution Index"] = _k->_variables[i]->_distributionIndex;
   _k->_js["Variables"][i]["Sampled Values"] = _k->_variables[i]->_sampledValues;
 } 
 Problem::getConfiguration(js);
} 

void Propagation::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Number Of Samples\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Problem::applyModuleDefaults(js);
} 

void Propagation::applyVariableDefaults() 
{

 std::string defaultString = "{\"Precomputed Values\": [], \"Prior Distribution\": \" \", \"Sampled Values\": [], \"Distribution Index\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

bool Propagation::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Execute")
 {
  execute(sample);
  return true;
 }

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Propagation.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;