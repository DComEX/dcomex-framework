#include "modules/problem/reinforcementLearning/continuous/continuous.hpp"
#include "modules/solver/agent/continuous/continuous.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
namespace reinforcementLearning
{
;

void Continuous::initialize()
{
  ReinforcementLearning::initialize();

  /*********************************************************************
   * Verifying Continuous Variable Information
   *********************************************************************/

  for (size_t i = 0; i < _actionVectorIndexes.size(); i++)
  {
    size_t varIdx = _actionVectorIndexes[i];
    if (_k->_variables[varIdx]->_upperBound < _k->_variables[varIdx]->_lowerBound) KORALI_LOG_ERROR("Upper bound for variable %lu (%s) is lower than the lower bound (%f < %f).\n", varIdx, _k->_variables[varIdx]->_name.c_str(), _k->_variables[varIdx]->_upperBound, _k->_variables[varIdx]->_lowerBound);
  }
}

void Continuous::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Agent/Continuous"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: continuous\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 ReinforcementLearning::setConfiguration(js);
 _type = "reinforcementLearning/continuous";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: continuous: \n%s\n", js.dump(2).c_str());
} 

void Continuous::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 ReinforcementLearning::getConfiguration(js);
} 

void Continuous::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 ReinforcementLearning::applyModuleDefaults(js);
} 

void Continuous::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 ReinforcementLearning::applyVariableDefaults();
} 

bool Continuous::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 operationDetected = operationDetected || ReinforcementLearning::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Continuous.\n", operation.c_str());
 return operationDetected;
}

;

} //reinforcementLearning
} //problem
} //korali
;
