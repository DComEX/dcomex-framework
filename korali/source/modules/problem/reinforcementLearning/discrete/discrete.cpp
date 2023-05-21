#include "modules/problem/reinforcementLearning/discrete/discrete.hpp"
#include "modules/solver/agent/discrete/discrete.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
namespace reinforcementLearning
{
;

void Discrete::initialize()
{
  ReinforcementLearning::initialize();

  /*********************************************************************
   * Verifying Discrete Action Space
   *********************************************************************/

  if (_possibleActions.empty())
    KORALI_LOG_ERROR("No possible actions have been defined for the discrete RL problem (empty set detected).\n");

  _actionCount = _possibleActions.size();

  for (size_t i = 0; i < _possibleActions.size(); i++)
    if (_possibleActions[i].size() != _actionVectorSize)
      KORALI_LOG_ERROR("For possible action %lu, incorrect vector size provided. Expected: %lu, Provided: %lu.\n", i, _actionVectorSize, _possibleActions[i].size());
}

void Discrete::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Possible Actions"))
 {
 try { _possibleActions = js["Possible Actions"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ discrete ] \n + Key:    ['Possible Actions']\n%s", e.what()); } 
   eraseValue(js, "Possible Actions");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Possible Actions'] required by discrete.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Agent/Discrete"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: discrete\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 ReinforcementLearning::setConfiguration(js);
 _type = "reinforcementLearning/discrete";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: discrete: \n%s\n", js.dump(2).c_str());
} 

void Discrete::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Possible Actions"] = _possibleActions;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 ReinforcementLearning::getConfiguration(js);
} 

void Discrete::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 ReinforcementLearning::applyModuleDefaults(js);
} 

void Discrete::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 ReinforcementLearning::applyVariableDefaults();
} 

bool Discrete::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 operationDetected = operationDetected || ReinforcementLearning::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Discrete.\n", operation.c_str());
 return operationDetected;
}

;

} //reinforcementLearning
} //problem
} //korali
;
