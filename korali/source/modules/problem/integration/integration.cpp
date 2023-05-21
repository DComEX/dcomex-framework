#include "modules/problem/integration/integration.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Integration::initialize()
{
  if (_k->_variables.size() == 0) KORALI_LOG_ERROR("Integration problems require at least one variable.\n");
}

void Integration::execute(Sample &sample)
{
  // Evaluating Sample
  sample.run(_integrand);

  auto evaluation = KORALI_GET(double, sample, "Evaluation");

  if (std::isnan(evaluation)) KORALI_LOG_ERROR("The function evaluation returned NaN.\n");
}

void Integration::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Integrand"))
 {
 try { _integrand = js["Integrand"].get<std::uint64_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Integrand']\n%s", e.what()); } 
   eraseValue(js, "Integrand");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Integrand'] required by integration.\n"); 

  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Integrator"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: integration\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "integration";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: integration: \n%s\n", js.dump(2).c_str());
} 

void Integration::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Integrand"] = _integrand;
 Problem::getConfiguration(js);
} 

void Integration::applyModuleDefaults(knlohmann::json& js) 
{

 Problem::applyModuleDefaults(js);
} 

void Integration::applyVariableDefaults() 
{

 Problem::applyVariableDefaults();
} 

bool Integration::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Execute")
 {
  execute(sample);
  return true;
 }

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Integration.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;
