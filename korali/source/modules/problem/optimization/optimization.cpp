#include "modules/problem/optimization/optimization.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Optimization::initialize()
{
  if (_k->_variables.size() == 0) KORALI_LOG_ERROR("Optimization Evaluation problems require at least one variable.\n");
}

void Optimization::evaluateConstraints(Sample &sample)
{
  for (size_t i = 0; i < _constraints.size(); i++)
  {
    sample.run(_constraints[i]);

    auto evaluation = KORALI_GET(double, sample, "F(x)");

    if (std::isfinite(evaluation) == false)
      KORALI_LOG_ERROR("Non finite value of constraint evaluation %lu detected: %f\n", i, evaluation);

    sample["Constraint Evaluations"][i] = evaluation;
  }
}

void Optimization::evaluate(Sample &sample)
{
  sample.run(_objectiveFunction);

  auto evaluation = KORALI_GET(double, sample, "F(x)");

  if (std::isfinite(evaluation) == false)
    KORALI_LOG_ERROR("Non finite value of function evaluation detected: %f\n", evaluation);
}

void Optimization::evaluateMultiple(Sample &sample)
{
  sample.run(_objectiveFunction);

  auto evaluation = KORALI_GET(std::vector<double>, sample, "F(x)");

  for (size_t i = 0; i < evaluation.size(); i++)
    if (std::isfinite(evaluation[i]) == false)
      KORALI_LOG_ERROR("Non finite value of function evaluation detected for variable %lu: %f\n", i, evaluation[i]);
}

void Optimization::evaluateWithGradients(Sample &sample)
{
  sample.run(_objectiveFunction);

  auto evaluation = KORALI_GET(double, sample, "F(x)");
  auto gradient = KORALI_GET(std::vector<double>, sample, "Gradient");

  if (gradient.size() != _k->_variables.size())
    KORALI_LOG_ERROR("Size of sample's gradient evaluations vector (%lu) is different from the number of problem variables defined (%lu).\n", gradient.size(), _k->_variables.size());

  if (std::isfinite(evaluation) == false)
    KORALI_LOG_ERROR("Non finite value of function evaluation detected: %f\n", evaluation);

  for (size_t i = 0; i < gradient.size(); i++)
    if (std::isfinite(gradient[i]) == false)
      KORALI_LOG_ERROR("Non finite value of gradient evaluation detected for variable %lu: %f\n", i, gradient[i]);
}

void Optimization::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Has Discrete Variables"))
 {
 try { _hasDiscreteVariables = js["Has Discrete Variables"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimization ] \n + Key:    ['Has Discrete Variables']\n%s", e.what()); } 
   eraseValue(js, "Has Discrete Variables");
 }

 if (isDefined(js, "Num Objectives"))
 {
 try { _numObjectives = js["Num Objectives"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimization ] \n + Key:    ['Num Objectives']\n%s", e.what()); } 
   eraseValue(js, "Num Objectives");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Num Objectives'] required by optimization.\n"); 

 if (isDefined(js, "Objective Function"))
 {
 try { _objectiveFunction = js["Objective Function"].get<std::uint64_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimization ] \n + Key:    ['Objective Function']\n%s", e.what()); } 
   eraseValue(js, "Objective Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Objective Function'] required by optimization.\n"); 

 if (isDefined(js, "Constraints"))
 {
 try { _constraints = js["Constraints"].get<std::vector<std::uint64_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimization ] \n + Key:    ['Constraints']\n%s", e.what()); } 
   eraseValue(js, "Constraints");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Constraints'] required by optimization.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Optimizer"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: optimization\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "optimization";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: optimization: \n%s\n", js.dump(2).c_str());
} 

void Optimization::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Num Objectives"] = _numObjectives;
   js["Objective Function"] = _objectiveFunction;
   js["Constraints"] = _constraints;
   js["Has Discrete Variables"] = _hasDiscreteVariables;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Problem::getConfiguration(js);
} 

void Optimization::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Num Objectives\": 1, \"Has Discrete Variables\": false, \"Constraints\": []}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Problem::applyModuleDefaults(js);
} 

void Optimization::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

bool Optimization::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Evaluate")
 {
  evaluate(sample);
  return true;
 }

 if (operation == "Evaluate Multiple")
 {
  evaluateMultiple(sample);
  return true;
 }

 if (operation == "Evaluate With Gradients")
 {
  evaluateWithGradients(sample);
  return true;
 }

 if (operation == "Evaluate Constraints")
 {
  evaluateConstraints(sample);
  return true;
 }

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Optimization.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;
