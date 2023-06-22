#include "modules/problem/sampling/sampling.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Sampling::initialize()
{
  if (_k->_variables.size() == 0) KORALI_LOG_ERROR("Sampling Evaluation problems require at least one variable.\n");
}

void Sampling::evaluate(Sample &sample)
{
  sample.run(_probabilityFunction);

  const auto evaluation = KORALI_GET(double, sample, "logP(x)");

  sample["logP(x)"] = evaluation;
  sample["F(x)"] = evaluation;
}

void Sampling::evaluateGradient(korali::Sample &sample)
{
  const size_t Nth = _k->_variables.size();
  const auto gradient = KORALI_GET(std::vector<double>, sample, "grad(logP(x))");
  if (gradient.size() != Nth)
    KORALI_LOG_ERROR("Dimension of Gradient must be %zu (is %zu).\n", Nth, gradient.size());
}

void Sampling::evaluateHessian(korali::Sample &sample)
{
  const size_t Nth = _k->_variables.size();
  const auto hessian = KORALI_GET(std::vector<std::vector<double>>, sample, "H(logP(x))");

  if (hessian.size() != Nth)
    KORALI_LOG_ERROR("Outer dimension of Hessian matrix must be %zu (is %zu).\n", Nth, hessian.size());

  std::vector<double> flat_hessian(0);
  auto it = flat_hessian.begin();
  for (size_t i = 0; i < Nth; ++i)
  {
    if (hessian[i].size() != Nth)
      KORALI_LOG_ERROR("Inner dimension of Hessian matrix must be %zu (is %zu).\n", Nth, hessian[i].size());
    flat_hessian.insert(it, std::cbegin(hessian[i]), std::cend(hessian[i]));
    it = flat_hessian.end();
  }
  sample["H(logP(x))"] = flat_hessian;
}

void Sampling::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Probability Function"))
 {
 try { _probabilityFunction = js["Probability Function"].get<std::uint64_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ sampling ] \n + Key:    ['Probability Function']\n%s", e.what()); } 
   eraseValue(js, "Probability Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Probability Function'] required by sampling.\n"); 

  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Sampler/MCMC"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
   candidateSolverName = toLower("Sampler/HMC"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: sampling\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "sampling";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: sampling: \n%s\n", js.dump(2).c_str());
} 

void Sampling::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Probability Function"] = _probabilityFunction;
 Problem::getConfiguration(js);
} 

void Sampling::applyModuleDefaults(knlohmann::json& js) 
{

 Problem::applyModuleDefaults(js);
} 

void Sampling::applyVariableDefaults() 
{

 Problem::applyVariableDefaults();
} 

bool Sampling::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Evaluate")
 {
  evaluate(sample);
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

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Sampling.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;
