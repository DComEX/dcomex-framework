#include "modules/distribution/univariate/geometric/geometric.hpp"
#include "modules/experiment/experiment.hpp"
#include <gsl/gsl_randist.h>

namespace korali
{
namespace distribution
{
namespace univariate
{
;

double
Geometric::getDensity(const double x) const
{
  return gsl_ran_geometric_pdf((int)x, _successProbability);
}

double Geometric::getLogDensity(const double x) const
{
  return std::log(_successProbability) + (x - 1) * std::log(1.0 - _successProbability);
}

double Geometric::getLogDensityGradient(const double x) const
{
  KORALI_LOG_ERROR("Gradient of discrete pdf %s not defined.\n", _name.c_str());
  return 0.;
}

double Geometric::getLogDensityHessian(const double x) const
{
  KORALI_LOG_ERROR("Hessian of discrete pdf %s not defined.\n", _name.c_str());
  return 0.;
}

double Geometric::getRandomNumber()
{
  return gsl_ran_geometric(_range, _successProbability);
}

void Geometric::updateDistribution()
{
  if (_successProbability <= 0.0) KORALI_LOG_ERROR("Incorrect success probability parameter of geometric distribution: %f.\n", _successProbability);
  if (_successProbability > 1.0) KORALI_LOG_ERROR("Incorrect success probability parameter of geometric distribution: %f.\n", _successProbability);
  _aux = 0.0;
}

void Geometric::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Success Probability"].is_number()) {_successProbability = js["Success Probability"]; _successProbabilityConditional = ""; } 
 if(js["Success Probability"].is_string()) { _hasConditionalVariables = true; _successProbabilityConditional = js["Success Probability"]; } 
 eraseValue(js, "Success Probability");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/geometric";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: geometric: \n%s\n", js.dump(2).c_str());
} 

void Geometric::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_successProbabilityConditional == "") js["Success Probability"] = _successProbability;
 if(_successProbabilityConditional != "") js["Success Probability"] = _successProbabilityConditional; 
 Univariate::getConfiguration(js);
} 

void Geometric::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Geometric::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Geometric::getPropertyPointer(const std::string& property)
{
 if (property == "Success Probability") return &_successProbability;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Geometric.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
