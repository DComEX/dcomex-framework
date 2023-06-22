#include "modules/distribution/univariate/poisson/poisson.hpp"
#include "modules/experiment/experiment.hpp"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>

namespace korali
{
namespace distribution
{
namespace univariate
{
;

double Poisson::getDensity(const double x) const
{
  if (x < 0.) return 0;
  return gsl_ran_poisson_pdf(x, _mean);
}

double Poisson::getLogDensity(const double x) const
{
  if (x < 0.) return -INFINITY;
  return x * log(_mean) - log(gsl_sf_fact(x)) - _mean;
}

double Poisson::getLogDensityGradient(const double x) const
{
  KORALI_LOG_ERROR("Gradient of discrete pdf %s not defined.\n", _name.c_str());
  return 0.;
}

double Poisson::getLogDensityHessian(const double x) const
{
  KORALI_LOG_ERROR("Hessian of discrete pdf %s not defined.\n", _name.c_str());
  return 0.;
}

double Poisson::getRandomNumber()
{
  return gsl_ran_poisson(_range, _mean);
}

void Poisson::updateDistribution()
{
  if (_mean <= 0.0) KORALI_LOG_ERROR("Incorrect mean parameter of poisson distribution: %f.\n", _mean);
}

void Poisson::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Mean"].is_number()) {_mean = js["Mean"]; _meanConditional = ""; } 
 if(js["Mean"].is_string()) { _hasConditionalVariables = true; _meanConditional = js["Mean"]; } 
 eraseValue(js, "Mean");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/poisson";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: poisson: \n%s\n", js.dump(2).c_str());
} 

void Poisson::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_meanConditional == "") js["Mean"] = _mean;
 if(_meanConditional != "") js["Mean"] = _meanConditional; 
 Univariate::getConfiguration(js);
} 

void Poisson::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Poisson::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Poisson::getPropertyPointer(const std::string& property)
{
 if (property == "Mean") return &_mean;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Poisson.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
