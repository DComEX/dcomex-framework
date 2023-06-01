#include "modules/distribution/univariate/exponential/exponential.hpp"
#include "modules/experiment/experiment.hpp"
#include <gsl/gsl_randist.h>

namespace korali
{
namespace distribution
{
namespace univariate
{
;

double Exponential::getDensity(const double x) const
{
  return gsl_ran_exponential_pdf(x - _location, _mean);
}

double Exponential::getLogDensity(const double x) const
{
  if (x - _location < 0) return -INFINITY;
  return -log(_mean) - (x - _location) / _mean;
}

double Exponential::getLogDensityGradient(const double x) const
{
  if (x - _location < 0) return 0.;
  return -1.0 / _mean;
}

double Exponential::getLogDensityHessian(const double x) const
{
  return 0.;
}

double Exponential::getRandomNumber()
{
  return _location + gsl_ran_exponential(_range, _mean);
}

void Exponential::updateDistribution()
{
  _aux = 0.0;
  if (_mean <= 0.0) KORALI_LOG_ERROR("Incorrect mean parameter of exponential distribution: %f.\n", _mean);
}

void Exponential::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Location"].is_number()) {_location = js["Location"]; _locationConditional = ""; } 
 if(js["Location"].is_string()) { _hasConditionalVariables = true; _locationConditional = js["Location"]; } 
 eraseValue(js, "Location");
 if(js["Mean"].is_number()) {_mean = js["Mean"]; _meanConditional = ""; } 
 if(js["Mean"].is_string()) { _hasConditionalVariables = true; _meanConditional = js["Mean"]; } 
 eraseValue(js, "Mean");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/exponential";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: exponential: \n%s\n", js.dump(2).c_str());
} 

void Exponential::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_locationConditional == "") js["Location"] = _location;
 if(_locationConditional != "") js["Location"] = _locationConditional; 
 if(_meanConditional == "") js["Mean"] = _mean;
 if(_meanConditional != "") js["Mean"] = _meanConditional; 
 Univariate::getConfiguration(js);
} 

void Exponential::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Exponential::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Exponential::getPropertyPointer(const std::string& property)
{
 if (property == "Location") return &_location;
 if (property == "Mean") return &_mean;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Exponential.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
