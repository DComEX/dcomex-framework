#include "modules/distribution/univariate/uniform/uniform.hpp"
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

double
Uniform::getDensity(const double x) const
{
  return gsl_ran_flat_pdf(x, _minimum, _maximum);
}

double Uniform::getLogDensity(const double x) const
{
  if (x >= _minimum && x <= _maximum) return _aux;
  return -Inf;
}

double Uniform::getLogDensityGradient(const double x) const
{
  return 0.0;
}

double Uniform::getLogDensityHessian(const double x) const
{
  return 0.0;
}

double Uniform::getRandomNumber()
{
  return gsl_ran_flat(_range, _minimum, _maximum);
}

void Uniform::updateDistribution()
{
  if (_maximum - _minimum <= 0.0)
    KORALI_LOG_ERROR("Maximum (%f) bound must be higher than Minimum (%f) bound in a Uniform distribution.\n", _maximum, _minimum);
  else
    _aux = -gsl_sf_log(_maximum - _minimum);
}

void Uniform::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Minimum"].is_number()) {_minimum = js["Minimum"]; _minimumConditional = ""; } 
 if(js["Minimum"].is_string()) { _hasConditionalVariables = true; _minimumConditional = js["Minimum"]; } 
 eraseValue(js, "Minimum");
 if(js["Maximum"].is_number()) {_maximum = js["Maximum"]; _maximumConditional = ""; } 
 if(js["Maximum"].is_string()) { _hasConditionalVariables = true; _maximumConditional = js["Maximum"]; } 
 eraseValue(js, "Maximum");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/uniform";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: uniform: \n%s\n", js.dump(2).c_str());
} 

void Uniform::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_minimumConditional == "") js["Minimum"] = _minimum;
 if(_minimumConditional != "") js["Minimum"] = _minimumConditional; 
 if(_maximumConditional == "") js["Maximum"] = _maximum;
 if(_maximumConditional != "") js["Maximum"] = _maximumConditional; 
 Univariate::getConfiguration(js);
} 

void Uniform::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Uniform::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Uniform::getPropertyPointer(const std::string& property)
{
 if (property == "Minimum") return &_minimum;
 if (property == "Maximum") return &_maximum;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Uniform.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
