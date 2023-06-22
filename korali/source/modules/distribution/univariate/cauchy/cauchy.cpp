#include "modules/distribution/univariate/cauchy/cauchy.hpp"
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
Cauchy::getDensity(const double x) const
{
  return gsl_ran_cauchy_pdf(x - _location, _scale);
}

double Cauchy::getLogDensity(const double x) const
{
  return _aux - std::log(1. + (x - _location) * (x - _location) / (_scale * _scale));
}

double Cauchy::getLogDensityGradient(const double x) const
{
  double tmp = (x - _location) / _scale;
  return -2. * tmp / (1. + tmp * tmp) / _scale;
}

double Cauchy::getLogDensityHessian(const double x) const
{
  double tmp = (x - _location) / _scale;
  double tmp2 = tmp * tmp;
  return -2. * ((1. + tmp2) - 2. * tmp2) / ((1. + tmp2) * (1. + tmp2) * _scale * _scale);
}

double Cauchy::getRandomNumber()
{
  return _location + gsl_ran_cauchy(_range, _scale);
}

void Cauchy::updateDistribution()
{
  if (_scale <= 0) KORALI_LOG_ERROR("Incorrect Scale parameter of Cauchy distribution: %f.\n", _scale);

  _aux = -gsl_sf_log(_scale * M_PI);
}

void Cauchy::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Location"].is_number()) {_location = js["Location"]; _locationConditional = ""; } 
 if(js["Location"].is_string()) { _hasConditionalVariables = true; _locationConditional = js["Location"]; } 
 eraseValue(js, "Location");
 if(js["Scale"].is_number()) {_scale = js["Scale"]; _scaleConditional = ""; } 
 if(js["Scale"].is_string()) { _hasConditionalVariables = true; _scaleConditional = js["Scale"]; } 
 eraseValue(js, "Scale");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/cauchy";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: cauchy: \n%s\n", js.dump(2).c_str());
} 

void Cauchy::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_locationConditional == "") js["Location"] = _location;
 if(_locationConditional != "") js["Location"] = _locationConditional; 
 if(_scaleConditional == "") js["Scale"] = _scale;
 if(_scaleConditional != "") js["Scale"] = _scaleConditional; 
 Univariate::getConfiguration(js);
} 

void Cauchy::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Cauchy::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Cauchy::getPropertyPointer(const std::string& property)
{
 if (property == "Location") return &_location;
 if (property == "Scale") return &_scale;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Cauchy.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
