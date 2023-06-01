#include "modules/distribution/univariate/gamma/gamma.hpp"
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
Gamma::getDensity(const double x) const
{
  return gsl_ran_gamma_pdf(x, _shape, _scale);
}

double Gamma::getLogDensity(const double x) const
{
  if (x < 0.) return -INFINITY;
  return _aux + (_shape - 1.) * log(x) - x / _scale;
}

double Gamma::getLogDensityGradient(const double x) const
{
  if (x < 0.) return 0.;
  return (1. - _shape) / x - 1. / _scale;
}

double Gamma::getLogDensityHessian(const double x) const
{
  if (x < 0.) return 0.;
  return (_shape - 1.) / (x * x);
}

double Gamma::getRandomNumber()
{
  return gsl_ran_gamma(_range, _shape, _scale);
}

void Gamma::updateDistribution()
{
  if (_shape <= 0.) KORALI_LOG_ERROR("Incorrect Shape parameter of Gamma distribution: %f.\n", _shape);
  if (_scale <= 0.) KORALI_LOG_ERROR("Incorrect Scale parameter of Gamma distribution: %f.\n", _scale);

  _aux = -gsl_sf_lngamma(_shape) - _shape * log(_scale);
}

void Gamma::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Shape"].is_number()) {_shape = js["Shape"]; _shapeConditional = ""; } 
 if(js["Shape"].is_string()) { _hasConditionalVariables = true; _shapeConditional = js["Shape"]; } 
 eraseValue(js, "Shape");
 if(js["Scale"].is_number()) {_scale = js["Scale"]; _scaleConditional = ""; } 
 if(js["Scale"].is_string()) { _hasConditionalVariables = true; _scaleConditional = js["Scale"]; } 
 eraseValue(js, "Scale");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/gamma";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: gamma: \n%s\n", js.dump(2).c_str());
} 

void Gamma::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_shapeConditional == "") js["Shape"] = _shape;
 if(_shapeConditional != "") js["Shape"] = _shapeConditional; 
 if(_scaleConditional == "") js["Scale"] = _scale;
 if(_scaleConditional != "") js["Scale"] = _scaleConditional; 
 Univariate::getConfiguration(js);
} 

void Gamma::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Gamma::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Gamma::getPropertyPointer(const std::string& property)
{
 if (property == "Shape") return &_shape;
 if (property == "Scale") return &_scale;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Gamma.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
