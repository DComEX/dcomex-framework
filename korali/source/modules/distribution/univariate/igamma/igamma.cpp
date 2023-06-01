#include "modules/distribution/univariate/igamma/igamma.hpp"
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
Igamma::getDensity(const double x) const
{
  if (x <= 0.0) return -INFINITY;
  return _aux * std::pow(x, -_shape - 1.) * std::exp(-_scale / x);
}

double Igamma::getLogDensity(const double x) const
{
  if (x <= 0) return -INFINITY;
  return _aux - (_shape + 1) * std::log(x) - (_scale / x);
}

double Igamma::getLogDensityGradient(const double x) const
{
  if (x <= 0) return -INFINITY;
  return _scale / (x * x) - (_shape + 1.) / x;
}

double Igamma::getLogDensityHessian(const double x) const
{
  if (x <= 0) return -INFINITY;
  return -2. * _scale / (x * x * x) + (_shape + 1.) / (x * x);
}

double Igamma::getRandomNumber()
{
  return 1. / gsl_ran_gamma(_range, _shape, 1. / _scale);
}

void Igamma::updateDistribution()
{
  if (_shape <= 0.0) KORALI_LOG_ERROR("Incorrect Shape parameter of Inverse Gamma distribution: %f.\n", _shape);
  if (_scale <= 0.0) KORALI_LOG_ERROR("Incorrect Scale parameter of Inverse Gamma distribution: %f.\n", _scale);

  _auxLog = _shape * log(_scale) - gsl_sf_lngamma(_shape);
  _aux = std::exp(_auxLog);
}

void Igamma::setConfiguration(knlohmann::json& js) 
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
 _type = "univariate/igamma";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: igamma: \n%s\n", js.dump(2).c_str());
} 

void Igamma::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_shapeConditional == "") js["Shape"] = _shape;
 if(_shapeConditional != "") js["Shape"] = _shapeConditional; 
 if(_scaleConditional == "") js["Scale"] = _scale;
 if(_scaleConditional != "") js["Scale"] = _scaleConditional; 
 Univariate::getConfiguration(js);
} 

void Igamma::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Igamma::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Igamma::getPropertyPointer(const std::string& property)
{
 if (property == "Shape") return &_shape;
 if (property == "Scale") return &_scale;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Igamma.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
