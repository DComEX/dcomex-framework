#include "modules/distribution/univariate/uniformratio/uniformratio.hpp"
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

double UniformRatio::getDensity(const double z) const
{
  const double b0 = _minimumX / _maximumY;
  const double b3 = _maximumX / _minimumY;

  if (z < b0) return 0.;
  if (z > b3) return 0.;

  return (std::pow(std::min(_maximumY, _maximumX / z), 2.) - std::pow(std::max(_minimumY, _minimumX / z), 2.)) / _aux;
}

double UniformRatio::getLogDensity(const double z) const
{
  const double b0 = _minimumX / _maximumY;
  const double b3 = _maximumX / _minimumY;

  if (z < b0) return -Inf;
  if (z > b3) return -Inf;

  return std::log(std::pow(std::min(_maximumY, _maximumX / z), 2.) - std::pow(std::max(_minimumY, _minimumX / z), 2.)) - std::log(_aux);
}

double UniformRatio::getLogDensityGradient(const double z) const
{
  const double b0 = _minimumX / _maximumY;
  const double b3 = _maximumX / _minimumY;

  const double density = getDensity(z);

  if (z < b0)
    return 0.;
  else if (z > b3)
    return 0.;
  else if (_maximumX / z >= _maximumY && _minimumX / z <= _minimumY)
    return 0.;
  else if (_maximumX / z > _maximumY && _minimumX / z > _minimumY)
    return 2. * _minimumX * _minimumX / (_aux * z * z * z * density);
  else if (_maximumX / z < _maximumY && _minimumX / z < _minimumY)
    return -2 * _maximumX * _maximumX / (_aux * z * z * z * density);
  else
    return 2. * (_minimumX * _minimumX - _maximumX * _maximumX) / (_aux * z * z * z * density);
}

double UniformRatio::getLogDensityHessian(const double z) const
{
  const double b0 = _minimumX / _maximumY;
  const double b3 = _maximumX / _minimumY;

  const double density = getDensity(z);
  const double logGrad = getLogDensityGradient(z);

  if (z < b0)
    return 0.;
  else if (z > b3)
    return 0.;
  else if (_maximumX / z >= _maximumY && _minimumX / z <= _minimumY)
    return 0.;
  else if (_maximumX / z > _maximumY && _minimumX / z > _minimumY)
    return -logGrad * logGrad + -6. * _minimumX * _minimumX / (_aux * z * z * z * z * density);
  else if (_maximumX / z < _maximumY && _minimumX / z < _minimumY)
    return -logGrad * logGrad + 6. * _maximumX * _maximumX / (_aux * z * z * z * z * density);
  else
    return -logGrad * logGrad + 6. * (_minimumX * _minimumX - _maximumX * _maximumX) / (_aux * z * z * z * z * density);
}

double UniformRatio::getRandomNumber()
{
  return gsl_ran_flat(_range, _minimumX, _maximumX) / gsl_ran_flat(_range, _minimumY, _maximumY);
}

void UniformRatio::updateDistribution()
{
  if (_maximumX - _minimumX <= 0.0)
    KORALI_LOG_ERROR("Maximum (%f) bound must be higher than Minimum (%f) bound of the first (dividend) Uniform distribution in order to draw a random number.\n", _maximumX, _minimumX);
  if (_maximumY - _minimumY <= 0.0)
    KORALI_LOG_ERROR("Maximum (%f) bound must be higher than Minimum (%f) bound of the second (divisor) Uniform distribution in order to draw a random number.\n", _maximumY, _minimumY);

  if (_minimumX <= 0.)
    KORALI_LOG_ERROR("Minimum (%f) bound of the first (dividend) Uniform distribution must be larger 0.\n", _minimumX);
  if (_minimumY <= 0.)
    KORALI_LOG_ERROR("Minimum (%f) bound of the second (divisor) Uniform distribution must be larger 0.\n", _minimumY);

  const double b0 = _minimumX / _maximumY;
  const double b1 = _minimumX / _minimumY;
  const double b2 = _maximumX / _maximumY;
  const double b3 = _maximumX / _minimumY;

  _aux = _maximumY * _maximumY * (b1 - b0) + _minimumX * _minimumX * (1. / b1 - 1. / b0) +
         (_maximumY * _maximumY - _minimumY * _minimumY) * (b2 - b1) -
         _maximumX * _maximumX * (1. / b3 - 1. / b2) - _minimumY * _minimumY * (b3 - b2);
}

void UniformRatio::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Minimum X"].is_number()) {_minimumX = js["Minimum X"]; _minimumXConditional = ""; } 
 if(js["Minimum X"].is_string()) { _hasConditionalVariables = true; _minimumXConditional = js["Minimum X"]; } 
 eraseValue(js, "Minimum X");
 if(js["Maximum X"].is_number()) {_maximumX = js["Maximum X"]; _maximumXConditional = ""; } 
 if(js["Maximum X"].is_string()) { _hasConditionalVariables = true; _maximumXConditional = js["Maximum X"]; } 
 eraseValue(js, "Maximum X");
 if(js["Minimum Y"].is_number()) {_minimumY = js["Minimum Y"]; _minimumYConditional = ""; } 
 if(js["Minimum Y"].is_string()) { _hasConditionalVariables = true; _minimumYConditional = js["Minimum Y"]; } 
 eraseValue(js, "Minimum Y");
 if(js["Maximum Y"].is_number()) {_maximumY = js["Maximum Y"]; _maximumYConditional = ""; } 
 if(js["Maximum Y"].is_string()) { _hasConditionalVariables = true; _maximumYConditional = js["Maximum Y"]; } 
 eraseValue(js, "Maximum Y");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/uniformratio";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: uniformratio: \n%s\n", js.dump(2).c_str());
} 

void UniformRatio::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_minimumXConditional == "") js["Minimum X"] = _minimumX;
 if(_minimumXConditional != "") js["Minimum X"] = _minimumXConditional; 
 if(_maximumXConditional == "") js["Maximum X"] = _maximumX;
 if(_maximumXConditional != "") js["Maximum X"] = _maximumXConditional; 
 if(_minimumYConditional == "") js["Minimum Y"] = _minimumY;
 if(_minimumYConditional != "") js["Minimum Y"] = _minimumYConditional; 
 if(_maximumYConditional == "") js["Maximum Y"] = _maximumY;
 if(_maximumYConditional != "") js["Maximum Y"] = _maximumYConditional; 
 Univariate::getConfiguration(js);
} 

void UniformRatio::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void UniformRatio::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* UniformRatio::getPropertyPointer(const std::string& property)
{
 if (property == "Minimum X") return &_minimumX;
 if (property == "Maximum X") return &_maximumX;
 if (property == "Minimum Y") return &_minimumY;
 if (property == "Maximum Y") return &_maximumY;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution UniformRatio.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
