#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/experiment/experiment.hpp"
#include <gsl/gsl_math.h>
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
Normal::getDensity(const double x) const
{
  double y = (x - _mean) / _standardDeviation;
  return gsl_sf_exp(-0.5 * y * y) / _normalization;
}

double Normal::getLogDensity(const double x) const
{
  double d = (x - _mean) / _standardDeviation;
  return _logNormalization - 0.5 * d * d;
}

double Normal::getLogDensityGradient(const double x) const
{
  return (x - _mean) / (_standardDeviation * _standardDeviation);
}

double Normal::getLogDensityHessian(const double x) const
{
  return 1. / (_standardDeviation * _standardDeviation);
}

double Normal::getRandomNumber()
{
  return _mean + gsl_ran_gaussian(_range, _standardDeviation);
}

void Normal::updateDistribution()
{
  if (_standardDeviation <= 0.0) KORALI_LOG_ERROR("Incorrect Standard Deviation parameter of Normal distribution: %f.\n", _standardDeviation);

  _normalization = M_SQRT2 * M_SQRTPI * _standardDeviation;
  _logNormalization = -0.5 * gsl_sf_log(2 * M_PI) - gsl_sf_log(_standardDeviation);
}

void Normal::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Mean"].is_number()) {_mean = js["Mean"]; _meanConditional = ""; } 
 if(js["Mean"].is_string()) { _hasConditionalVariables = true; _meanConditional = js["Mean"]; } 
 eraseValue(js, "Mean");
 if(js["Standard Deviation"].is_number()) {_standardDeviation = js["Standard Deviation"]; _standardDeviationConditional = ""; } 
 if(js["Standard Deviation"].is_string()) { _hasConditionalVariables = true; _standardDeviationConditional = js["Standard Deviation"]; } 
 eraseValue(js, "Standard Deviation");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/normal";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: normal: \n%s\n", js.dump(2).c_str());
} 

void Normal::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_meanConditional == "") js["Mean"] = _mean;
 if(_meanConditional != "") js["Mean"] = _meanConditional; 
 if(_standardDeviationConditional == "") js["Standard Deviation"] = _standardDeviation;
 if(_standardDeviationConditional != "") js["Standard Deviation"] = _standardDeviationConditional; 
 Univariate::getConfiguration(js);
} 

void Normal::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Normal::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Normal::getPropertyPointer(const std::string& property)
{
 if (property == "Mean") return &_mean;
 if (property == "Standard Deviation") return &_standardDeviation;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Normal.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
