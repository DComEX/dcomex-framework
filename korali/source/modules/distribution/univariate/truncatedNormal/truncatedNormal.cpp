#include "modules/distribution/univariate/truncatedNormal/truncatedNormal.hpp"
#include "modules/experiment/experiment.hpp"
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

#include "auxiliar/rtnorm/rtnorm.hpp"

using namespace Rtnorm;

namespace korali
{
namespace distribution
{
namespace univariate
{
;

double
TruncatedNormal::getDensity(const double x) const
{
  double d = (x - _mu) / _sigma;
  return gsl_sf_exp(-0.5 * d * d) / _normalization;
}

double TruncatedNormal::getLogDensity(const double x) const
{
  double d = (x - _mu) / _sigma;
  return -0.5 * d * d - _logNormalization;
}

double TruncatedNormal::getLogDensityGradient(const double x) const
{
  return (x - _mu) / (_sigma * _sigma);
}

double TruncatedNormal::getLogDensityHessian(const double x) const
{
  return 1. / (_sigma * _sigma);
}

double TruncatedNormal::getRandomNumber()
{
  std::pair<double, double> s = rtnorm(_range, _minimum, _maximum, _mu, _sigma);
  return s.first;
}

void TruncatedNormal::updateDistribution()
{
  if (_sigma <= 0.0) KORALI_LOG_ERROR("Incorrect Standard Deviation parameter of Truncated Normal distribution: %f.\n", _sigma);

  if (_maximum - _minimum <= 0.0)
    KORALI_LOG_ERROR("Incorrect Minimum/Maximum configuration: %f/%f.\n", _minimum, _maximum);

  else
  {
    double a = (_minimum - _mu) / _sigma;
    double b = (_maximum - _mu) / _sigma;

    _normalization = 0.5 * M_SQRT2 * M_SQRTPI * _sigma * (gsl_sf_erf(b * M_SQRT1_2) - gsl_sf_erf(a * M_SQRT1_2));
    _logNormalization = gsl_sf_log(_normalization);
  }
}

void TruncatedNormal::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Mu"].is_number()) {_mu = js["Mu"]; _muConditional = ""; } 
 if(js["Mu"].is_string()) { _hasConditionalVariables = true; _muConditional = js["Mu"]; } 
 eraseValue(js, "Mu");
 if(js["Sigma"].is_number()) {_sigma = js["Sigma"]; _sigmaConditional = ""; } 
 if(js["Sigma"].is_string()) { _hasConditionalVariables = true; _sigmaConditional = js["Sigma"]; } 
 eraseValue(js, "Sigma");
 if(js["Minimum"].is_number()) {_minimum = js["Minimum"]; _minimumConditional = ""; } 
 if(js["Minimum"].is_string()) { _hasConditionalVariables = true; _minimumConditional = js["Minimum"]; } 
 eraseValue(js, "Minimum");
 if(js["Maximum"].is_number()) {_maximum = js["Maximum"]; _maximumConditional = ""; } 
 if(js["Maximum"].is_string()) { _hasConditionalVariables = true; _maximumConditional = js["Maximum"]; } 
 eraseValue(js, "Maximum");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/truncatedNormal";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: truncatedNormal: \n%s\n", js.dump(2).c_str());
} 

void TruncatedNormal::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_muConditional == "") js["Mu"] = _mu;
 if(_muConditional != "") js["Mu"] = _muConditional; 
 if(_sigmaConditional == "") js["Sigma"] = _sigma;
 if(_sigmaConditional != "") js["Sigma"] = _sigmaConditional; 
 if(_minimumConditional == "") js["Minimum"] = _minimum;
 if(_minimumConditional != "") js["Minimum"] = _minimumConditional; 
 if(_maximumConditional == "") js["Maximum"] = _maximum;
 if(_maximumConditional != "") js["Maximum"] = _maximumConditional; 
 Univariate::getConfiguration(js);
} 

void TruncatedNormal::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void TruncatedNormal::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* TruncatedNormal::getPropertyPointer(const std::string& property)
{
 if (property == "Mu") return &_mu;
 if (property == "Sigma") return &_sigma;
 if (property == "Minimum") return &_minimum;
 if (property == "Maximum") return &_maximum;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution TruncatedNormal.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
