#include "modules/distribution/univariate/logNormal/logNormal.hpp"
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
LogNormal::getDensity(const double x) const
{
  return gsl_ran_lognormal_pdf(x, _mu, _sigma);
}

double LogNormal::getLogDensity(const double x) const
{
  if (x <= 0) return -INFINITY;
  double logx = std::log(x);
  double d = (logx - _mu) / _sigma;
  return _aux - logx - 0.5 * d * d;
}

double LogNormal::getLogDensityGradient(const double x) const
{
  if (x <= 0) return 0.;
  double d = (std::log(x) - _mu) / _sigma;
  return -1. / x - d / (x * _sigma);
}

double LogNormal::getLogDensityHessian(const double x) const
{
  if (x <= 0) return 0.;
  double d = (std::log(x) - _mu) / _sigma;
  return 1. / (x * x) - d / ((x * _sigma) * (x * _sigma)) + d / (x * x * _sigma);
}

double LogNormal::getRandomNumber()
{
  return gsl_ran_lognormal(_range, _mu, _sigma);
}

void LogNormal::updateDistribution()
{
  if (_sigma <= 0.0) KORALI_LOG_ERROR("Incorrect Sigma parameter of LogNormal distribution: %f.\n", _sigma);

  _aux = -0.5 * gsl_sf_log(2 * M_PI) - gsl_sf_log(_sigma);
}

void LogNormal::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Mu"].is_number()) {_mu = js["Mu"]; _muConditional = ""; } 
 if(js["Mu"].is_string()) { _hasConditionalVariables = true; _muConditional = js["Mu"]; } 
 eraseValue(js, "Mu");
 if(js["Sigma"].is_number()) {_sigma = js["Sigma"]; _sigmaConditional = ""; } 
 if(js["Sigma"].is_string()) { _hasConditionalVariables = true; _sigmaConditional = js["Sigma"]; } 
 eraseValue(js, "Sigma");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/logNormal";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: logNormal: \n%s\n", js.dump(2).c_str());
} 

void LogNormal::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_muConditional == "") js["Mu"] = _mu;
 if(_muConditional != "") js["Mu"] = _muConditional; 
 if(_sigmaConditional == "") js["Sigma"] = _sigma;
 if(_sigmaConditional != "") js["Sigma"] = _sigmaConditional; 
 Univariate::getConfiguration(js);
} 

void LogNormal::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void LogNormal::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* LogNormal::getPropertyPointer(const std::string& property)
{
 if (property == "Mu") return &_mu;
 if (property == "Sigma") return &_sigma;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution LogNormal.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
