#include "modules/distribution/univariate/beta/beta.hpp"
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
Beta::getDensity(const double x) const
{
  return gsl_ran_beta_pdf(x, _alpha, _beta);
}

double Beta::getLogDensity(const double x) const
{
  if (x < 0.) return -INFINITY;
  if (x > 1.) return -INFINITY;
  return _aux + (_alpha - 1.) * std::log(x) + (_beta - 1.) * std::log(1. - x);
}

double Beta::getRandomNumber()
{
  return gsl_ran_beta(_range, _alpha, _beta);
}

double Beta::getLogDensityGradient(const double x) const
{
  if (x < 0.) return 0.;
  if (x > 1.) return 0.;
  return (_alpha - 1.) / x - (_beta - 1.) / (1. - x);
}

double Beta::getLogDensityHessian(const double x) const
{
  if (x < 0.) return 0.;
  if (x > 1.) return 0.;
  return (1. - _alpha) / (x * x) - (_beta - 1.) / ((1. - x) * (1. - x));
}

void Beta::updateDistribution()
{
  if (_alpha <= 0.0) KORALI_LOG_ERROR("Incorrect Shape parameter (alpha) of Beta distribution: %f.\n", _alpha);
  if (_beta <= 0.0) KORALI_LOG_ERROR("Incorrect Shape (beta) parameter of Beta distribution: %f.\n", _beta);

  _aux = gsl_sf_lngamma(_alpha + _beta) - gsl_sf_lngamma(_alpha) - gsl_sf_lngamma(_beta);
}

void Beta::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Alpha"].is_number()) {_alpha = js["Alpha"]; _alphaConditional = ""; } 
 if(js["Alpha"].is_string()) { _hasConditionalVariables = true; _alphaConditional = js["Alpha"]; } 
 eraseValue(js, "Alpha");
 if(js["Beta"].is_number()) {_beta = js["Beta"]; _betaConditional = ""; } 
 if(js["Beta"].is_string()) { _hasConditionalVariables = true; _betaConditional = js["Beta"]; } 
 eraseValue(js, "Beta");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/beta";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: beta: \n%s\n", js.dump(2).c_str());
} 

void Beta::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_alphaConditional == "") js["Alpha"] = _alpha;
 if(_alphaConditional != "") js["Alpha"] = _alphaConditional; 
 if(_betaConditional == "") js["Beta"] = _beta;
 if(_betaConditional != "") js["Beta"] = _betaConditional; 
 Univariate::getConfiguration(js);
} 

void Beta::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Beta::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Beta::getPropertyPointer(const std::string& property)
{
 if (property == "Alpha") return &_alpha;
 if (property == "Beta") return &_beta;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Beta.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
