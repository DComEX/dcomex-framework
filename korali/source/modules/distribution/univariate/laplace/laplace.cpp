#include "modules/distribution/univariate/laplace/laplace.hpp"
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
Laplace::getDensity(const double x) const
{
  return gsl_ran_laplace_pdf(x - _mean, _width);
}

double Laplace::getLogDensity(const double x) const
{
  return _aux - fabs(x - _mean) / _width;
}

double Laplace::getLogDensityGradient(const double x) const
{
  if (x >= _mean)
    return 1.0 / _width;
  else
    return -1.0 / _width;
}

double Laplace::getLogDensityHessian(const double x) const
{
  return 0.;
}

double Laplace::getRandomNumber()
{
  return _mean + gsl_ran_laplace(_range, _width);
}

void Laplace::updateDistribution()
{
  if (_width <= 0.0) KORALI_LOG_ERROR("Incorrect Width parameter of Laplace distribution: %f.\n", _width);

  _aux = -gsl_sf_log(2. * _width);
}

void Laplace::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

  _hasConditionalVariables = false; 
 if(js["Mean"].is_number()) {_mean = js["Mean"]; _meanConditional = ""; } 
 if(js["Mean"].is_string()) { _hasConditionalVariables = true; _meanConditional = js["Mean"]; } 
 eraseValue(js, "Mean");
 if(js["Width"].is_number()) {_width = js["Width"]; _widthConditional = ""; } 
 if(js["Width"].is_string()) { _hasConditionalVariables = true; _widthConditional = js["Width"]; } 
 eraseValue(js, "Width");
 if (_hasConditionalVariables == false) updateDistribution(); // If distribution is not conditioned to external values, update from the beginning 

 Univariate::setConfiguration(js);
 _type = "univariate/laplace";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: laplace: \n%s\n", js.dump(2).c_str());
} 

void Laplace::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 if(_meanConditional == "") js["Mean"] = _mean;
 if(_meanConditional != "") js["Mean"] = _meanConditional; 
 if(_widthConditional == "") js["Width"] = _width;
 if(_widthConditional != "") js["Width"] = _widthConditional; 
 Univariate::getConfiguration(js);
} 

void Laplace::applyModuleDefaults(knlohmann::json& js) 
{

 Univariate::applyModuleDefaults(js);
} 

void Laplace::applyVariableDefaults() 
{

 Univariate::applyVariableDefaults();
} 

double* Laplace::getPropertyPointer(const std::string& property)
{
 if (property == "Mean") return &_mean;
 if (property == "Width") return &_width;
 KORALI_LOG_ERROR(" + Property %s not recognized for distribution Laplace.\n", property.c_str());
 return NULL;
}

;

} //univariate
} //distribution
} //korali
;
