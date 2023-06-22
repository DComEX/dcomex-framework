#include "modules/distribution/multivariate/multivariate.hpp"
#include "modules/experiment/experiment.hpp"

namespace korali
{
namespace distribution
{
;

void Multivariate::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Distribution::setConfiguration(js);
 _type = "multivariate";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: multivariate: \n%s\n", js.dump(2).c_str());
} 

void Multivariate::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Distribution::getConfiguration(js);
} 

void Multivariate::applyModuleDefaults(knlohmann::json& js) 
{

 Distribution::applyModuleDefaults(js);
} 

void Multivariate::applyVariableDefaults() 
{

 Distribution::applyVariableDefaults();
} 

;

} //distribution
} //korali
;
