#include "modules/distribution/univariate/univariate.hpp"
#include "modules/experiment/experiment.hpp"

namespace korali
{
namespace distribution
{
;

void Univariate::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Distribution::setConfiguration(js);
 _type = "univariate";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: univariate: \n%s\n", js.dump(2).c_str());
} 

void Univariate::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Distribution::getConfiguration(js);
} 

void Univariate::applyModuleDefaults(knlohmann::json& js) 
{

 Distribution::applyModuleDefaults(js);
} 

void Univariate::applyVariableDefaults() 
{

 Distribution::applyVariableDefaults();
} 

;

} //distribution
} //korali
;