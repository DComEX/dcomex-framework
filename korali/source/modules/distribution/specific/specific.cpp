#include "modules/distribution/specific/specific.hpp"
#include "modules/experiment/experiment.hpp"

namespace korali
{
namespace distribution
{
;

void Specific::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Distribution::setConfiguration(js);
 _type = "specific";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: specific: \n%s\n", js.dump(2).c_str());
} 

void Specific::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Distribution::getConfiguration(js);
} 

void Specific::applyModuleDefaults(knlohmann::json& js) 
{

 Distribution::applyModuleDefaults(js);
} 

void Specific::applyVariableDefaults() 
{

 Distribution::applyVariableDefaults();
} 

;

} //distribution
} //korali
;