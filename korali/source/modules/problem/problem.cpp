#include "modules/problem/problem.hpp"

namespace korali
{
;

void Problem::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Name"))
 {
 try { _k->_variables[i]->_name = _k->_js["Variables"][i]["Name"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ problem ] \n + Key:    ['Name']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Name");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Name'] required by problem.\n"); 

 } 
 Module::setConfiguration(js);
 _type = ".";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: problem: \n%s\n", js.dump(2).c_str());
} 

void Problem::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Name"] = _k->_variables[i]->_name;
 } 
 Module::getConfiguration(js);
} 

void Problem::applyModuleDefaults(knlohmann::json& js) 
{

 Module::applyModuleDefaults(js);
} 

void Problem::applyVariableDefaults() 
{

 Module::applyVariableDefaults();
} 

;

} //korali
;
