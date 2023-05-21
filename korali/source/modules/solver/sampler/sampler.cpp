#include "modules/solver/sampler/sampler.hpp"

namespace korali
{
namespace solver
{
;

void Sampler::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Lower Bound"))
 {
 try { _k->_variables[i]->_lowerBound = _k->_js["Variables"][i]["Lower Bound"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ sampler ] \n + Key:    ['Lower Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Lower Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Lower Bound'] required by sampler.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Upper Bound"))
 {
 try { _k->_variables[i]->_upperBound = _k->_js["Variables"][i]["Upper Bound"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ sampler ] \n + Key:    ['Upper Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Upper Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Upper Bound'] required by sampler.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Initial Value"))
 {
 try { _k->_variables[i]->_initialValue = _k->_js["Variables"][i]["Initial Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ sampler ] \n + Key:    ['Initial Value']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Value'] required by sampler.\n"); 

 } 
 Solver::setConfiguration(js);
 _type = "sampler";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: sampler: \n%s\n", js.dump(2).c_str());
} 

void Sampler::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Lower Bound"] = _k->_variables[i]->_lowerBound;
   _k->_js["Variables"][i]["Upper Bound"] = _k->_variables[i]->_upperBound;
   _k->_js["Variables"][i]["Initial Value"] = _k->_variables[i]->_initialValue;
 } 
 Solver::getConfiguration(js);
} 

void Sampler::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Termination Criteria\": {}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Sampler::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Value\": -Infinity, \"Lower Bound\": -Infinity, \"Upper Bound\": Infinity}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool Sampler::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;