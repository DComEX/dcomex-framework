#include "modules/solver/optimizer/optimizer.hpp"

namespace korali
{
namespace solver
{
;

bool Optimizer::isSampleFeasible(const std::vector<double> &sample)
{
  for (size_t i = 0; i < sample.size(); i++)
  {
    if (std::isfinite(sample[i]) == false)
    {
      _infeasibleSampleCount++;
      return false;
    }
    if (sample[i] < _k->_variables[i]->_lowerBound)
    {
      _infeasibleSampleCount++;
      return false;
    }
    if (sample[i] > _k->_variables[i]->_upperBound)
    {
      _infeasibleSampleCount++;
      return false;
    }
  }
  return true;
}

void Optimizer::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Current Best Value"))
 {
 try { _currentBestValue = js["Current Best Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Current Best Value']\n%s", e.what()); } 
   eraseValue(js, "Current Best Value");
 }

 if (isDefined(js, "Previous Best Value"))
 {
 try { _previousBestValue = js["Previous Best Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Previous Best Value']\n%s", e.what()); } 
   eraseValue(js, "Previous Best Value");
 }

 if (isDefined(js, "Best Ever Value"))
 {
 try { _bestEverValue = js["Best Ever Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Best Ever Value']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Value");
 }

 if (isDefined(js, "Best Ever Variables"))
 {
 try { _bestEverVariables = js["Best Ever Variables"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Best Ever Variables']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Variables");
 }

 if (isDefined(js, "Infeasible Sample Count"))
 {
 try { _infeasibleSampleCount = js["Infeasible Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Infeasible Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Infeasible Sample Count");
 }

 if (isDefined(js, "Termination Criteria", "Max Value"))
 {
 try { _maxValue = js["Termination Criteria"]["Max Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Termination Criteria']['Max Value']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Value'] required by optimizer.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Value Difference Threshold"))
 {
 try { _minValueDifferenceThreshold = js["Termination Criteria"]["Min Value Difference Threshold"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Termination Criteria']['Min Value Difference Threshold']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Value Difference Threshold");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Value Difference Threshold'] required by optimizer.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Infeasible Resamplings"))
 {
 try { _maxInfeasibleResamplings = js["Termination Criteria"]["Max Infeasible Resamplings"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Termination Criteria']['Max Infeasible Resamplings']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Infeasible Resamplings");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Infeasible Resamplings'] required by optimizer.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Lower Bound"))
 {
 try { _k->_variables[i]->_lowerBound = _k->_js["Variables"][i]["Lower Bound"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Lower Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Lower Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Lower Bound'] required by optimizer.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Upper Bound"))
 {
 try { _k->_variables[i]->_upperBound = _k->_js["Variables"][i]["Upper Bound"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Upper Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Upper Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Upper Bound'] required by optimizer.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Initial Value"))
 {
 try { _k->_variables[i]->_initialValue = _k->_js["Variables"][i]["Initial Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Initial Value']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Value'] required by optimizer.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Initial Mean"))
 {
 try { _k->_variables[i]->_initialMean = _k->_js["Variables"][i]["Initial Mean"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Initial Mean']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Mean");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Mean'] required by optimizer.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Initial Standard Deviation"))
 {
 try { _k->_variables[i]->_initialStandardDeviation = _k->_js["Variables"][i]["Initial Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Initial Standard Deviation']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Standard Deviation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Standard Deviation'] required by optimizer.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Minimum Standard Deviation Update"))
 {
 try { _k->_variables[i]->_minimumStandardDeviationUpdate = _k->_js["Variables"][i]["Minimum Standard Deviation Update"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Minimum Standard Deviation Update']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Minimum Standard Deviation Update");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Minimum Standard Deviation Update'] required by optimizer.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Values"))
 {
 try { _k->_variables[i]->_values = _k->_js["Variables"][i]["Values"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ optimizer ] \n + Key:    ['Values']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Values");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Values'] required by optimizer.\n"); 

 } 
 Solver::setConfiguration(js);
 _type = "optimizer";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: optimizer: \n%s\n", js.dump(2).c_str());
} 

void Optimizer::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Termination Criteria"]["Max Value"] = _maxValue;
   js["Termination Criteria"]["Min Value Difference Threshold"] = _minValueDifferenceThreshold;
   js["Termination Criteria"]["Max Infeasible Resamplings"] = _maxInfeasibleResamplings;
   js["Current Best Value"] = _currentBestValue;
   js["Previous Best Value"] = _previousBestValue;
   js["Best Ever Value"] = _bestEverValue;
   js["Best Ever Variables"] = _bestEverVariables;
   js["Infeasible Sample Count"] = _infeasibleSampleCount;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Lower Bound"] = _k->_variables[i]->_lowerBound;
   _k->_js["Variables"][i]["Upper Bound"] = _k->_variables[i]->_upperBound;
   _k->_js["Variables"][i]["Initial Value"] = _k->_variables[i]->_initialValue;
   _k->_js["Variables"][i]["Initial Mean"] = _k->_variables[i]->_initialMean;
   _k->_js["Variables"][i]["Initial Standard Deviation"] = _k->_variables[i]->_initialStandardDeviation;
   _k->_js["Variables"][i]["Minimum Standard Deviation Update"] = _k->_variables[i]->_minimumStandardDeviationUpdate;
   _k->_js["Variables"][i]["Values"] = _k->_variables[i]->_values;
 } 
 Solver::getConfiguration(js);
} 

void Optimizer::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Termination Criteria\": {\"Max Value\": Infinity, \"Min Value Difference Threshold\": -Infinity, \"Max Infeasible Resamplings\": 1000000}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Optimizer::applyVariableDefaults() 
{

 std::string defaultString = "{\"Lower Bound\": -Infinity, \"Upper Bound\": Infinity, \"Initial Value\": NaN, \"Initial Mean\": NaN, \"Initial Standard Deviation\": NaN, \"Minimum Standard Deviation Update\": 0.0, \"Values\": []}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool Optimizer::checkTermination()
{
 bool hasFinished = false;

 if (_k->_currentGeneration > 1 && (+_bestEverValue > _maxValue))
 {
  _terminationCriteria.push_back("optimizer['Max Value'] = " + std::to_string(_maxValue) + ".");
  hasFinished = true;
 }

 if (_k->_currentGeneration > 1 && (fabs(_currentBestValue - _previousBestValue) < _minValueDifferenceThreshold))
 {
  _terminationCriteria.push_back("optimizer['Min Value Difference Threshold'] = " + std::to_string(_minValueDifferenceThreshold) + ".");
  hasFinished = true;
 }

 if ((_maxInfeasibleResamplings > 0) && (_infeasibleSampleCount >= _maxInfeasibleResamplings))
 {
  _terminationCriteria.push_back("optimizer['Max Infeasible Resamplings'] = " + std::to_string(_maxInfeasibleResamplings) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;
