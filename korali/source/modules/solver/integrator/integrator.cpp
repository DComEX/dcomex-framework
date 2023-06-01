#include "engine.hpp"
#include "modules/solver/integrator/integrator.hpp"

namespace korali
{
namespace solver
{
;

void Integrator::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  for (size_t i = 0; i < _variableCount; ++i)
  {
    if (_k->_variables[i]->_upperBound <= _k->_variables[i]->_lowerBound) KORALI_LOG_ERROR("'Upper Bound' is not strictly bigger then 'Lower Bound' for variable %s.\n", _k->_variables[i]->_name.c_str());
  }

  _accumulatedIntegral = 0.;
}

void Integrator::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  _executionsPerGeneration = std::min(_executionsPerGeneration, _maxModelEvaluations - _modelEvaluationCount);
  _samples.resize(_executionsPerGeneration);

  for (size_t i = 0; i < _executionsPerGeneration; i++)
  {
    launchSample(i);
    _modelEvaluationCount++;
  }

  KORALI_WAITALL(_samples);

  for (size_t i = 0; i < _executionsPerGeneration; i++)
  {
    auto f = KORALI_GET(double, _samples[i], "Evaluation");
    auto w = KORALI_GET(double, _samples[i], "Weight");
    _accumulatedIntegral += w * f;
  }

  (*_k)["Results"]["Integral"] = _accumulatedIntegral;
}

void Integrator::printGenerationBefore()
{
}

void Integrator::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Total evaluations accumulated %lu/%lu.\n", _modelEvaluationCount, _maxModelEvaluations);
}

void Integrator::finalize()
{
  _k->_logger->logInfo("Minimal", "Integral Calculated: %e\n", _accumulatedIntegral);
}

void Integrator::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Accumulated Integral"))
 {
 try { _accumulatedIntegral = js["Accumulated Integral"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Accumulated Integral']\n%s", e.what()); } 
   eraseValue(js, "Accumulated Integral");
 }

 if (isDefined(js, "Grid Points"))
 {
 try { _gridPoints = js["Grid Points"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Grid Points']\n%s", e.what()); } 
   eraseValue(js, "Grid Points");
 }

 if (isDefined(js, "Weight"))
 {
 try { _weight = js["Weight"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Weight']\n%s", e.what()); } 
   eraseValue(js, "Weight");
 }

 if (isDefined(js, "Executions Per Generation"))
 {
 try { _executionsPerGeneration = js["Executions Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Executions Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Executions Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Executions Per Generation'] required by integrator.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Lower Bound"))
 {
 try { _k->_variables[i]->_lowerBound = _k->_js["Variables"][i]["Lower Bound"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Lower Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Lower Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Lower Bound'] required by integrator.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Upper Bound"))
 {
 try { _k->_variables[i]->_upperBound = _k->_js["Variables"][i]["Upper Bound"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Upper Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Upper Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Upper Bound'] required by integrator.\n"); 

 } 
 Solver::setConfiguration(js);
 _type = "integrator";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: integrator: \n%s\n", js.dump(2).c_str());
} 

void Integrator::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Executions Per Generation"] = _executionsPerGeneration;
   js["Accumulated Integral"] = _accumulatedIntegral;
   js["Grid Points"] = _gridPoints;
   js["Weight"] = _weight;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Lower Bound"] = _k->_variables[i]->_lowerBound;
   _k->_js["Variables"][i]["Upper Bound"] = _k->_variables[i]->_upperBound;
 } 
 Solver::getConfiguration(js);
} 

void Integrator::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Executions Per Generation\": 100}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Integrator::applyVariableDefaults() 
{

 Solver::applyVariableDefaults();
} 

;

} //solver
} //korali
;
