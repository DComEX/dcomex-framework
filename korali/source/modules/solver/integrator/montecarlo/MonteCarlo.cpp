#include "engine.hpp"
#include "modules/solver/integrator/montecarlo/MonteCarlo.hpp"

namespace korali
{
namespace solver
{
namespace integrator
{
;

void MonteCarlo::setInitialConfiguration()
{
  Integrator::setInitialConfiguration();

  // Calculate weight
  _weight = 1. / (double)_numberOfSamples;
  for (size_t d = 0; d < _variableCount; ++d)
  {
    _weight *= (_k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound);
  }

  // Init max model evaluations
  _maxModelEvaluations = std::min(_maxModelEvaluations, _numberOfSamples);
}

void MonteCarlo::launchSample(size_t sampleIndex)
{
  std::vector<float> params(_variableCount);

  /// Uniformly sample parameter
  for (size_t d = 0; d < _variableCount; ++d)
  {
    params[d] = (_k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound) * _uniformGenerator->getRandomNumber();
  }

  _samples[sampleIndex]["Sample Id"] = sampleIndex;
  _samples[sampleIndex]["Module"] = "Problem";
  _samples[sampleIndex]["Operation"] = "Execute";
  _samples[sampleIndex]["Parameters"] = params;
  _samples[sampleIndex]["Weight"] = _weight;

  // Store parameter
  _gridPoints.push_back(params);

  KORALI_START(_samples[sampleIndex]);
}

void MonteCarlo::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Number Of Samples"))
 {
 try { _numberOfSamples = js["Number Of Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ montecarlo ] \n + Key:    ['Number Of Samples']\n%s", e.what()); } 
   eraseValue(js, "Number Of Samples");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Number Of Samples'] required by montecarlo.\n"); 

 Integrator::setConfiguration(js);
 _type = "integrator/montecarlo";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: montecarlo: \n%s\n", js.dump(2).c_str());
} 

void MonteCarlo::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Number Of Samples"] = _numberOfSamples;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
 Integrator::getConfiguration(js);
} 

void MonteCarlo::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Integrator::applyModuleDefaults(js);
} 

void MonteCarlo::applyVariableDefaults() 
{

 Integrator::applyVariableDefaults();
} 

;

} //integrator
} //solver
} //korali
;
