#include "modules/solver/deepSupervisor/optimizers/fAdaBelief/fAdaBelief.hpp"

namespace korali
{
;

void fAdaBelief::initialize()
{
  _beta1Pow = 1.0f;
  _beta2Pow = 1.0f;
  fGradientBasedOptimizer::initialize();
  _firstMoment.resize(_nVars, 0.0f);
  _secondCentralMoment.resize(_nVars, 0.0f);
  reset();
}

void fAdaBelief::reset()
{
  _beta1Pow = 1.0f;
  _beta2Pow = 1.0f;
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] = 0.0f;
    _firstMoment[i] = 0.0f;
    _secondCentralMoment[i] = 0.0f;
    ;
  }
}

void fAdaBelief::processResult(std::vector<float> &gradient)
{
  fGradientBasedOptimizer::preProcessResult(gradient);

  float biasCorrectedFirstMoment;
  float secondMomentGradientDiff;
  float biasCorrectedSecondCentralMoment;

  // Calculate powers of beta1 & beta2
  _beta1Pow *= _beta1;
  _beta2Pow *= _beta2;
  const float firstCentralMomentFactor = 1.0f / (1.0f - _beta1Pow);
  const float secondCentralMomentFactor = 1.0f / (1.0f - _beta2Pow);
  const float notBeta1 = 1.0f - _beta1;
  const float notBeta2 = 1.0f - _beta2;

// update first and second moment estimators and parameters
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _firstMoment[i] = _beta1 * _firstMoment[i] - notBeta1 * gradient[i];
    biasCorrectedFirstMoment = _firstMoment[i] * firstCentralMomentFactor;
    secondMomentGradientDiff = gradient[i] + _firstMoment[i];
    _secondCentralMoment[i] = _beta2 * _secondCentralMoment[i] + notBeta2 * secondMomentGradientDiff * secondMomentGradientDiff;

    biasCorrectedSecondCentralMoment = _secondCentralMoment[i] * secondCentralMomentFactor;
    _currentValue[i] -= _eta / (std::sqrt(biasCorrectedSecondCentralMoment) + _epsilon) * biasCorrectedFirstMoment;
  }

  fGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fAdaBelief::printInternals()
{
  printf("_beta1Pow=%f, _beta2Pow=%f, ", _beta1Pow, _beta2Pow);
  printf("_currentValue[i], _firstMoment[i], _secondCentralMoment[i]:\n");
  for (size_t i = 0; i < 10; i++)
    printf("%f %f %f\n", _currentValue[i], _firstMoment[i], _secondCentralMoment[i]);
  fflush(stdout);
}

void fAdaBelief::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Beta1 Pow"))
 {
 try { _beta1Pow = js["Beta1 Pow"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fAdaBelief ] \n + Key:    ['Beta1 Pow']\n%s", e.what()); } 
   eraseValue(js, "Beta1 Pow");
 }

 if (isDefined(js, "Beta2 Pow"))
 {
 try { _beta2Pow = js["Beta2 Pow"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fAdaBelief ] \n + Key:    ['Beta2 Pow']\n%s", e.what()); } 
   eraseValue(js, "Beta2 Pow");
 }

 if (isDefined(js, "First Moment"))
 {
 try { _firstMoment = js["First Moment"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fAdaBelief ] \n + Key:    ['First Moment']\n%s", e.what()); } 
   eraseValue(js, "First Moment");
 }

 if (isDefined(js, "Second Central Moment"))
 {
 try { _secondCentralMoment = js["Second Central Moment"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fAdaBelief ] \n + Key:    ['Second Central Moment']\n%s", e.what()); } 
   eraseValue(js, "Second Central Moment");
 }

 if (isDefined(js, "Beta1"))
 {
 try { _beta1 = js["Beta1"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fAdaBelief ] \n + Key:    ['Beta1']\n%s", e.what()); } 
   eraseValue(js, "Beta1");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Beta1'] required by fAdaBelief.\n"); 

 if (isDefined(js, "Beta2"))
 {
 try { _beta2 = js["Beta2"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fAdaBelief ] \n + Key:    ['Beta2']\n%s", e.what()); } 
   eraseValue(js, "Beta2");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Beta2'] required by fAdaBelief.\n"); 

 fGradientBasedOptimizer::setConfiguration(js);
 _type = "deepSupervisor/optimizers/fAdaBelief";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fAdaBelief: \n%s\n", js.dump(2).c_str());
} 

void fAdaBelief::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Beta1"] = _beta1;
   js["Beta2"] = _beta2;
   js["Beta1 Pow"] = _beta1Pow;
   js["Beta2 Pow"] = _beta2Pow;
   js["First Moment"] = _firstMoment;
   js["Second Central Moment"] = _secondCentralMoment;
 fGradientBasedOptimizer::getConfiguration(js);
} 

void fAdaBelief::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Beta1\": 0.9, \"Beta2\": 0.999}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 fGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fAdaBelief::applyVariableDefaults() 
{

 fGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //korali
;
