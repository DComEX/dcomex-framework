#include "modules/solver/deepSupervisor/optimizers/fMadGrad/fMadGrad.hpp"

namespace korali
{
;

void fMadGrad::initialize()
{
  fGradientBasedOptimizer::initialize();
  _initialValue.resize(_nVars, 0.0f);
  _s.resize(_nVars, 0.0f);
  _v.resize(_nVars, 0.0f);
  _z.resize(_nVars, 0.0f);
  reset();
}

void fMadGrad::reset()
{
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] = 0.0f;
    _initialValue[i] = 0.0f;
    _s[i] = 0.0f;
    _v[i] = 0.0f;
    _z[i] = 0.0f;
  }
}

void fMadGrad::processResult(std::vector<float> &gradient)
{
  fGradientBasedOptimizer::preProcessResult(gradient);

  float lambda = _eta; // * std::sqrt((float)_modelEvaluationCount + 1.0f);
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _s[i] = _s[i] + lambda * gradient[i];
    _v[i] = _v[i] - lambda * (gradient[i] * gradient[i]);
    _z[i] = _initialValue[i] - (1.0f / (std::cbrt(_v[i]) + _epsilon)) * _s[i];
    _currentValue[i] = (1.0f - _momentum) * _currentValue[i] + _momentum * _z[i];
  }

  fGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fMadGrad::printInternals()
{
  printf("_currentValue[i], _s[i], _v[i], _z[i]:\n");
  for (size_t i = 0; i < 10; i++)
    printf("%f %f %f %f\n", _currentValue[i], _s[i], _v[i], _z[i]);
  fflush(stdout);
}

void fMadGrad::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Initial Value"))
 {
 try { _initialValue = js["Initial Value"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fMadGrad ] \n + Key:    ['Initial Value']\n%s", e.what()); } 
   eraseValue(js, "Initial Value");
 }

 if (isDefined(js, "s"))
 {
 try { _s = js["s"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fMadGrad ] \n + Key:    ['s']\n%s", e.what()); } 
   eraseValue(js, "s");
 }

 if (isDefined(js, "v"))
 {
 try { _v = js["v"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fMadGrad ] \n + Key:    ['v']\n%s", e.what()); } 
   eraseValue(js, "v");
 }

 if (isDefined(js, "z"))
 {
 try { _z = js["z"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fMadGrad ] \n + Key:    ['z']\n%s", e.what()); } 
   eraseValue(js, "z");
 }

 if (isDefined(js, "Momentum"))
 {
 try { _momentum = js["Momentum"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fMadGrad ] \n + Key:    ['Momentum']\n%s", e.what()); } 
   eraseValue(js, "Momentum");
 }

 fGradientBasedOptimizer::setConfiguration(js);
 _type = "deepSupervisor/optimizers/fMadGrad";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fMadGrad: \n%s\n", js.dump(2).c_str());
} 

void fMadGrad::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Initial Value"] = _initialValue;
   js["s"] = _s;
   js["v"] = _v;
   js["z"] = _z;
   js["Momentum"] = _momentum;
 fGradientBasedOptimizer::getConfiguration(js);
} 

void fMadGrad::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Eta\": 0.001, \"Momentum\": 0.9}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 fGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fMadGrad::applyVariableDefaults() 
{

 fGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //korali
;
