#include "modules/solver/deepSupervisor/optimizers/fAdaGrad/fAdaGrad.hpp"

namespace korali
{
;

void fAdaGrad::initialize()
{
  fGradientBasedOptimizer::initialize();
  _gdiag.resize(_nVars, 0.0f);
  reset();
}

void fAdaGrad::reset()
{
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] = 0.0f;
    _gdiag[i] = 0.0f;
  }
}

void fAdaGrad::processResult(std::vector<float> &gradient)
{
  fGradientBasedOptimizer::preProcessResult(gradient);

#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _gdiag[i] = _gdiag[i] + (gradient[i] * gradient[i]);
    _currentValue[i] += (_eta / std::sqrt(_gdiag[i] + _epsilon)) * gradient[i];
  }

  fGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fAdaGrad::printInternals()
{
  printf("_currentValue[i], _gdiag[i]:\n");
  for (size_t i = 0; i < 10; i++)
    printf("%f %f\n", _currentValue[i], _gdiag[i]);
  fflush(stdout);
}

void fAdaGrad::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Gdiag"))
 {
 try { _gdiag = js["Gdiag"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fAdaGrad ] \n + Key:    ['Gdiag']\n%s", e.what()); } 
   eraseValue(js, "Gdiag");
 }

 fGradientBasedOptimizer::setConfiguration(js);
 _type = "deepSupervisor/optimizers/fAdaGrad";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fAdaGrad: \n%s\n", js.dump(2).c_str());
} 

void fAdaGrad::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Gdiag"] = _gdiag;
 fGradientBasedOptimizer::getConfiguration(js);
} 

void fAdaGrad::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 fGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fAdaGrad::applyVariableDefaults() 
{

 fGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //korali
;
