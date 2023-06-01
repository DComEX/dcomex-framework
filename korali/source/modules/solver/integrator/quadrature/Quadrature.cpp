#include "engine.hpp"
#include "modules/solver/integrator/quadrature/Quadrature.hpp"

namespace korali
{
namespace solver
{
namespace integrator
{
;

void Quadrature::setInitialConfiguration()
{
  Integrator::setInitialConfiguration();

  _indicesHelper.resize(_variableCount);
  _weight = 1.;

  size_t numEval = 1;
  for (size_t i = 0; i < _variableCount; i++)
  {
    double intervalSize = _k->_variables[i]->_upperBound - _k->_variables[i]->_lowerBound;

    // Initialize quadrature weight
    if (_method == "Rectangle")
    {
      if (_k->_variables[i]->_numberOfGridpoints <= 0) KORALI_LOG_ERROR("'Number Of Gridpoints' for variable %s must be larger 0", _k->_variables[i]->_name.c_str());
      _weight *= intervalSize / (_k->_variables[i]->_numberOfGridpoints);
    }
    else if (_method == "Trapezoidal")
    {
      if (_k->_variables[i]->_numberOfGridpoints <= 1) KORALI_LOG_ERROR("'Number Of Gridpoints' for variable %s must be larger 1", _k->_variables[i]->_name.c_str());
      _weight *= intervalSize / (_k->_variables[i]->_numberOfGridpoints - 1.);
    }
    else if (_method == "Simpson")
    {
      if (_k->_variables[i]->_numberOfGridpoints <= 2) KORALI_LOG_ERROR("'Number Of Gridpoints' for variable %s must be larger 2", _k->_variables[i]->_name.c_str());
      if (_k->_variables[i]->_numberOfGridpoints % 2 == 0) KORALI_LOG_ERROR("'Number Of Gridpoints' for variable %s must be odd", _k->_variables[i]->_name.c_str());
      _weight *= intervalSize / (3. * (_k->_variables[i]->_numberOfGridpoints - 1.));
    }

    // Initialize indices helper
    if (i == 0)
      _indicesHelper[i] = 1;
    else
      _indicesHelper[i] = _k->_variables[i - 1]->_numberOfGridpoints * _indicesHelper[i - 1];

    numEval *= _k->_variables[i]->_numberOfGridpoints;
  }

  // Init max model evaluations
  _maxModelEvaluations = std::min(_maxModelEvaluations, numEval);
}

void Quadrature::launchSample(size_t sampleIndex)
{
  std::vector<float> params(_variableCount);

  const size_t index = _gridPoints.size();

  float weight = _weight;
  // Calculate params and adjust weights
  for (size_t d = 0; d < _variableCount; ++d)
  {
    const size_t dimIdx = (size_t)(index / _indicesHelper[d]) % _k->_variables[d]->_numberOfGridpoints;

    if (_method == "Rectangle")
    {
      params[d] = (dimIdx + 1) * (_k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound) / _k->_variables[d]->_numberOfGridpoints;
    }
    else if (_method == "Trapezoidal")
    {
      if ((dimIdx == 0) || (dimIdx == _k->_variables[d]->_numberOfGridpoints - 1)) weight *= 0.5;

      params[d] = dimIdx * (_k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound) / (_k->_variables[d]->_numberOfGridpoints - 1.);
    }
    else if (_method == "Simpson")
    {
      if ((dimIdx == 0) || (dimIdx == _k->_variables[d]->_numberOfGridpoints - 1))
        weight *= 1.;
      else if (dimIdx % 2 == 0)
        weight *= 2.;
      else
        weight *= 4.;

      params[d] = dimIdx * (_k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound) / (_k->_variables[d]->_numberOfGridpoints - 1.);
    }
  }

  // Store parameter
  _gridPoints.push_back(params);

  _samples[sampleIndex]["Sample Id"] = sampleIndex;
  _samples[sampleIndex]["Module"] = "Problem";
  _samples[sampleIndex]["Operation"] = "Execute";
  _samples[sampleIndex]["Parameters"] = params;
  _samples[sampleIndex]["Weight"] = weight;

  KORALI_START(_samples[sampleIndex]);
}

void Quadrature::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Indices Helper"))
 {
 try { _indicesHelper = js["Indices Helper"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ quadrature ] \n + Key:    ['Indices Helper']\n%s", e.what()); } 
   eraseValue(js, "Indices Helper");
 }

 if (isDefined(js, "Method"))
 {
 try { _method = js["Method"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ quadrature ] \n + Key:    ['Method']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_method == "Rectangle") validOption = true; 
 if (_method == "Trapezoidal") validOption = true; 
 if (_method == "Simpson") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Method'] required by quadrature.\n", _method.c_str()); 
}
   eraseValue(js, "Method");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Method'] required by quadrature.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Number Of Gridpoints"))
 {
 try { _k->_variables[i]->_numberOfGridpoints = _k->_js["Variables"][i]["Number Of Gridpoints"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ quadrature ] \n + Key:    ['Number Of Gridpoints']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Number Of Gridpoints");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Number Of Gridpoints'] required by quadrature.\n"); 

 } 
 Integrator::setConfiguration(js);
 _type = "integrator/quadrature";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: quadrature: \n%s\n", js.dump(2).c_str());
} 

void Quadrature::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Method"] = _method;
   js["Indices Helper"] = _indicesHelper;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Number Of Gridpoints"] = _k->_variables[i]->_numberOfGridpoints;
 } 
 Integrator::getConfiguration(js);
} 

void Quadrature::applyModuleDefaults(knlohmann::json& js) 
{

 Integrator::applyModuleDefaults(js);
} 

void Quadrature::applyVariableDefaults() 
{

 Integrator::applyVariableDefaults();
} 

;

} //integrator
} //solver
} //korali
;
