
#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/optimizer/Rprop/Rprop.hpp"
#include "sample/sample.hpp"

#include <stdio.h>

namespace korali
{
namespace solver
{
namespace optimizer
{
;

void Rprop::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  for (size_t i = 0; i < _variableCount; i++)
    if (std::isfinite(_k->_variables[i]->_initialValue) == false)
      KORALI_LOG_ERROR("Initial Value of variable \'%s\' not defined (no defaults can be calculated).\n", _k->_variables[i]->_name.c_str());

  _currentVariable.resize(_variableCount, 0.0);
  for (size_t i = 0; i < _variableCount; i++)
    _currentVariable[i] = _k->_variables[i]->_initialValue;

  _bestEverVariables = _currentVariable;
  _delta.resize(_variableCount, _delta0);
  _currentGradient.resize(_variableCount, 0);
  _bestEverGradient.resize(_variableCount, 0);
  _previousGradient.resize(_variableCount, 0.0);

  _bestEverValue = Inf;
  _xDiff = Inf;
  _maxStallCounter = 0;
  _normPreviousGradient = Inf;
  _previousBestValue = Inf;
}

void Rprop::evaluateFunctionAndGradient(Sample &sample)
{
  // Initializing Sample Evaluation
  sample["Module"] = "Problem";
  sample["Operation"] = "Evaluate With Gradients";
  sample["Parameters"] = _currentVariable;
  sample["Sample Id"] = 0;
  _modelEvaluationCount++;
  KORALI_START(sample);

  // Waiting for samples to finish
  KORALI_WAIT(sample);

  // Processing results
  // The 'minus' is there because we want Rprop to do Maximization be default.
  _currentBestValue = -KORALI_GET(double, sample, "F(x)");
  _currentGradient = KORALI_GET(std::vector<double>, sample, "Gradient");

  for (size_t i = 0; i < _variableCount; i++)
    _currentGradient[i] = -_currentGradient[i];
}

void Rprop::runGeneration(void)
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  Sample sample;

  evaluateFunctionAndGradient(sample);

  performUpdate();

  _previousBestValue = _currentBestValue;
  _previousGradient = _currentGradient;
  _normPreviousGradient = vectorNorm(_previousGradient);

  if (_currentBestValue < _bestEverValue)
  {
    _bestEverValue = _currentBestValue;

    std::vector<double> tmp(_variableCount);
    for (size_t j = 0; j < _variableCount; j++) tmp[j] = _bestEverVariables[j] - _currentVariable[j];
    _xDiff = vectorNorm(tmp);
    _bestEverVariables = _currentVariable;
    _bestEverGradient = _currentGradient;
    _maxStallCounter = 0;
  }
  else
  {
    _maxStallCounter++;
  }
}

// iRprop_minus
void Rprop::performUpdate(void)
{
  for (size_t i = 0; i < _variableCount; i++)
  {
    double productGradient = _previousGradient[i] * _currentGradient[i];
    if (productGradient > 0)
    {
      _delta[i] = std::min(_delta[i] * _etaPlus, _deltaMax);
    }
    else if (productGradient < 0)
    {
      _delta[i] = std::max(_delta[i] * _etaMinus, _deltaMin);
      _currentGradient[i] = 0;
    }
    _currentVariable[i] += -sign(_currentGradient[i]) * _delta[i];
  }
}

void Rprop::printGenerationBefore()
{
  return;
}

void Rprop::printGenerationAfter()
{
  _k->_logger->logInfo("Normal", "X = [ ");
  for (size_t k = 0; k < _variableCount; k++) _k->_logger->logData("Normal", " %.5le  ", _currentVariable[k]);
  _k->_logger->logData("Normal", " ]\n");

  _k->_logger->logInfo("Normal", "F(X) = %le \n", _currentBestValue);

  _k->_logger->logInfo("Normal", "DF(X) = [ ");
  for (size_t k = 0; k < _variableCount; k++) _k->_logger->logData("Normal", " %.5le  ", _currentGradient[k]);
  _k->_logger->logData("Normal", " ]\n");

  _k->_logger->logInfo("Normal", "X_best = [ ");
  for (size_t k = 0; k < _variableCount; k++) _k->_logger->logData("Normal", " %.5le  ", _bestEverVariables[k]);
  _k->_logger->logData("Normal", " ]\n");
}

void Rprop::finalize()
{
  // Updating Results
  (*_k)["Results"]["Best Sample"]["F(x)"] = _bestEverValue;
  (*_k)["Results"]["Best Sample"]["Gradient(x)"] = _bestEverGradient;
  (*_k)["Results"]["Best Sample"]["Parameters"] = _bestEverVariables;
  return;
}

void Rprop::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Current Variable"))
 {
 try { _currentVariable = js["Current Variable"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Current Variable']\n%s", e.what()); } 
   eraseValue(js, "Current Variable");
 }

 if (isDefined(js, "Best Ever Variable"))
 {
 try { _bestEverVariable = js["Best Ever Variable"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Best Ever Variable']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Variable");
 }

 if (isDefined(js, "Delta"))
 {
 try { _delta = js["Delta"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Delta']\n%s", e.what()); } 
   eraseValue(js, "Delta");
 }

 if (isDefined(js, "Current Gradient"))
 {
 try { _currentGradient = js["Current Gradient"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Current Gradient']\n%s", e.what()); } 
   eraseValue(js, "Current Gradient");
 }

 if (isDefined(js, "Previous Gradient"))
 {
 try { _previousGradient = js["Previous Gradient"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Previous Gradient']\n%s", e.what()); } 
   eraseValue(js, "Previous Gradient");
 }

 if (isDefined(js, "Best Ever Gradient"))
 {
 try { _bestEverGradient = js["Best Ever Gradient"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Best Ever Gradient']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Gradient");
 }

 if (isDefined(js, "Norm Previous Gradient"))
 {
 try { _normPreviousGradient = js["Norm Previous Gradient"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Norm Previous Gradient']\n%s", e.what()); } 
   eraseValue(js, "Norm Previous Gradient");
 }

 if (isDefined(js, "Max Stall Counter"))
 {
 try { _maxStallCounter = js["Max Stall Counter"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Max Stall Counter']\n%s", e.what()); } 
   eraseValue(js, "Max Stall Counter");
 }

 if (isDefined(js, "X Diff"))
 {
 try { _xDiff = js["X Diff"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['X Diff']\n%s", e.what()); } 
   eraseValue(js, "X Diff");
 }

 if (isDefined(js, "Delta0"))
 {
 try { _delta0 = js["Delta0"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Delta0']\n%s", e.what()); } 
   eraseValue(js, "Delta0");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Delta0'] required by Rprop.\n"); 

 if (isDefined(js, "Delta Min"))
 {
 try { _deltaMin = js["Delta Min"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Delta Min']\n%s", e.what()); } 
   eraseValue(js, "Delta Min");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Delta Min'] required by Rprop.\n"); 

 if (isDefined(js, "Delta Max"))
 {
 try { _deltaMax = js["Delta Max"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Delta Max']\n%s", e.what()); } 
   eraseValue(js, "Delta Max");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Delta Max'] required by Rprop.\n"); 

 if (isDefined(js, "Eta Minus"))
 {
 try { _etaMinus = js["Eta Minus"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Eta Minus']\n%s", e.what()); } 
   eraseValue(js, "Eta Minus");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Eta Minus'] required by Rprop.\n"); 

 if (isDefined(js, "Eta Plus"))
 {
 try { _etaPlus = js["Eta Plus"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Eta Plus']\n%s", e.what()); } 
   eraseValue(js, "Eta Plus");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Eta Plus'] required by Rprop.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Gradient Norm"))
 {
 try { _maxGradientNorm = js["Termination Criteria"]["Max Gradient Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Termination Criteria']['Max Gradient Norm']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Gradient Norm");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Gradient Norm'] required by Rprop.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Stall Generations"))
 {
 try { _maxStallGenerations = js["Termination Criteria"]["Max Stall Generations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Termination Criteria']['Max Stall Generations']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Stall Generations");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Stall Generations'] required by Rprop.\n"); 

 if (isDefined(js, "Termination Criteria", "Parameter Relative Tolerance"))
 {
 try { _parameterRelativeTolerance = js["Termination Criteria"]["Parameter Relative Tolerance"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Rprop ] \n + Key:    ['Termination Criteria']['Parameter Relative Tolerance']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Parameter Relative Tolerance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Parameter Relative Tolerance'] required by Rprop.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/Rprop";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: Rprop: \n%s\n", js.dump(2).c_str());
} 

void Rprop::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Delta0"] = _delta0;
   js["Delta Min"] = _deltaMin;
   js["Delta Max"] = _deltaMax;
   js["Eta Minus"] = _etaMinus;
   js["Eta Plus"] = _etaPlus;
   js["Termination Criteria"]["Max Gradient Norm"] = _maxGradientNorm;
   js["Termination Criteria"]["Max Stall Generations"] = _maxStallGenerations;
   js["Termination Criteria"]["Parameter Relative Tolerance"] = _parameterRelativeTolerance;
   js["Current Variable"] = _currentVariable;
   js["Best Ever Variable"] = _bestEverVariable;
   js["Delta"] = _delta;
   js["Current Gradient"] = _currentGradient;
   js["Previous Gradient"] = _previousGradient;
   js["Best Ever Gradient"] = _bestEverGradient;
   js["Norm Previous Gradient"] = _normPreviousGradient;
   js["Max Stall Counter"] = _maxStallCounter;
   js["X Diff"] = _xDiff;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Optimizer::getConfiguration(js);
} 

void Rprop::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Delta0\": 0.1, \"Delta Min\": 1e-06, \"Delta Max\": 50, \"Eta Minus\": 0.5, \"Eta Plus\": 1.2, \"Termination Criteria\": {\"Max Gradient Norm\": 0.0, \"Max Stall Generations\": 20, \"Parameter Relative Tolerance\": 0.0001}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void Rprop::applyVariableDefaults() 
{

 Optimizer::applyVariableDefaults();
} 

bool Rprop::checkTermination()
{
 bool hasFinished = false;

 if (_normPreviousGradient < _maxGradientNorm)
 {
  _terminationCriteria.push_back("Rprop['Max Gradient Norm'] = " + std::to_string(_maxGradientNorm) + ".");
  hasFinished = true;
 }

 if ( _maxStallCounter >= _maxStallGenerations)
 {
  _terminationCriteria.push_back("Rprop['Max Stall Generations'] = " + std::to_string(_maxStallGenerations) + ".");
  hasFinished = true;
 }

 if (_xDiff<_parameterRelativeTolerance && _xDiff>0)
 {
  _terminationCriteria.push_back("Rprop['Parameter Relative Tolerance'] = " + std::to_string(_parameterRelativeTolerance) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Optimizer::checkTermination();
 return hasFinished;
}

;

} //optimizer
} //solver
} //korali
;
