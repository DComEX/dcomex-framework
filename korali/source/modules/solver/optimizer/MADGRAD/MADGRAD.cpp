#include "engine.hpp"
#include "modules/solver/optimizer/MADGRAD/MADGRAD.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace optimizer
{
;

void MADGRAD::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  for (size_t i = 0; i < _variableCount; i++)
    if (std::isfinite(_k->_variables[i]->_initialValue) == false)
      KORALI_LOG_ERROR("Initial Value of variable \'%s\' not defined (no defaults can be calculated).\n", _k->_variables[i]->_name.c_str());

  _currentVariable.resize(_variableCount);
  for (size_t i = 0; i < _variableCount; i++)
    _currentVariable[i] = _k->_variables[i]->_initialValue;

  _bestEverVariables = _currentVariable;
  _initialParameter = _currentVariable;
  _gradient.resize(_variableCount, 0.0);
  _bestEverGradient.resize(_variableCount, 0);
  _gradientSum.resize(_variableCount, 0.0);
  _squaredGradientSum.resize(_variableCount, 0.0);

  _bestEverValue = -Inf;
  _gradientNorm = 0.0;
  _scaledLearningRate = _eta;

  if (_eta <= 0) KORALI_LOG_ERROR("Learning Rate 'eta' must be larger 0 (is %lf).\n", _eta);
  if (_weightDecay <= 0) KORALI_LOG_ERROR("Weight decaymust be larger 0 (is %lf).\n", _weightDecay);
  if (_epsilon <= 0) KORALI_LOG_ERROR("Epsilon must be larger 0 (is %lf).\n", _epsilon);
}

void MADGRAD::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  // update parameters
  for (size_t i = 0; i < _variableCount; i++)
  {
    double intermediateParam = _initialParameter[i] + 1.0 / (std::cbrt(_squaredGradientSum[i]) + _epsilon) * _gradientSum[i];
    _currentVariable[i] = (1.0 - _weightDecay) * _currentVariable[i] + _weightDecay * intermediateParam;
  }

  // Initializing Sample Evaluation
  Sample sample;
  sample["Module"] = "Problem";
  sample["Operation"] = "Evaluate With Gradients";
  sample["Parameters"] = _currentVariable;
  sample["Sample Id"] = 0;
  KORALI_START(sample);

  // Waiting for sample to finish
  KORALI_WAIT(sample);

  auto evaluation = KORALI_GET(double, sample, "F(x)");
  auto gradient = KORALI_GET(std::vector<double>, sample, "Gradient");

  // Processing results
  processResult(evaluation, gradient);
}

void MADGRAD::processResult(double evaluation, std::vector<double> &gradient)
{
  _modelEvaluationCount++;
  _previousBestValue = _currentBestValue;
  _currentBestValue = evaluation;

  _gradientNorm = 0.0;
  _gradient = gradient;

  //_scaledLearningRate = std::sqrt(_modelEvaluationCount+1.0) * _eta;

  for (size_t i = 0; i < _variableCount; i++)
  {
    _gradientSum[i] += _scaledLearningRate * _gradient[i];
    _squaredGradientSum[i] += _scaledLearningRate * _gradient[i] * _gradient[i];
    _gradientNorm += _gradient[i] * _gradient[i];
  }
  _gradientNorm = std::sqrt(_gradientNorm);

  if (_currentBestValue > _bestEverValue)
  {
    _bestEverValue = _currentBestValue;
    _bestEverGradient = _gradient;
    _bestEverVariables = _currentVariable;
  }
}

void MADGRAD::printGenerationBefore()
{
  _k->_logger->logInfo("Normal", "Starting generation %lu...\n", _k->_currentGeneration);
}

void MADGRAD::printGenerationAfter()
{
  _k->_logger->logInfo("Normal", "x = [ ");
  for (size_t k = 0; k < _variableCount; k++) _k->_logger->logData("Normal", " %.5le  ", _currentVariable[k]);
  _k->_logger->logData("Normal", " ]\n");

  _k->_logger->logInfo("Normal", "F(X) = %le \n", _currentBestValue);

  _k->_logger->logInfo("Normal", "DF(X) = [ ");
  for (size_t k = 0; k < _variableCount; k++) _k->_logger->logData("Normal", " %.5le  ", _gradient[k]);
  _k->_logger->logData("Normal", " ]\n");

  _k->_logger->logInfo("Normal", "|DF(X)| = %le \n", _gradientNorm);
}

void MADGRAD::finalize()
{
  // Updating Results
  (*_k)["Results"]["Best Sample"]["F(x)"] = _bestEverValue;
  (*_k)["Results"]["Best Sample"]["Gradient(x)"] = _bestEverGradient;
  (*_k)["Results"]["Best Sample"]["Parameters"] = _bestEverVariables;
}

void MADGRAD::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Current Variable"))
 {
 try { _currentVariable = js["Current Variable"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Current Variable']\n%s", e.what()); } 
   eraseValue(js, "Current Variable");
 }

 if (isDefined(js, "Scaled Learning Rate"))
 {
 try { _scaledLearningRate = js["Scaled Learning Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Scaled Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Scaled Learning Rate");
 }

 if (isDefined(js, "Initial Parameter"))
 {
 try { _initialParameter = js["Initial Parameter"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Initial Parameter']\n%s", e.what()); } 
   eraseValue(js, "Initial Parameter");
 }

 if (isDefined(js, "Gradient"))
 {
 try { _gradient = js["Gradient"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Gradient']\n%s", e.what()); } 
   eraseValue(js, "Gradient");
 }

 if (isDefined(js, "Best Ever Gradient"))
 {
 try { _bestEverGradient = js["Best Ever Gradient"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Best Ever Gradient']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Gradient");
 }

 if (isDefined(js, "Gradient Norm"))
 {
 try { _gradientNorm = js["Gradient Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Gradient Norm']\n%s", e.what()); } 
   eraseValue(js, "Gradient Norm");
 }

 if (isDefined(js, "Gradient Sum"))
 {
 try { _gradientSum = js["Gradient Sum"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Gradient Sum']\n%s", e.what()); } 
   eraseValue(js, "Gradient Sum");
 }

 if (isDefined(js, "Squared Gradient Sum"))
 {
 try { _squaredGradientSum = js["Squared Gradient Sum"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Squared Gradient Sum']\n%s", e.what()); } 
   eraseValue(js, "Squared Gradient Sum");
 }

 if (isDefined(js, "Eta"))
 {
 try { _eta = js["Eta"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Eta']\n%s", e.what()); } 
   eraseValue(js, "Eta");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Eta'] required by MADGRAD.\n"); 

 if (isDefined(js, "Weight Decay"))
 {
 try { _weightDecay = js["Weight Decay"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Weight Decay']\n%s", e.what()); } 
   eraseValue(js, "Weight Decay");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Weight Decay'] required by MADGRAD.\n"); 

 if (isDefined(js, "Epsilon"))
 {
 try { _epsilon = js["Epsilon"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Epsilon']\n%s", e.what()); } 
   eraseValue(js, "Epsilon");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Epsilon'] required by MADGRAD.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Gradient Norm"))
 {
 try { _minGradientNorm = js["Termination Criteria"]["Min Gradient Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Termination Criteria']['Min Gradient Norm']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Gradient Norm");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Gradient Norm'] required by MADGRAD.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Gradient Norm"))
 {
 try { _maxGradientNorm = js["Termination Criteria"]["Max Gradient Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MADGRAD ] \n + Key:    ['Termination Criteria']['Max Gradient Norm']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Gradient Norm");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Gradient Norm'] required by MADGRAD.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/MADGRAD";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: MADGRAD: \n%s\n", js.dump(2).c_str());
} 

void MADGRAD::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Eta"] = _eta;
   js["Weight Decay"] = _weightDecay;
   js["Epsilon"] = _epsilon;
   js["Termination Criteria"]["Min Gradient Norm"] = _minGradientNorm;
   js["Termination Criteria"]["Max Gradient Norm"] = _maxGradientNorm;
   js["Current Variable"] = _currentVariable;
   js["Scaled Learning Rate"] = _scaledLearningRate;
   js["Initial Parameter"] = _initialParameter;
   js["Gradient"] = _gradient;
   js["Best Ever Gradient"] = _bestEverGradient;
   js["Gradient Norm"] = _gradientNorm;
   js["Gradient Sum"] = _gradientSum;
   js["Squared Gradient Sum"] = _squaredGradientSum;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Optimizer::getConfiguration(js);
} 

void MADGRAD::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Eta\": 0.01, \"Weight Decay\": 0.9, \"Epsilon\": 1e-06, \"Termination Criteria\": {\"Min Gradient Norm\": 1e-12, \"Max Gradient Norm\": 1000000000000.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void MADGRAD::applyVariableDefaults() 
{

 Optimizer::applyVariableDefaults();
} 

bool MADGRAD::checkTermination()
{
 bool hasFinished = false;

 if ((_k->_currentGeneration > 1) && (_gradientNorm <= _minGradientNorm))
 {
  _terminationCriteria.push_back("MADGRAD['Min Gradient Norm'] = " + std::to_string(_minGradientNorm) + ".");
  hasFinished = true;
 }

 if ((_k->_currentGeneration > 1) && (_gradientNorm >= _maxGradientNorm))
 {
  _terminationCriteria.push_back("MADGRAD['Max Gradient Norm'] = " + std::to_string(_maxGradientNorm) + ".");
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
