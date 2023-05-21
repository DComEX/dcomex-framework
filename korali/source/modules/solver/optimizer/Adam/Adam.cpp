#include "engine.hpp"
#include "modules/solver/optimizer/Adam/Adam.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace optimizer
{
;

void Adam::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  for (size_t i = 0; i < _variableCount; i++)
    if (std::isfinite(_k->_variables[i]->_initialValue) == false)
      KORALI_LOG_ERROR("Initial Value of variable \'%s\' not defined (no defaults can be calculated).\n", _k->_variables[i]->_name.c_str());

  _currentVariable.resize(_variableCount);
  for (size_t i = 0; i < _variableCount; i++)
    _currentVariable[i] = _k->_variables[i]->_initialValue;

  _bestEverVariables = _currentVariable;
  _gradient.resize(_variableCount);
  _bestEverGradient.resize(_variableCount, 0);
  _squaredGradient.resize(_variableCount);
  _firstMoment.resize(_variableCount, 0.0);
  _biasCorrectedFirstMoment.resize(_variableCount, 0.0);
  _secondMoment.resize(_variableCount, 0.0);
  _biasCorrectedSecondMoment.resize(_variableCount, 0.0);

  _bestEverValue = -Inf;
  _gradientNorm = 0.0;
}

void Adam::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  // update parameters
  for (size_t i = 0; i < _variableCount; i++)
  {
    _currentVariable[i] += _eta / (std::sqrt(_biasCorrectedSecondMoment[i]) + _epsilon) * _biasCorrectedFirstMoment[i];
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

void Adam::processResult(double evaluation, std::vector<double> &gradient)
{
  _modelEvaluationCount++;
  _previousBestValue = _currentBestValue;
  _currentBestValue = evaluation;

  _gradientNorm = 0.0;

  _gradient = gradient;

  for (size_t i = 0; i < _variableCount; i++)
  {
    _squaredGradient[i] = _gradient[i] * _gradient[i];
    _gradientNorm += _squaredGradient[i];
  }
  _gradientNorm = std::sqrt(_gradientNorm);

  if (_currentBestValue > _bestEverValue)
  {
    _bestEverValue = _currentBestValue;
    _bestEverGradient = _gradient;
    _bestEverVariables = _currentVariable;
  }

  // update first and second moment estimators and bias corrected versions
  for (size_t i = 0; i < _variableCount; i++)
  {
    _firstMoment[i] = _beta1 * _firstMoment[i] + (1 - _beta1) * _gradient[i];
    _biasCorrectedFirstMoment[i] = _firstMoment[i] / (1 - std::pow(_beta1, _modelEvaluationCount));
    _secondMoment[i] = _beta2 * _secondMoment[i] + (1 - _beta2) * _squaredGradient[i];
    _biasCorrectedSecondMoment[i] = _secondMoment[i] / (1 - std::pow(_beta2, _modelEvaluationCount));
  }
}

void Adam::printGenerationBefore()
{
  _k->_logger->logInfo("Normal", "Starting generation %lu...\n", _k->_currentGeneration);
}

void Adam::printGenerationAfter()
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

void Adam::finalize()
{
  // Updating Results
  (*_k)["Results"]["Best Sample"]["F(x)"] = _bestEverValue;
  (*_k)["Results"]["Best Sample"]["Gradient(x)"] = _bestEverGradient;
  (*_k)["Results"]["Best Sample"]["Parameters"] = _bestEverVariables;
}

void Adam::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Current Variable"))
 {
 try { _currentVariable = js["Current Variable"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Current Variable']\n%s", e.what()); } 
   eraseValue(js, "Current Variable");
 }

 if (isDefined(js, "Gradient"))
 {
 try { _gradient = js["Gradient"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Gradient']\n%s", e.what()); } 
   eraseValue(js, "Gradient");
 }

 if (isDefined(js, "Best Ever Gradient"))
 {
 try { _bestEverGradient = js["Best Ever Gradient"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Best Ever Gradient']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Gradient");
 }

 if (isDefined(js, "Squared Gradient"))
 {
 try { _squaredGradient = js["Squared Gradient"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Squared Gradient']\n%s", e.what()); } 
   eraseValue(js, "Squared Gradient");
 }

 if (isDefined(js, "Gradient Norm"))
 {
 try { _gradientNorm = js["Gradient Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Gradient Norm']\n%s", e.what()); } 
   eraseValue(js, "Gradient Norm");
 }

 if (isDefined(js, "First Moment"))
 {
 try { _firstMoment = js["First Moment"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['First Moment']\n%s", e.what()); } 
   eraseValue(js, "First Moment");
 }

 if (isDefined(js, "Bias Corrected First Moment"))
 {
 try { _biasCorrectedFirstMoment = js["Bias Corrected First Moment"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Bias Corrected First Moment']\n%s", e.what()); } 
   eraseValue(js, "Bias Corrected First Moment");
 }

 if (isDefined(js, "Second Moment"))
 {
 try { _secondMoment = js["Second Moment"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Second Moment']\n%s", e.what()); } 
   eraseValue(js, "Second Moment");
 }

 if (isDefined(js, "Bias Corrected Second Moment"))
 {
 try { _biasCorrectedSecondMoment = js["Bias Corrected Second Moment"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Bias Corrected Second Moment']\n%s", e.what()); } 
   eraseValue(js, "Bias Corrected Second Moment");
 }

 if (isDefined(js, "Beta1"))
 {
 try { _beta1 = js["Beta1"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Beta1']\n%s", e.what()); } 
   eraseValue(js, "Beta1");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Beta1'] required by Adam.\n"); 

 if (isDefined(js, "Beta2"))
 {
 try { _beta2 = js["Beta2"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Beta2']\n%s", e.what()); } 
   eraseValue(js, "Beta2");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Beta2'] required by Adam.\n"); 

 if (isDefined(js, "Eta"))
 {
 try { _eta = js["Eta"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Eta']\n%s", e.what()); } 
   eraseValue(js, "Eta");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Eta'] required by Adam.\n"); 

 if (isDefined(js, "Epsilon"))
 {
 try { _epsilon = js["Epsilon"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Epsilon']\n%s", e.what()); } 
   eraseValue(js, "Epsilon");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Epsilon'] required by Adam.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Gradient Norm"))
 {
 try { _minGradientNorm = js["Termination Criteria"]["Min Gradient Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Termination Criteria']['Min Gradient Norm']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Gradient Norm");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Gradient Norm'] required by Adam.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Gradient Norm"))
 {
 try { _maxGradientNorm = js["Termination Criteria"]["Max Gradient Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Adam ] \n + Key:    ['Termination Criteria']['Max Gradient Norm']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Gradient Norm");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Gradient Norm'] required by Adam.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/Adam";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: Adam: \n%s\n", js.dump(2).c_str());
} 

void Adam::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Beta1"] = _beta1;
   js["Beta2"] = _beta2;
   js["Eta"] = _eta;
   js["Epsilon"] = _epsilon;
   js["Termination Criteria"]["Min Gradient Norm"] = _minGradientNorm;
   js["Termination Criteria"]["Max Gradient Norm"] = _maxGradientNorm;
   js["Current Variable"] = _currentVariable;
   js["Gradient"] = _gradient;
   js["Best Ever Gradient"] = _bestEverGradient;
   js["Squared Gradient"] = _squaredGradient;
   js["Gradient Norm"] = _gradientNorm;
   js["First Moment"] = _firstMoment;
   js["Bias Corrected First Moment"] = _biasCorrectedFirstMoment;
   js["Second Moment"] = _secondMoment;
   js["Bias Corrected Second Moment"] = _biasCorrectedSecondMoment;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Optimizer::getConfiguration(js);
} 

void Adam::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Beta1\": 0.9, \"Beta2\": 0.999, \"Eta\": 0.001, \"Epsilon\": 1e-08, \"Termination Criteria\": {\"Min Gradient Norm\": 1e-12, \"Max Gradient Norm\": 1000000000000.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void Adam::applyVariableDefaults() 
{

 Optimizer::applyVariableDefaults();
} 

bool Adam::checkTermination()
{
 bool hasFinished = false;

 if ((_k->_currentGeneration > 1) && (_gradientNorm <= _minGradientNorm))
 {
  _terminationCriteria.push_back("Adam['Min Gradient Norm'] = " + std::to_string(_minGradientNorm) + ".");
  hasFinished = true;
 }

 if ((_k->_currentGeneration > 1) && (_gradientNorm >= _maxGradientNorm))
 {
  _terminationCriteria.push_back("Adam['Max Gradient Norm'] = " + std::to_string(_maxGradientNorm) + ".");
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
