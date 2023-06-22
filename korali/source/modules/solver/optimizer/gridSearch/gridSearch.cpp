#include "engine.hpp"
#include "modules/solver/optimizer/gridSearch/gridSearch.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace optimizer
{
;

void GridSearch::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  _numberOfValues = 1;
  for (size_t i = 0; i < _variableCount; i++)
    _numberOfValues *= _k->_variables[i]->_values.size();

  if (_numberOfValues > _maxModelEvaluations)
  {
    _k->_logger->logWarning("Normal", "%lu > %lu. More evaluations required than the maximum specified.\n", _numberOfValues, _maxModelEvaluations);
    _numberOfValues = _maxModelEvaluations;
  }

  // Resetting execution counter
  _modelEvaluationCount = 0;

  _maxModelEvaluations = _numberOfValues;

  _objective.resize(_numberOfValues);
  _bestEverVariables.resize(_variableCount);

  // We assume i = _index[0] + _index[1]*_sample[0].size() + _index[1]*_sample[0].size()*_sample[1].size() + .....
  _indexHelper.resize(_variableCount);
  for (size_t i = 0; i < _variableCount; i++)
  {
    if (i == 0 || i == 1)
      _indexHelper[i] = _k->_variables[0]->_values.size();
    else
      _indexHelper[i] = _indexHelper[i - 1] * _k->_variables[i - 1]->_values.size();
  }
}

void GridSearch::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  // Create Sample
  std::vector<Sample> samples(_numberOfValues);
  std::vector<double> sampleData(_variableCount);

  size_t rest, index;
  for (size_t i = 0; i < _numberOfValues; i++)
  {
    rest = i;
    for (int d = _variableCount - 1; d >= 0; d--)
    {
      // We assume i = _index[0] + _index[1]*_sample[0].size() + _index[1]*_sample[0].size()*_sample[1].size() + .....
      if (d == 0)
        index = rest % _indexHelper[d];
      else
        index = rest / _indexHelper[d];

      rest -= index * _indexHelper[d];

      sampleData[d] = _k->_variables[d]->_values[index];
    }
    _k->_logger->logInfo("Detailed", "Running sample %zu/%zu with values:\n         ", i + 1, _numberOfValues);
    for (auto &x : sampleData) _k->_logger->logData("Detailed", " %f   ", x);
    _k->_logger->logData("Detailed", "\n");

    samples[i]["Module"] = "Problem";
    samples[i]["Operation"] = "Evaluate";
    samples[i]["Parameters"] = sampleData;
    samples[i]["Sample Id"] = i;
    KORALI_START(samples[i]);
    _modelEvaluationCount++;
  }
  KORALI_WAITALL(samples);
  for (size_t i = 0; i < _numberOfValues; i++)
  {
    _objective[i] = KORALI_GET(double, samples[i], "F(x)");
  }

  std::vector<double>::iterator maximum = std::max_element(_objective.begin(), _objective.end());
  size_t maxIndex = std::distance(_objective.begin(), maximum);

  _bestEverVariables = KORALI_GET(std::vector<double>, samples[maxIndex], "Parameters");
  _bestEverValue = KORALI_GET(double, samples[maxIndex], "F(x)");
}

void GridSearch::printGenerationBefore()
{
}

void GridSearch::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Found Maximum with Objective %+6.3e at:\n", _bestEverValue);
  for (size_t i = 0; i < _bestEverVariables.size(); i++)
    _k->_logger->logData("Normal", " %+6.3e", _bestEverVariables[i]);
  _k->_logger->logData("Normal", "\n");
}

void GridSearch::finalize()
{
  // Updating Results
  (*_k)["Results"]["Best Sample"]["Parameters"] = _bestEverVariables;
  (*_k)["Results"]["Best Sample"]["F(x)"] = _bestEverValue;
}

void GridSearch::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Number Of Values"))
 {
 try { _numberOfValues = js["Number Of Values"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gridSearch ] \n + Key:    ['Number Of Values']\n%s", e.what()); } 
   eraseValue(js, "Number Of Values");
 }

 if (isDefined(js, "Objective"))
 {
 try { _objective = js["Objective"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gridSearch ] \n + Key:    ['Objective']\n%s", e.what()); } 
   eraseValue(js, "Objective");
 }

 if (isDefined(js, "Index Helper"))
 {
 try { _indexHelper = js["Index Helper"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gridSearch ] \n + Key:    ['Index Helper']\n%s", e.what()); } 
   eraseValue(js, "Index Helper");
 }

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/gridSearch";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: gridSearch: \n%s\n", js.dump(2).c_str());
} 

void GridSearch::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Number Of Values"] = _numberOfValues;
   js["Objective"] = _objective;
   js["Index Helper"] = _indexHelper;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Optimizer::getConfiguration(js);
} 

void GridSearch::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void GridSearch::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Optimizer::applyVariableDefaults();
} 

bool GridSearch::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Optimizer::checkTermination();
 return hasFinished;
}

;

} //optimizer
} //solver
} //korali
;
