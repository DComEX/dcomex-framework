#include "engine.hpp"
#include "modules/solver/executor/executor.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
;

void Executor::runGeneration()
{
  _variableCount = _k->_variables.size();
  _sampleCount = std::max(_k->_variables[0]->_precomputedValues.size(), _k->_variables[0]->_sampledValues.size());

  _maxModelEvaluations = std::min(_maxModelEvaluations, _sampleCount);
  _executionsPerGeneration = std::min(_executionsPerGeneration, _maxModelEvaluations - _modelEvaluationCount);

  std::vector<Sample> samples(_executionsPerGeneration);
  std::vector<double> sampleData(_variableCount);

  for (size_t i = 0; i < _executionsPerGeneration; i++)
  {
    for (size_t d = 0; d < _variableCount; d++)
    {
      if (_k->_variables[0]->_precomputedValues.size() > 0)
        sampleData[d] = _k->_variables[d]->_precomputedValues[_modelEvaluationCount];
      else
        sampleData[d] = _k->_distributions[_k->_variables[d]->_distributionIndex]->getRandomNumber();
    }

    _k->_logger->logInfo("Detailed", "Running sample %zu with values:\n         ", _modelEvaluationCount);
    for (auto &x : sampleData) _k->_logger->logData("Detailed", " %le   ", x);
    _k->_logger->logData("Detailed", "\n");

    samples[i]["Module"] = "Problem";
    samples[i]["Operation"] = "Execute";
    samples[i]["Parameters"] = sampleData;
    samples[i]["Sample Id"] = _modelEvaluationCount;
    KORALI_START(samples[i]);
    _modelEvaluationCount++;
  }

  KORALI_WAITALL(samples);
}

void Executor::printGenerationBefore()
{
}

void Executor::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Total Executions %lu.\n", _modelEvaluationCount);
}

void Executor::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Sample Count"))
 {
 try { _sampleCount = js["Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ executor ] \n + Key:    ['Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Sample Count");
 }

 if (isDefined(js, "Executions Per Generation"))
 {
 try { _executionsPerGeneration = js["Executions Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ executor ] \n + Key:    ['Executions Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Executions Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Executions Per Generation'] required by executor.\n"); 

 Solver::setConfiguration(js);
 _type = "executor";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: executor: \n%s\n", js.dump(2).c_str());
} 

void Executor::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Executions Per Generation"] = _executionsPerGeneration;
   js["Sample Count"] = _sampleCount;
 Solver::getConfiguration(js);
} 

void Executor::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Executions Per Generation\": 500000000}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Executor::applyVariableDefaults() 
{

 Solver::applyVariableDefaults();
} 

;

} //solver
} //korali
;