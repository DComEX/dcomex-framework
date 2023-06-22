#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/optimizer/DEA/DEA.hpp"
#include "sample/sample.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdio.h>
#include <unistd.h>

namespace korali
{
namespace solver
{
namespace optimizer
{
;

void DEA::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  for (size_t d = 0; d < _variableCount; ++d)
    if (_k->_variables[d]->_upperBound < _k->_variables[d]->_lowerBound)
      KORALI_LOG_ERROR("Lower Bound (%.4f) of variable \'%s\'  exceeds Upper Bound (%.4f).\n", _k->_variables[d]->_lowerBound, _k->_variables[d]->_name.c_str(), _k->_variables[d]->_upperBound);

  // Allocating Memory
  _samplePopulation.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _samplePopulation[i].resize(_variableCount);

  _candidatePopulation.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _candidatePopulation[i].resize(_variableCount);

  _previousMean.resize(_variableCount);
  _currentMean.resize(_variableCount);
  _bestEverVariables.resize(_variableCount);
  _currentBestVariables.resize(_variableCount);
  _maxDistances.resize(_variableCount);

  _valueVector.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _valueVector[i] = -Inf;

  _valueVector.resize(_populationSize);

  _infeasibleSampleCount = 0;
  _bestSampleIndex = 0;

  _previousBestValue = -Inf;
  _currentBestValue = -Inf;
  _previousBestEverValue = -Inf;
  _bestEverValue = -Inf;
  _currentMinimumStepSize = +Inf;

  initSamples();

  for (size_t d = 0; d < _variableCount; ++d)
  {
    _previousMean[d] = 0.0;
    _currentMean[d] = 0.0;
  }

  for (size_t i = 0; i < _populationSize; ++i)
    for (size_t d = 0; d < _variableCount; ++d)
      _currentMean[d] += _samplePopulation[i][d] / ((double)_populationSize);
}

void DEA::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  prepareGeneration();

  // Initializing Sample Evaluation
  std::vector<Sample> samples(_populationSize);
  for (size_t i = 0; i < _populationSize; i++)
  {
    samples[i]["Module"] = "Problem";
    samples[i]["Operation"] = "Evaluate";
    samples[i]["Parameters"] = _candidatePopulation[i];
    samples[i]["Sample Id"] = i;
    _modelEvaluationCount++;
    KORALI_START(samples[i]);
  }

  // Waiting for samples to finish
  KORALI_WAITALL(samples);

  updateSolver(samples);
}

void DEA::initSamples()
{
  /* skip sampling in gen 1 */
  for (size_t i = 0; i < _populationSize; ++i)
    for (size_t d = 0; d < _variableCount; ++d)
    {
      double width = _k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound;
      _candidatePopulation[i][d] = _k->_variables[d]->_lowerBound + width * _uniformGenerator->getRandomNumber();
      _samplePopulation[i][d] = _candidatePopulation[i][d];
    }
}

void DEA::prepareGeneration()
{
  /* at gen 1 candidates initialized in initialize() */
  if (_k->_currentGeneration > 1)
    for (size_t i = 0; i < _populationSize; ++i)
    {
      bool isFeasible = true;
      do
      {
        mutateSingle(i);
        if (_fixInfeasible && isFeasible == false) fixInfeasible(i);

        isFeasible = isSampleFeasible(_candidatePopulation[i]);
      } while (isFeasible == false);
    }
  _previousValueVector = _valueVector;
}

void DEA::mutateSingle(size_t sampleIdx)
{
  size_t a, b;
  do
  {
    a = _uniformGenerator->getRandomNumber() * _populationSize;
  } while (a == sampleIdx);
  do
  {
    b = _uniformGenerator->getRandomNumber() * _populationSize;
  } while (b == sampleIdx || b == a);

  if (_mutationRule == "Self Adaptive")
  {
    // Brest [2006]
    double tau1 = 0.1;
    double tau2 = 0.1;
    double Fl = 0.1;
    double Fu = 0.9;

    double rd2 = _uniformGenerator->getRandomNumber();
    double rd3 = _uniformGenerator->getRandomNumber();

    if (rd2 < tau1)
    {
      double rd1 = _uniformGenerator->getRandomNumber();
      _mutationRate = Fl + rd1 * Fu;
    }
    if (rd3 < tau2)
    {
      double rd4 = _uniformGenerator->getRandomNumber();
      _crossoverRate = rd4;
    }
  }

  double *parent;
  if (_parentSelectionRule == "Random")
  {
    size_t c;
    do
    {
      c = _uniformGenerator->getRandomNumber() * _populationSize;
    } while (c == sampleIdx || c == a || c == b);
    parent = &_samplePopulation[c][0];
  }
  else /* _parentSelectionRule == "Best" */
  {
    parent = &_samplePopulation[_bestSampleIndex][0];
  }

  size_t rn = _uniformGenerator->getRandomNumber() * _variableCount;
  for (size_t d = 0; d < _variableCount; ++d)
  {
    if ((_uniformGenerator->getRandomNumber() < _crossoverRate) || (d == rn))
      _candidatePopulation[sampleIdx][d] = parent[d] + _mutationRate * (_samplePopulation[a][d] - _samplePopulation[b][d]);
    else
      _candidatePopulation[sampleIdx][d] = _samplePopulation[sampleIdx][d];
  }
}

void DEA::fixInfeasible(size_t sampleIdx)
{
  for (size_t d = 0; d < _variableCount; ++d)
  {
    double len = 0.0;
    if (_candidatePopulation[sampleIdx][d] < _k->_variables[d]->_lowerBound)
      len = _candidatePopulation[sampleIdx][d] - _k->_variables[d]->_lowerBound;
    if (_candidatePopulation[sampleIdx][d] > _k->_variables[d]->_upperBound)
      len = _candidatePopulation[sampleIdx][d] - _k->_variables[d]->_upperBound;

    _candidatePopulation[sampleIdx][d] = _samplePopulation[sampleIdx][d] - len * _uniformGenerator->getRandomNumber();
  }
}

void DEA::updateSolver(std::vector<Sample> &samples)
{
  // Processing results
  for (size_t i = 0; i < _populationSize; i++)
    _valueVector[i] = KORALI_GET(double, samples[i], "F(x)");

  _bestSampleIndex = std::distance(std::begin(_valueVector), std::max_element(std::begin(_valueVector), std::end(_valueVector)));
  _previousBestEverValue = _bestEverValue;
  _previousBestValue = _currentBestValue;
  _currentBestValue = _valueVector[_bestSampleIndex];

  for (size_t d = 0; d < _variableCount; ++d) _currentBestVariables[d] = _candidatePopulation[_bestSampleIndex][d];

  _previousMean = _currentMean;
  std::fill(std::begin(_currentMean), std::end(_currentMean), 0.0);

  if (_currentBestValue > _bestEverValue) _bestEverVariables = _currentBestVariables;

  bool acceptRuleRecognized = false;

  if (_acceptRule == "Best") // only update best sample
  {
    if (_currentBestValue > _bestEverValue)
    {
      for (size_t d = 0; d < _variableCount; ++d) _samplePopulation[_bestSampleIndex][d] = _candidatePopulation[_bestSampleIndex][d];
      _bestEverValue = _currentBestValue;
    }
    acceptRuleRecognized = true;
  }

  if (_acceptRule == "Greedy") // accept all mutations better than parent
  {
    for (size_t i = 0; i < _populationSize; ++i)
      if (_valueVector[i] > _previousValueVector[i])
        _samplePopulation[i] = _candidatePopulation[i];
    if (_currentBestValue > _bestEverValue)
    {
      _bestEverValue = _currentBestValue;
    }
    acceptRuleRecognized = true;
  }

  if (_acceptRule == "Improved") // update all samples better than _bestEverValue
  {
    for (size_t i = 0; i < _populationSize; ++i)
      if (_valueVector[i] > _bestEverValue)
        for (size_t d = 0; d < _variableCount; ++d) _samplePopulation[i][d] = _candidatePopulation[i][d];
    if (_currentBestValue > _bestEverValue)
    {
      _bestEverValue = _currentBestValue;
    }
    acceptRuleRecognized = true;
  }

  if (_acceptRule == "Iterative") // iteratively update _bestEverValue and accept samples
  {
    for (size_t i = 0; i < _populationSize; ++i)
      if (_valueVector[i] > _bestEverValue)
        for (size_t d = 0; d < _variableCount; ++d)
        {
          _samplePopulation[i][d] = _candidatePopulation[i][d];
          _bestEverValue = _valueVector[i];
        }
    acceptRuleRecognized = true;
  }

  if (acceptRuleRecognized == false) KORALI_LOG_ERROR("Accept Rule (%s) not recognized.\n", _acceptRule.c_str());

  for (size_t i = 0; i < _populationSize; ++i)
    for (size_t d = 0; d < _variableCount; ++d)
      _currentMean[d] += _samplePopulation[i][d] / ((double)_populationSize);

  for (size_t d = 0; d < _variableCount; ++d)
  {
    double max = -Inf;
    double min = +Inf;
    for (size_t i = 0; i < _populationSize; ++i)
    {
      if (_samplePopulation[i][d] > max) max = _samplePopulation[i][d];
      if (_samplePopulation[i][d] < min) min = _samplePopulation[i][d];
    }
    _maxDistances[d] = max - min;
  }

  _currentMinimumStepSize = +Inf;
  for (size_t d = 0; d < _variableCount; ++d) std::min(_currentMinimumStepSize, fabs(_currentMean[d] - _previousMean[d]));
}

/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/

void DEA::printGenerationBefore() { return; }

void DEA::printGenerationAfter()
{
  _k->_logger->logInfo("Normal", "Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
  _k->_logger->logInfo("Detailed", "Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _variableCount; d++) _k->_logger->logData("Detailed", "         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _currentMean[d], _bestEverVariables[d]);
  _k->_logger->logInfo("Detailed", "Max Width:\n");
  for (size_t d = 0; d < _variableCount; d++) _k->_logger->logData("Detailed", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _maxDistances[d]);
  _k->_logger->logInfo("Detailed", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
}

void DEA::finalize()
{
  // Updating Results
  (*_k)["Results"]["Best Sample"]["F(x)"] = _bestEverValue;
  (*_k)["Results"]["Best Sample"]["Parameters"] = _bestEverVariables;

  _k->_logger->logInfo("Minimal", "Optimum found: %e\n", _bestEverValue);
  _k->_logger->logInfo("Minimal", "Optimum found at:\n");
  for (size_t d = 0; d < _variableCount; ++d) _k->_logger->logData("Minimal", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverVariables[d]);
  _k->_logger->logInfo("Minimal", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
}

void DEA::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Normal Generator"))
 {
 _normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(js["Normal Generator"], _k));
 _normalGenerator->applyVariableDefaults();
 _normalGenerator->applyModuleDefaults(js["Normal Generator"]);
 _normalGenerator->setConfiguration(js["Normal Generator"]);
   eraseValue(js, "Normal Generator");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Value Vector"))
 {
 try { _valueVector = js["Value Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Value Vector']\n%s", e.what()); } 
   eraseValue(js, "Value Vector");
 }

 if (isDefined(js, "Previous Value Vector"))
 {
 try { _previousValueVector = js["Previous Value Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Previous Value Vector']\n%s", e.what()); } 
   eraseValue(js, "Previous Value Vector");
 }

 if (isDefined(js, "Sample Population"))
 {
 try { _samplePopulation = js["Sample Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Sample Population']\n%s", e.what()); } 
   eraseValue(js, "Sample Population");
 }

 if (isDefined(js, "Candidate Population"))
 {
 try { _candidatePopulation = js["Candidate Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Candidate Population']\n%s", e.what()); } 
   eraseValue(js, "Candidate Population");
 }

 if (isDefined(js, "Best Sample Index"))
 {
 try { _bestSampleIndex = js["Best Sample Index"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Best Sample Index']\n%s", e.what()); } 
   eraseValue(js, "Best Sample Index");
 }

 if (isDefined(js, "Previous Best Ever Value"))
 {
 try { _previousBestEverValue = js["Previous Best Ever Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Previous Best Ever Value']\n%s", e.what()); } 
   eraseValue(js, "Previous Best Ever Value");
 }

 if (isDefined(js, "Current Mean"))
 {
 try { _currentMean = js["Current Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Current Mean']\n%s", e.what()); } 
   eraseValue(js, "Current Mean");
 }

 if (isDefined(js, "Previous Mean"))
 {
 try { _previousMean = js["Previous Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Previous Mean']\n%s", e.what()); } 
   eraseValue(js, "Previous Mean");
 }

 if (isDefined(js, "Current Best Variables"))
 {
 try { _currentBestVariables = js["Current Best Variables"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Current Best Variables']\n%s", e.what()); } 
   eraseValue(js, "Current Best Variables");
 }

 if (isDefined(js, "Max Distances"))
 {
 try { _maxDistances = js["Max Distances"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Max Distances']\n%s", e.what()); } 
   eraseValue(js, "Max Distances");
 }

 if (isDefined(js, "Current Minimum Step Size"))
 {
 try { _currentMinimumStepSize = js["Current Minimum Step Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Current Minimum Step Size']\n%s", e.what()); } 
   eraseValue(js, "Current Minimum Step Size");
 }

 if (isDefined(js, "Population Size"))
 {
 try { _populationSize = js["Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Population Size']\n%s", e.what()); } 
   eraseValue(js, "Population Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Population Size'] required by DEA.\n"); 

 if (isDefined(js, "Crossover Rate"))
 {
 try { _crossoverRate = js["Crossover Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Crossover Rate']\n%s", e.what()); } 
   eraseValue(js, "Crossover Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Crossover Rate'] required by DEA.\n"); 

 if (isDefined(js, "Mutation Rate"))
 {
 try { _mutationRate = js["Mutation Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Mutation Rate']\n%s", e.what()); } 
   eraseValue(js, "Mutation Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mutation Rate'] required by DEA.\n"); 

 if (isDefined(js, "Mutation Rule"))
 {
 try { _mutationRule = js["Mutation Rule"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Mutation Rule']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_mutationRule == "Fixed") validOption = true; 
 if (_mutationRule == "Self Adaptive") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mutation Rule'] required by DEA.\n", _mutationRule.c_str()); 
}
   eraseValue(js, "Mutation Rule");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mutation Rule'] required by DEA.\n"); 

 if (isDefined(js, "Parent Selection Rule"))
 {
 try { _parentSelectionRule = js["Parent Selection Rule"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Parent Selection Rule']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_parentSelectionRule == "Random") validOption = true; 
 if (_parentSelectionRule == "Best") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Parent Selection Rule'] required by DEA.\n", _parentSelectionRule.c_str()); 
}
   eraseValue(js, "Parent Selection Rule");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Parent Selection Rule'] required by DEA.\n"); 

 if (isDefined(js, "Accept Rule"))
 {
 try { _acceptRule = js["Accept Rule"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Accept Rule']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_acceptRule == "Best") validOption = true; 
 if (_acceptRule == "Greedy") validOption = true; 
 if (_acceptRule == "Iterative") validOption = true; 
 if (_acceptRule == "Improved") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Accept Rule'] required by DEA.\n", _acceptRule.c_str()); 
}
   eraseValue(js, "Accept Rule");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Accept Rule'] required by DEA.\n"); 

 if (isDefined(js, "Fix Infeasible"))
 {
 try { _fixInfeasible = js["Fix Infeasible"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Fix Infeasible']\n%s", e.what()); } 
   eraseValue(js, "Fix Infeasible");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Fix Infeasible'] required by DEA.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Value"))
 {
 try { _minValue = js["Termination Criteria"]["Min Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Termination Criteria']['Min Value']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Value'] required by DEA.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Step Size"))
 {
 try { _minStepSize = js["Termination Criteria"]["Min Step Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ DEA ] \n + Key:    ['Termination Criteria']['Min Step Size']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Step Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Step Size'] required by DEA.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/DEA";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: DEA: \n%s\n", js.dump(2).c_str());
} 

void DEA::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Population Size"] = _populationSize;
   js["Crossover Rate"] = _crossoverRate;
   js["Mutation Rate"] = _mutationRate;
   js["Mutation Rule"] = _mutationRule;
   js["Parent Selection Rule"] = _parentSelectionRule;
   js["Accept Rule"] = _acceptRule;
   js["Fix Infeasible"] = _fixInfeasible;
   js["Termination Criteria"]["Min Value"] = _minValue;
   js["Termination Criteria"]["Min Step Size"] = _minStepSize;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Value Vector"] = _valueVector;
   js["Previous Value Vector"] = _previousValueVector;
   js["Sample Population"] = _samplePopulation;
   js["Candidate Population"] = _candidatePopulation;
   js["Best Sample Index"] = _bestSampleIndex;
   js["Previous Best Ever Value"] = _previousBestEverValue;
   js["Current Mean"] = _currentMean;
   js["Previous Mean"] = _previousMean;
   js["Current Best Variables"] = _currentBestVariables;
   js["Max Distances"] = _maxDistances;
   js["Current Minimum Step Size"] = _currentMinimumStepSize;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Optimizer::getConfiguration(js);
} 

void DEA::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Population Size\": 200, \"Crossover Rate\": 0.9, \"Mutation Rate\": 0.5, \"Mutation Rule\": \"Fixed\", \"Parent Selection Rule\": \"Random\", \"Accept Rule\": \"Greedy\", \"Fix Infeasible\": true, \"Termination Criteria\": {\"Min Value\": -Infinity, \"Max Value\": Infinity, \"Min Step Size\": -Infinity}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Value Vector\": [], \"Previous Value Vector\": [], \"Sample Population\": [[]], \"Candidate Population\": [[]], \"Best Sample Index\": 0, \"Best Ever Value\": -Infinity, \"Previous Best Ever Value\": -Infinity, \"Current Mean\": [], \"Previous Mean\": [], \"Current Best Variables\": [], \"Max Distances\": [], \"Current Minimum Step Size\": 0.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void DEA::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Optimizer::applyVariableDefaults();
} 

bool DEA::checkTermination()
{
 bool hasFinished = false;

 if ((_k->_currentGeneration > 1) && (-_bestEverValue < _minValue))
 {
  _terminationCriteria.push_back("DEA['Min Value'] = " + std::to_string(_minValue) + ".");
  hasFinished = true;
 }

 if (_currentMinimumStepSize < _minStepSize)
 {
  _terminationCriteria.push_back("DEA['Min Step Size'] = " + std::to_string(_minStepSize) + ".");
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
