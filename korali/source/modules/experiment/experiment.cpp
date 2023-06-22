#include "auxiliar/fs.hpp"
#include "auxiliar/koraliJson.hpp"
#include "engine.hpp"
#include "modules/conduit/conduit.hpp"
#include "modules/conduit/distributed/distributed.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/agent/agent.hpp"
#include "modules/solver/deepSupervisor/deepSupervisor.hpp"
#include "modules/solver/solver.hpp"
#include "sample/sample.hpp"
#include <chrono>
#include <cstdio>
#include <map>
#include <stdlib.h>

namespace korali
{
;
/**
 * @brief Pointer to the current experiment in execution
 */
Experiment *__expPointer;

/**
 * @brief Pointer to the calling thread
 */
cothread_t __returnThread;

/**
 * @brief Function for the initialization of new coroutine threads.
 */
void threadWrapper()
{
  auto e = __expPointer;
  e->run();

  co_switch(e->_engine->_thread);
  KORALI_LOG_ERROR("Trying to continue finished Experiment thread.\n");
}

void Experiment::run()
{
  co_switch(__returnThread);

  auto t0 = std::chrono::system_clock::now();

  // Saving initial configuration
  if (_currentGeneration == 0)
    if (_fileOutputEnabled)
    {
      _timestamp = getTimestamp();
      getConfiguration(_js.getJson());
      saveState();
    }

  _currentGeneration++;

  _solver->_terminationCriteria.clear();
  while (_solver->checkTermination() == false)
  {
    if (_consoleOutputFrequency > 0)
      if (_currentGeneration % _consoleOutputFrequency == 0)
      {
        _logger->logInfo("Minimal", "--------------------------------------------------------------------\n");
        _logger->logInfo("Minimal", "Current Generation: #%zu\n", _currentGeneration);
        _solver->printGenerationBefore();
      }

    // Cleaning sample information from previous generation
    _js["Samples"] = knlohmann::json();

    // Timing and Profiling Start
    auto t0 = std::chrono::system_clock::now();
    _solver->runGeneration();

    // Timing and Profiling End
    auto t1 = std::chrono::system_clock::now();

    // Printing results to console
    if (_consoleOutputFrequency > 0)
      if (_currentGeneration % _consoleOutputFrequency == 0)
      {
        _solver->printGenerationAfter();
        _logger->logInfo("Detailed", "Experiment: %lu - Generation Time: %.3fs\n", _experimentId, std::chrono::duration<double>(t1 - t0).count());
      }

    // Saving state to a file
    if (_fileOutputEnabled)
      if (_fileOutputFrequency > 0)
        if (_currentGeneration % _fileOutputFrequency == 0)
        {
          _timestamp = getTimestamp();
          getConfiguration(_js.getJson());
          saveState();
        }

    _currentGeneration++;

    // Check for error signals from python
    if (isPythonActive && PyErr_CheckSignals() != 0) KORALI_LOG_ERROR("User requested break.\n");
  }

  auto t1 = std::chrono::system_clock::now();

  // Finalizing experiment
  _currentGeneration--;
  _isFinished = true;
  _solver->finalize();

  // Saving last generation and final results
  _timestamp = getTimestamp();
  getConfiguration(_js.getJson());
  if (_fileOutputEnabled) saveState();

  _logger->logInfo("Minimal", "--------------------------------------------------------------------\n");
  _logger->logInfo("Minimal", "%s finished correctly.\n", _solver->getType().c_str());
  for (size_t i = 0; i < _solver->_terminationCriteria.size(); i++) _logger->logInfo("Normal", "Termination Criterion Met: %s\n", _solver->_terminationCriteria[i].c_str());
  _logger->logInfo("Normal", "Final Generation: %lu\n", _currentGeneration);
  _logger->logInfo("Normal", "Elapsed Time: %.3fs\n", std::chrono::duration<double>(t1 - t0).count());
}

void Experiment::saveState()
{
  auto beginTime = std::chrono::steady_clock::now();

  if (_storeSampleInformation == true) _js["Samples"] = _sampleInfo["Samples"];

  char genFileName[256];

  // Naming result files depends on whether incremental numbering is used, or we overwrite previous results
  if (_fileOutputUseMultipleFiles == true)
    sprintf(genFileName, "gen%08lu.json", _currentGeneration);
  else
    sprintf(genFileName, "genLatest.json");

  // If results directory doesn't exist, create it
  if (!dirExists(_fileOutputPath)) mkdir(_fileOutputPath);

  std::string filePath = "./" + _fileOutputPath + "/" + genFileName;

  if (saveJsonToFile(filePath.c_str(), _js.getJson()) != 0) KORALI_LOG_ERROR("Error trying to save result file: %s.\n", filePath.c_str());

  // If using multiple files, create a hard link to the latest result
  std::string linkPath = "./" + _fileOutputPath + "/latest";
  remove(linkPath.c_str());
  link(filePath.c_str(), linkPath.c_str());

  auto endTime = std::chrono::steady_clock::now();
  _resultSavingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();
}

bool Experiment::loadState(const std::string &path)
{
  return loadJsonFromFile(_js.getJson(), path.c_str());
}

Experiment::Experiment()
{
  _runID = getTimehash();
  _k = this;
  _logger = NULL;
  _engine = NULL;
  _currentGeneration = 0;
  _isInitialized = false;
}

void Experiment::initialize()
{
  // Initializing profiling timers
  _resultSavingTime = 0.0;

  __expPointer = this;
  _thread = co_create(1 << 20, threadWrapper);
  __returnThread = co_active();
  co_switch(_thread);

  // Clearning sample and previous result information
  _js["Results"] = knlohmann::json();
  _js["Samples"] = knlohmann::json();

  // Initializing Variables
  if (isDefined(_js.getJson(), "Variables"))
  {
    _variables.resize(_js["Variables"].size());
    for (size_t i = 0; i < _variables.size(); i++) _variables[i] = new Variable;
  }

  // Applying experiment defaults
  applyModuleDefaults(_js.getJson());

  // If this is the initial run, apply defaults
  if (_currentGeneration == 0) setSeed(_js.getJson());

  // Setting configuration
  setConfiguration(_js.getJson());

  // Getting configuration back into the JSON storage
  getConfiguration(_js.getJson());

  // Updating verbosity level
  _logger = new Logger(_consoleOutputVerbosity, stdout);

  // Initializing problem and solver modules
  _problem->initialize();
  _solver->initialize();

  _isInitialized = true;
}

void Experiment::finalize()
{
  for (size_t i = 0; i < _variables.size(); i++) delete _variables[i];
  _variables.clear();
  for (size_t i = 0; i < _distributions.size(); i++) delete _distributions[i];
  _distributions.clear();
  if (_isInitialized == true) co_delete(_thread);
  delete _logger;
  delete _problem;
}

knlohmann::json &Experiment::operator[](const std::string &key) { return _js[key]; }
pybind11::object Experiment::getItem(const pybind11::object key) { return _js.getItem(key); }
void Experiment::setItem(const pybind11::object key, const pybind11::object val) { _js.setItem(key, val); }

void Experiment::setSeed(knlohmann::json &js)
{
  if (isDefined(js, "Random Seed"))
  {
    try
    {
      if (js["Random Seed"].get<size_t>() == 0)
      {
        js["Random Seed"] = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      }
    }
    catch (const std::exception &e)
    {
      KORALI_LOG_ERROR(" + Object: [ Experiment ] \n + Key:    ['Random Seed']\n%s", e.what());
    }
  }
}

void Experiment::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Current Generation"))
 {
 try { _currentGeneration = js["Current Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Current Generation']\n%s", e.what()); } 
   eraseValue(js, "Current Generation");
 }

 if (isDefined(js, "Is Finished"))
 {
 try { _isFinished = js["Is Finished"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Is Finished']\n%s", e.what()); } 
   eraseValue(js, "Is Finished");
 }

 if (isDefined(js, "Run ID"))
 {
 try { _runID = js["Run ID"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Run ID']\n%s", e.what()); } 
   eraseValue(js, "Run ID");
 }

 if (isDefined(js, "Timestamp"))
 {
 try { _timestamp = js["Timestamp"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Timestamp']\n%s", e.what()); } 
   eraseValue(js, "Timestamp");
 }

 if (isDefined(js, "Random Seed"))
 {
 try { _randomSeed = js["Random Seed"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Random Seed']\n%s", e.what()); } 
   eraseValue(js, "Random Seed");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Random Seed'] required by experiment.\n"); 

 if (isDefined(js, "Preserve Random Number Generator States"))
 {
 try { _preserveRandomNumberGeneratorStates = js["Preserve Random Number Generator States"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Preserve Random Number Generator States']\n%s", e.what()); } 
   eraseValue(js, "Preserve Random Number Generator States");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Preserve Random Number Generator States'] required by experiment.\n"); 

 if (isDefined(js, "Distributions"))
 {
 _distributions.resize(js["Distributions"].size());
 for(size_t i = 0; i < js["Distributions"].size(); i++) {
   _distributions[i] = (korali::distribution::Univariate*) korali::Module::getModule(js["Distributions"][i], _k);
   _distributions[i]->applyVariableDefaults();
   _distributions[i]->applyModuleDefaults(js["Distributions"][i]);
   _distributions[i]->setConfiguration(js["Distributions"][i]);
 }
   eraseValue(js, "Distributions");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Distributions'] required by experiment.\n"); 

 if (isDefined(js, "Problem"))
 {
 _problem = dynamic_cast<korali::Problem*>(korali::Module::getModule(js["Problem"], _k));
 _problem->applyVariableDefaults();
 _problem->applyModuleDefaults(js["Problem"]);
 _problem->setConfiguration(js["Problem"]);
   eraseValue(js, "Problem");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Problem'] required by experiment.\n"); 

 if (isDefined(js, "Solver"))
 {
  if (_k->_isInitialized == false) _solver = dynamic_cast<korali::Solver*>(korali::Module::getModule(js["Solver"], _k));
 _solver->applyVariableDefaults();
 _solver->applyModuleDefaults(js["Solver"]);
 _solver->setConfiguration(js["Solver"]);
   eraseValue(js, "Solver");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Solver'] required by experiment.\n"); 

 if (isDefined(js, "File Output", "Path"))
 {
 try { _fileOutputPath = js["File Output"]["Path"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['File Output']['Path']\n%s", e.what()); } 
   eraseValue(js, "File Output", "Path");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['File Output']['Path'] required by experiment.\n"); 

 if (isDefined(js, "File Output", "Use Multiple Files"))
 {
 try { _fileOutputUseMultipleFiles = js["File Output"]["Use Multiple Files"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['File Output']['Use Multiple Files']\n%s", e.what()); } 
   eraseValue(js, "File Output", "Use Multiple Files");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['File Output']['Use Multiple Files'] required by experiment.\n"); 

 if (isDefined(js, "File Output", "Enabled"))
 {
 try { _fileOutputEnabled = js["File Output"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['File Output']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "File Output", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['File Output']['Enabled'] required by experiment.\n"); 

 if (isDefined(js, "File Output", "Frequency"))
 {
 try { _fileOutputFrequency = js["File Output"]["Frequency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['File Output']['Frequency']\n%s", e.what()); } 
   eraseValue(js, "File Output", "Frequency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['File Output']['Frequency'] required by experiment.\n"); 

 if (isDefined(js, "Store Sample Information"))
 {
 try { _storeSampleInformation = js["Store Sample Information"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Store Sample Information']\n%s", e.what()); } 
   eraseValue(js, "Store Sample Information");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Store Sample Information'] required by experiment.\n"); 

 if (isDefined(js, "Console Output", "Verbosity"))
 {
 try { _consoleOutputVerbosity = js["Console Output"]["Verbosity"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Console Output']['Verbosity']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_consoleOutputVerbosity == "Silent") validOption = true; 
 if (_consoleOutputVerbosity == "Minimal") validOption = true; 
 if (_consoleOutputVerbosity == "Normal") validOption = true; 
 if (_consoleOutputVerbosity == "Detailed") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Console Output']['Verbosity'] required by experiment.\n", _consoleOutputVerbosity.c_str()); 
}
   eraseValue(js, "Console Output", "Verbosity");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Console Output']['Verbosity'] required by experiment.\n"); 

 if (isDefined(js, "Console Output", "Frequency"))
 {
 try { _consoleOutputFrequency = js["Console Output"]["Frequency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ experiment ] \n + Key:    ['Console Output']['Frequency']\n%s", e.what()); } 
   eraseValue(js, "Console Output", "Frequency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Console Output']['Frequency'] required by experiment.\n"); 

 Module::setConfiguration(js);
 _type = ".";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: experiment: \n%s\n", js.dump(2).c_str());
} 

void Experiment::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Random Seed"] = _randomSeed;
   js["Preserve Random Number Generator States"] = _preserveRandomNumberGeneratorStates;
 for(size_t i = 0; i < _distributions.size(); i++) _distributions[i]->getConfiguration(js["Distributions"][i]);
 if(_problem != NULL) _problem->getConfiguration(js["Problem"]);
 if(_solver != NULL) _solver->getConfiguration(js["Solver"]);
   js["File Output"]["Path"] = _fileOutputPath;
   js["File Output"]["Use Multiple Files"] = _fileOutputUseMultipleFiles;
   js["File Output"]["Enabled"] = _fileOutputEnabled;
   js["File Output"]["Frequency"] = _fileOutputFrequency;
   js["Store Sample Information"] = _storeSampleInformation;
   js["Console Output"]["Verbosity"] = _consoleOutputVerbosity;
   js["Console Output"]["Frequency"] = _consoleOutputFrequency;
   js["Current Generation"] = _currentGeneration;
   js["Is Finished"] = _isFinished;
   js["Run ID"] = _runID;
   js["Timestamp"] = _timestamp;
 Module::getConfiguration(js);
} 

void Experiment::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Random Seed\": 0, \"Preserve Random Number Generator States\": false, \"Distributions\": [], \"Current Generation\": 0, \"File Output\": {\"Enabled\": true, \"Path\": \"_korali_result\", \"Frequency\": 1, \"Use Multiple Files\": true}, \"Console Output\": {\"Verbosity\": \"Normal\", \"Frequency\": 1}, \"Store Sample Information\": false, \"Is Finished\": false}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Module::applyModuleDefaults(js);
} 

void Experiment::applyVariableDefaults() 
{

 Module::applyVariableDefaults();
} 

;

} //korali
;
