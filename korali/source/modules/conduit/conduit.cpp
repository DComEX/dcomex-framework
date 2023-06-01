#include "engine.hpp"
#include "modules/conduit/conduit.hpp"
#include "modules/experiment/experiment.hpp"
#include "sample/sample.hpp"
#include <chrono>

using namespace std;

namespace korali
{
;

/**
 * @brief Temporary storage to hold the pointer to the current sample to process
 */
Sample *_currentSample;
void Conduit::coroutineWrapper()
{
  // Getting pointers for sample and engine
  Sample *sample = _currentSample;
  Engine *engine = _engineStack.top();

  engine->_conduit->runSample(sample, engine);

  // Now that the sample is finished, set its state to finished and come back to the experiment thread
  sample->_state = SampleState::finished;
  co_switch(engine->_currentExperiment->_thread);
}
//}

void Conduit::runSample(Sample *sample, Engine *engine)
{
  // Setting sample information
  (*sample)["Experiment Id"] = engine->_currentExperiment->_experimentId;
  (*sample)["Current Generation"] = engine->_currentExperiment->_currentGeneration;
  (*sample)["Has Finished"] = false;

  // Identifier of the worker that will execute the sample
  size_t workerId = 0;

  // Check whether there are available workers to compute this sample.
  while (engine->_conduit->_workerQueue.empty())
  {
    //  If none are available, set sample's state back to initialized
    sample->_state = SampleState::initialized;

    // And come back to the experiment's thread
    co_switch(engine->_currentExperiment->_thread);
  }

  // Selecting the next available worker
  workerId = engine->_conduit->_workerQueue.front();
  engine->_conduit->_workerQueue.pop_front();

  // Assigning worker to sample ids and vice-versa for bookkeeping
  (*sample)["Worker Id"] = workerId;
  sample->_workerId = workerId;
  engine->_conduit->_workerToSampleMap[workerId] = sample;

  // Storing profiling information
  auto timelineJs = knlohmann::json();
  timelineJs["Start Time"] = chrono::duration<double>(chrono::high_resolution_clock::now() - _startTime).count() + _cumulativeTime;

  // Sending sample information to worker
  auto sampleJs = sample->_js.getJson();
  sampleJs["Conduit Action"] = "Process Sample";
  engine->_conduit->sendMessageToSample(*sample, sampleJs);

  // Waiting for ending message from sample
  knlohmann::json endMessage;
  do
  {
    // If the sample ending message hasn't arrived, set sample's state back to waiting
    sample->_state = SampleState::waiting;

    // And come back to the experiment's thread
    co_switch(engine->_currentExperiment->_thread);

  } while (sample->retrievePendingMessage(endMessage) == false);

  // Now replacing sample's information by that of the end message
  sample->_js.getJson() = endMessage;

  // Putting worker back to the available worker queue
  engine->_conduit->_workerQueue.push_back(sample->_workerId);

  // Storing profiling information
  timelineJs["End Time"] = chrono::duration<double>(chrono::high_resolution_clock::now() - _startTime).count() + _cumulativeTime;
  timelineJs["Solver Id"] = engine->_currentExperiment->_experimentId;
  timelineJs["Current Generation"] = engine->_currentExperiment->_currentGeneration;
  __profiler["Timelines"]["Worker " + to_string(sample->_workerId)] += timelineJs;

  endMessage.clear();
}

void Conduit::worker()
{
  while (true)
  {
    auto js = recvMessageFromEngine();

    if (js["Conduit Action"] == "Terminate") break;
    if (js["Conduit Action"] == "Process Sample") workerProcessSample(js);
    if (js["Conduit Action"] == "Stack Engine") workerStackEngine(js);
    if (js["Conduit Action"] == "Pop Engine") workerPopEngine();
  }
}

void Conduit::workerProcessSample(const knlohmann::json &js)
{
  auto expId = js["Experiment Id"];
  Sample s;
  s._js.getJson() = js;
  s.sampleLauncher();
  sendMessageToEngine(s._js.getJson());
}

void Conduit::workerStackEngine(const knlohmann::json &js)
{
  auto k = new Engine;

  for (size_t i = 0; i < js["Engine"]["Experiment Vector"].size(); i++)
  {
    auto e = new Experiment;
    e->_js.getJson() = js["Engine"]["Experiment Vector"][i];

    // Random seed needs to be changed to give workers independent distributions.
    e->_js["Random Seed"] = e->_js["Random Seed"].get<size_t>() + 1337 * getProcessId();

    k->_experimentVector.push_back(e);
  }

  k->initializeExperiments();
  k->_conduit = this;

  _engineStack.push(k);
}

void Conduit::workerPopEngine()
{
  _engineStack.pop();
}

void Conduit::start(Sample &sample)
{
  // Checking if sample id was defined
  KORALI_GET(size_t, sample, "Sample Id");

  if (sample._state != SampleState::uninitialized) KORALI_LOG_ERROR("Sample has already been initialized.\n");
  sample._sampleThread = co_create(1 << 28, Conduit::coroutineWrapper);

  _currentSample = &sample;

  sample._state = SampleState::initialized;
  co_switch(sample._sampleThread);
}

void Conduit::wait(Sample &sample)
{
  Engine *engine = _engineStack.top();

  while (sample._state == SampleState::waiting || sample._state == SampleState::initialized)
  {
    // Listen for any pending messages
    listenWorkers();

    // Check for error signals from python
    if (isPythonActive && PyErr_CheckSignals() != 0) KORALI_LOG_ERROR("User requested break.\n");

    sample._state = SampleState::running;
    co_switch(sample._sampleThread);

    if (sample._state == SampleState::waiting || sample._state == SampleState::initialized) co_switch(engine->_thread);
  }

  size_t sampleId = KORALI_GET(size_t, sample, "Sample Id");

  // If the user wants to store sample information, this is where we store its information
  if (engine->_currentExperiment->_storeSampleInformation == true)
    engine->_currentExperiment->_sampleInfo["Samples"][sampleId] = sample._js.getJson();

  sample._state = SampleState::uninitialized;
  co_delete(sample._sampleThread);
}

size_t Conduit::waitAny(vector<Sample> &samples)
{
  Engine *engine = _engineStack.top();
  bool isFinished = false;
  size_t currentSample;

  while (isFinished == false)
  {
    // Listen for any pending messages
    listenWorkers();

    // Check for error signals from python
    if (isPythonActive && PyErr_CheckSignals() != 0) throw pybind11::error_already_set();

    for (currentSample = 0; currentSample < samples.size(); currentSample++)
    {
      if (samples[currentSample]._state == SampleState::waiting || samples[currentSample]._state == SampleState::initialized)
      {
        samples[currentSample]._state = SampleState::running;
        co_switch(samples[currentSample]._sampleThread);
      }

      if (samples[currentSample]._state == SampleState::finished)
      {
        auto sampleId = KORALI_GET(size_t, samples[currentSample], "Sample Id");

        // If the user wants to store sample information, this is where we store its information
        if (engine->_currentExperiment->_storeSampleInformation == true)
          engine->_currentExperiment->_sampleInfo["Samples"][sampleId] = samples[currentSample]._js.getJson();

        samples[currentSample]._state = SampleState::uninitialized;
        co_delete(samples[currentSample]._sampleThread);
        isFinished = true;
        break;
      }
    }

    if (isFinished == false) co_switch(engine->_thread);
  }

  return currentSample;
}

void Conduit::waitAll(vector<Sample> &samples)
{
  Engine *engine = _engineStack.top();
  bool isFinished = false;

  while (isFinished == false)
  {
    // Listen for any pending messages
    listenWorkers();

    // Check for error signals from python
    if (isPythonActive && PyErr_CheckSignals() != 0) KORALI_LOG_ERROR("User requested break.\n");

    isFinished = true;

    for (size_t i = 0; i < samples.size(); i++)
      if (samples[i]._state == SampleState::waiting || samples[i]._state == SampleState::initialized)
      {
        isFinished = false;
        samples[i]._state = SampleState::running;
        co_switch(samples[i]._sampleThread);
      }

    if (isFinished == false) co_switch(engine->_thread);
  }

  for (size_t i = 0; i < samples.size(); i++)
  {
    auto sampleId = KORALI_GET(size_t, samples[i], "Sample Id");

    // If the user wants to store sample information, this is where we store its information
    if (engine->_currentExperiment->_storeSampleInformation == true)
      engine->_currentExperiment->_sampleInfo["Samples"][sampleId] = samples[i]._js.getJson();

    samples[i]._state = SampleState::uninitialized;
    co_delete(samples[i]._sampleThread);
  }
}

void Conduit::listen(std::vector<Sample> &samples)
{
  // Listen for any pending messages
  listenWorkers();

  // Check for error signals from python
  if (isPythonActive && PyErr_CheckSignals() != 0) throw pybind11::error_already_set();

  // Doing a pass on samples in case they are waiting to execute
  for (size_t currentSample = 0; currentSample < samples.size(); currentSample++)
    if (samples[currentSample]._state == SampleState::initialized)
    {
      samples[currentSample]._state = SampleState::running;
      co_switch(samples[currentSample]._sampleThread);
    }
}

void Conduit::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Module::setConfiguration(js);
 _type = ".";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: conduit: \n%s\n", js.dump(2).c_str());
} 

void Conduit::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Module::getConfiguration(js);
} 

void Conduit::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Module::applyModuleDefaults(js);
} 

void Conduit::applyVariableDefaults() 
{

 Module::applyVariableDefaults();
} 

;

} //korali
;
