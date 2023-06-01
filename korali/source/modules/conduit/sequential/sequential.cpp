#include "engine.hpp"
#include "modules/conduit/sequential/sequential.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/solver.hpp"
#include "sample/sample.hpp"
#include <fcntl.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/wait.h>

using namespace std;

namespace korali
{
namespace conduit
{
;

/**
 * @brief Temporary storage to hold the pointer to the current conduit
 */
Sequential *_currentConduit;

void _workerWrapper()
{
  if (_currentConduit != NULL) _currentConduit->worker();
}

void Sequential::initialize()
{
  _workerQueue.push_back(0);
}

void Sequential::initServer()
{
  _currentConduit = this;
  _workerThread = co_create(1 << 28, _workerWrapper);
}

void Sequential::terminateServer()
{
  co_delete(_workerThread);
}

void Sequential::broadcastMessageToWorkers(knlohmann::json &message)
{
  // Queueing outgoing message directly
  _workerMessageQueue.push(message);
}

void Sequential::sendMessageToEngine(knlohmann::json &message)
{
  // Identifying sender sample
  auto sample = _workerToSampleMap[0];

  // Queueing outgoing message directly
  sample->_messageQueue.push(message);
}

knlohmann::json Sequential::recvMessageFromEngine()
{
  Engine *engine = _engineStack.top();

  // Identifying sample
  auto sample = _workerToSampleMap[0];

  // While there's no message, keep executing sample until there is
  while (_workerMessageQueue.empty())
  {
    if (sample->_state == SampleState::running)
      sample->_state = SampleState::waiting;

    co_switch(engine->_currentExperiment->_thread);
  }

  // Pulling message from incoming message queue
  auto message = _workerMessageQueue.front();
  _workerMessageQueue.pop();

  return message;
}

void Sequential::sendMessageToSample(Sample &sample, knlohmann::json &message)
{
  // Queueing message directly
  _workerMessageQueue.push(message);

  co_switch(_workerThread);
}

void Sequential::listenWorkers()
{
  // Just switch back to worker to see if a new message appears
  co_switch(_workerThread);
}

void Sequential::stackEngine(Engine *engine)
{
  // (Engine-Side) Adding engine to the stack to support Korali-in-Korali execution
  _engineStack.push(engine);
}

void Sequential::popEngine()
{
  // (Engine-Side) Removing the current engine to the conduit's engine stack
  _engineStack.pop();
}

bool Sequential::isRoot() const
{
  return true;
}

size_t Sequential::getProcessId() const
{
  return 0;
}

size_t Sequential::getWorkerCount() const
{
  return 1;
}

void Sequential::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Conduit::setConfiguration(js);
 _type = "sequential";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: sequential: \n%s\n", js.dump(2).c_str());
} 

void Sequential::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Conduit::getConfiguration(js);
} 

void Sequential::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Conduit::applyModuleDefaults(js);
} 

void Sequential::applyVariableDefaults() 
{

 Conduit::applyVariableDefaults();
} 

;

} //conduit
} //korali
;
