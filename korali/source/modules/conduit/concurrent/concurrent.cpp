#include "engine.hpp"
#include "modules/conduit/concurrent/concurrent.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/solver.hpp"
#include "sample/sample.hpp"
#include <fcntl.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/wait.h>

#define BUFFERSIZE 4096

using namespace std;

namespace korali
{
namespace conduit
{
;

void Concurrent::initialize()
{
  // Setting workerId to -1 to identify master process
  _workerId = -1;

  if (_concurrentJobs < 1) KORALI_LOG_ERROR("You need to define at least 1 concurrent job(s) for external models \n");
  _resultSizePipe.clear();
  _resultContentPipe.clear();
  _inputsPipe.clear();
  _workerQueue.clear();

  for (size_t i = 0; i < _concurrentJobs; i++) _resultSizePipe.push_back(vector<int>(2));
  for (size_t i = 0; i < _concurrentJobs; i++) _resultContentPipe.push_back(vector<int>(2));
  for (size_t i = 0; i < _concurrentJobs; i++) _inputsPipe.push_back(vector<int>(2));
  for (size_t i = 0; i < _concurrentJobs; i++) _workerQueue.push_back(i);

  // Opening Inter-process communicator pipes
  for (size_t i = 0; i < _concurrentJobs; i++)
  {
    if (pipe(_inputsPipe[i].data()) == -1) KORALI_LOG_ERROR("Unable to create inter-process pipe. \n");
    if (pipe(_resultSizePipe[i].data()) == -1) KORALI_LOG_ERROR("Unable to create inter-process pipe. \n");
    if (pipe(_resultContentPipe[i].data()) == -1) KORALI_LOG_ERROR("Unable to create inter-process pipe. \n");
    fcntl(_resultSizePipe[i][0], F_SETFL, fcntl(_resultSizePipe[i][0], F_GETFL) | O_NONBLOCK);
    fcntl(_resultSizePipe[i][1], F_SETFL, fcntl(_resultSizePipe[i][1], F_GETFL) | O_NONBLOCK);
    fcntl(_resultContentPipe[i][0], F_SETFL, fcntl(_resultContentPipe[i][0], F_GETFL));
    fcntl(_resultContentPipe[i][1], F_SETFL, fcntl(_resultContentPipe[i][1], F_GETFL));
  }
}

void Concurrent::terminateServer()
{
  auto terminationJs = knlohmann::json();
  terminationJs["Conduit Action"] = "Terminate";

  // Serializing message in binary form
  std::vector<std::uint8_t> msgData = knlohmann::json::to_cbor(terminationJs);
  size_t msgSize = msgData.size();

  for (size_t i = 0; i < _concurrentJobs; i++)
  {
    write(_inputsPipe[i][1], &msgSize, sizeof(size_t));
    write(_inputsPipe[i][1], msgData.data(), msgSize * sizeof(uint8_t));
  }

  for (size_t i = 0; i < _concurrentJobs; i++)
  {
    int status;
    ::wait(&status);
  }

  for (size_t i = 0; i < _concurrentJobs; i++)
  {
    close(_resultContentPipe[i][1]); // Closing pipes
    close(_resultContentPipe[i][0]); // Closing pipes
    close(_resultSizePipe[i][1]);    // Closing pipes
    close(_resultSizePipe[i][0]);    // Closing pipes
    close(_inputsPipe[i][1]);        // Closing pipes
    close(_inputsPipe[i][0]);        // Closing pipes
  }
}

void Concurrent::initServer()
{
  for (size_t i = 0; i < _concurrentJobs; i++)
  {
    pid_t processId = fork();
    if (processId == 0)
    {
      _workerId = i;
      worker();
      exit(0);
    }
    _workerPids.push_back(processId);
  }
}

void Concurrent::broadcastMessageToWorkers(knlohmann::json &message)
{
  // Serializing message in binary form
  std::vector<std::uint8_t> msgData = knlohmann::json::to_cbor(message);
  size_t messageSize = msgData.size();

  for (size_t i = 0; i < _concurrentJobs; i++)
  {
    write(_inputsPipe[i][1], &messageSize, sizeof(size_t));
    write(_inputsPipe[i][1], msgData.data(), messageSize * sizeof(uint8_t));
  }
}

void Concurrent::sendMessageToEngine(knlohmann::json &message)
{
  // Serializing message in binary form
  std::vector<std::uint8_t> msgData = knlohmann::json::to_cbor(message);
  size_t messageSize = msgData.size();

  write(_resultSizePipe[_workerId][1], &messageSize, sizeof(size_t));
  write(_resultContentPipe[_workerId][1], msgData.data(), messageSize * sizeof(uint8_t));
}

knlohmann::json Concurrent::recvMessageFromEngine()
{
  size_t inputSize;
  read(_inputsPipe[_workerId][0], &inputSize, sizeof(size_t));
  std::vector<uint8_t> msgData(inputSize);
  read(_inputsPipe[_workerId][0], msgData.data(), inputSize * sizeof(uint8_t));
  auto message = knlohmann::json::from_cbor(msgData);
  return message;
}

void Concurrent::listenWorkers()
{
  // Check for child defunction
  for (size_t i = 0; i < _workerPids.size(); i++)
  {
    int status;
    pid_t result = waitpid(_workerPids[i], &status, WNOHANG);
    if (result != 0) KORALI_LOG_ERROR("Worker %i (Pid: %d) exited unexpectedly.\n", i, _workerPids[i]);
  }

  // Reading pending messages from all workers
  for (size_t i = 0; i < _workerPids.size(); i++)
  {
    // Identifying current sample
    auto sample = _workerToSampleMap[i];

    size_t inputSize;
    int readBytes = read(_resultSizePipe[i][0], &inputSize, sizeof(size_t));

    if (readBytes > 0)
    {
      std::vector<uint8_t> msgData(inputSize);
      read(_resultContentPipe[i][0], msgData.data(), inputSize * sizeof(uint8_t));
      auto message = knlohmann::json::from_cbor(msgData);
      sample->_messageQueue.push(message);
    }
  }
}

void Concurrent::stackEngine(Engine *engine)
{
  // (Engine-Side) Adding engine to the stack to support Korali-in-Korali execution
  _engineStack.push(engine);

  knlohmann::json engineJs;
  engineJs["Conduit Action"] = "Stack Engine";
  engine->serialize(engineJs["Engine"]);

  broadcastMessageToWorkers(engineJs);
}

void Concurrent::popEngine()
{
  // (Engine-Side) Removing the current engine to the conduit's engine stack
  _engineStack.pop();

  auto popJs = knlohmann::json();
  popJs["Conduit Action"] = "Pop Engine";
  broadcastMessageToWorkers(popJs);
}

void Concurrent::sendMessageToSample(Sample &sample, knlohmann::json &message)
{
  // Serializing message in binary form
  std::vector<std::uint8_t> msgData = knlohmann::json::to_cbor(message);
  size_t messageSize = msgData.size();

  write(_inputsPipe[sample._workerId][1], &messageSize, sizeof(size_t));
  write(_inputsPipe[sample._workerId][1], msgData.data(), messageSize * sizeof(uint8_t));
}

bool Concurrent::isRoot() const
{
  return _workerId == -1;
}

size_t Concurrent::getProcessId() const
{
  return _workerId;
}

size_t Concurrent::getWorkerCount() const
{
  return _concurrentJobs;
}

void Concurrent::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Concurrent Jobs"))
 {
 try { _concurrentJobs = js["Concurrent Jobs"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ concurrent ] \n + Key:    ['Concurrent Jobs']\n%s", e.what()); } 
   eraseValue(js, "Concurrent Jobs");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Concurrent Jobs'] required by concurrent.\n"); 

 Conduit::setConfiguration(js);
 _type = "concurrent";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: concurrent: \n%s\n", js.dump(2).c_str());
} 

void Concurrent::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Concurrent Jobs"] = _concurrentJobs;
 Conduit::getConfiguration(js);
} 

void Concurrent::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Concurrent Jobs\": 1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Conduit::applyModuleDefaults(js);
} 

void Concurrent::applyVariableDefaults() 
{

 Conduit::applyVariableDefaults();
} 

;

} //conduit
} //korali
;
