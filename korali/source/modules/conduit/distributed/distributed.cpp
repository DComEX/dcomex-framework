#include "auxiliar/MPIUtils.hpp"
#include "engine.hpp"
#include "modules/conduit/distributed/distributed.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/solver.hpp"
#include "sample/sample.hpp"

using namespace std;

namespace korali
{
namespace conduit
{
;

void Distributed::initialize()
{
#ifndef _KORALI_USE_MPI
  KORALI_LOG_ERROR("Running an Distributed-based Korali application, but Korali was installed without support for MPI.\n");
#else

  // Sanity checks for the correct initialization of MPI
  int isInitialized = 0;
  MPI_Initialized(&isInitialized);
  if (__isMPICommGiven == false) KORALI_LOG_ERROR("Korali requires that MPI communicator is passed (via setMPIComm) prior to running the engine.\n");
  if (isInitialized == 0) KORALI_LOG_ERROR("Korali requires that the MPI is initialized by the user (e.g., via MPI_init) prior to running the engine.\n");

  // Getting size and id from the user-defined communicator
  MPI_Comm_size(__KoraliGlobalMPIComm, &_rankCount);
  MPI_Comm_rank(__KoraliGlobalMPIComm, &_rankId);

  // Determining ranks per worker
  _workerCount = (_rankCount - _engineRanks) / _ranksPerWorker;
  size_t workerRemainder = (_rankCount - _engineRanks) % _ranksPerWorker;

  // If this is a root rank, check whether configuration is correct
  if (isRoot())
  {
    if (_rankCount < 2) KORALI_LOG_ERROR("Korali Distributed applications require at least 2 MPI ranks to run (provided %d).\n", _rankCount);
    if (_ranksPerWorker < 1) KORALI_LOG_ERROR("The distributed conduit requires that the ranks per worker is equal or larger than 1, provided: %d\n", _ranksPerWorker);
    if (_engineRanks < 1) KORALI_LOG_ERROR("The distributed conduit requires that the engine ranks is equal or larger than 1, provided: %d\n", _engineRanks);
    if (workerRemainder != 0) KORALI_LOG_ERROR("Korali was instantiated with %lu MPI ranks (minus %lu for the engine), divided into %lu workers. This setup does not provide a perfectly divisible distribution, and %lu unused ranks remain.\n", _rankCount, _engineRanks, _workerCount, workerRemainder);
  }

  // Synchronizing all ranks
  MPI_Barrier(__KoraliGlobalMPIComm);

  // Initializing worker id setting storage
  _localRankId = 0;
  _workerIdSet = false;

  // Storage to map MPI ranks to their corresponding worker
  _workerTeams.resize(_workerCount);
  for (int i = 0; i < _workerCount; i++)
    _workerTeams[i].resize(_ranksPerWorker);

  // Storage to map workers to MPI ranks
  _rankToWorkerMap.resize(_rankCount);

  // Initializing available worker queue
  _workerQueue.clear();

  // Putting workers in the queue
  for (int i = 0; i < _workerCount; i++) _workerQueue.push_back(i);

  // Korali engine as default, is the n+1th worker
  int curWorker = _workerCount + 1;

  // Now assigning ranks to workers and viceversa
  int currentRank = 0;
  for (int i = 0; i < _workerCount; i++)
    for (int j = 0; j < _ranksPerWorker; j++)
    {
      if (currentRank == _rankId)
      {
        curWorker = i;
        _localRankId = j;
        _workerIdSet = true;
      }

      _workerTeams[i][j] = currentRank;
      _rankToWorkerMap[currentRank] = i;
      currentRank++;
    }

  // Creating communicator
  MPI_Comm_split(__KoraliGlobalMPIComm, curWorker, _rankId, &__koraliWorkerMPIComm);

  // Waiting for all ranks to reach this point
  MPI_Barrier(__KoraliGlobalMPIComm);
#endif
}

void Distributed::initServer()
{
#ifdef _KORALI_USE_MPI

  // Workers run and synchronize at the end
  if (isRoot() == false && _workerIdSet == true) worker();

  // Non root engine ranks passively wait and finish upon synchronization
  if (isRoot() == false && _workerIdSet == false)
  {
    MPI_Request req;
    int flag;
    MPI_Irecv(&flag, 1, MPI_INT, getRootRank(), __KORALI_MPI_MESSAGE_JSON_TAG, MPI_COMM_WORLD, &req);
    int finalized = 0;
    while (finalized == 0)
    {
      // Passive wait, 1s at a time
      usleep(1000000);
      MPI_Test(&req, &finalized, MPI_STATUS_IGNORE);
    }
  }

#endif
}

void Distributed::terminateServer()
{
#ifdef _KORALI_USE_MPI
  auto terminationJs = knlohmann::json();
  terminationJs["Conduit Action"] = "Terminate";

  // Serializing message in binary form
  const std::vector<std::uint8_t> msgData = knlohmann::json::to_cbor(terminationJs);

  if (isRoot())
  {
    // Sending message to workers for termination
    for (int i = 0; i < _workerCount; i++)
      for (int j = 0; j < _ranksPerWorker; j++)
        MPI_Send(msgData.data(), msgData.size(), MPI_UINT8_T, _workerTeams[i][j], __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm);

    // Sending message to Korali non-root engine ranks for termination
    int flag = 0;
    for (int i = _rankCount - _engineRanks; i < _rankCount - 1; i++)
      MPI_Send(&flag, 1, MPI_INT, i, __KORALI_MPI_MESSAGE_JSON_TAG, MPI_COMM_WORLD);
  }

#endif

  Conduit::finalize();
}

void Distributed::broadcastMessageToWorkers(knlohmann::json &message)
{
#ifdef _KORALI_USE_MPI
  // Run broadcast only if this is the master process
  if (!isRoot()) return;

  // Serializing message in binary form
  const std::vector<std::uint8_t> msgData = knlohmann::json::to_cbor(message);

  for (int i = 0; i < _workerCount; i++)
    for (int j = 0; j < _ranksPerWorker; j++)
      MPI_Send(msgData.data(), msgData.size(), MPI_UINT8_T, _workerTeams[i][j], __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm);
#endif
}

int Distributed::getRootRank() const
{
#ifdef _KORALI_USE_MPI
  return _rankCount - 1;
#endif

  return 0;
}

bool Distributed::isRoot() const
{
#ifdef _KORALI_USE_MPI
  return _rankId == getRootRank();
#endif

  return true;
}

bool Distributed::isWorkerLeadRank() const
{
  // Arbitrarily, we decide that rank 0 is the root rank
  return _localRankId == 0;
}

void Distributed::sendMessageToEngine(knlohmann::json &message)
{
#ifdef _KORALI_USE_MPI
  if (_localRankId == 0)
  {
    // Serializing message in binary form
    const std::vector<std::uint8_t> msgData = knlohmann::json::to_cbor(message);
    MPI_Send(msgData.data(), msgData.size(), MPI_UINT8_T, getRootRank(), __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm);
  }
#endif
}

knlohmann::json Distributed::recvMessageFromEngine()
{
  auto message = knlohmann::json();

#ifdef _KORALI_USE_MPI
  MPI_Barrier(__koraliWorkerMPIComm);

  MPI_Status status;
  MPI_Probe(getRootRank(), __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm, &status);
  int messageSize = 0;
  MPI_Get_count(&status, MPI_UINT8_T, &messageSize);

  // Allocating receive buffer
  auto msgData = std::vector<std::uint8_t>(messageSize);
  MPI_Recv(msgData.data(), messageSize, MPI_UINT8_T, getRootRank(), __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm, MPI_STATUS_IGNORE);
  message = knlohmann::json::from_cbor(msgData);
#endif

  return message;
}

void Distributed::listenWorkers()
{
#ifdef _KORALI_USE_MPI

  // Scanning all incoming messages
  int foundMessage = 0;

  // Reading pending messages from any worker
  MPI_Status status;
  MPI_Iprobe(MPI_ANY_SOURCE, __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm, &foundMessage, &status);

  // If message found, receive it and storing in the corresponding sample's queue
  if (foundMessage == 1)
  {
    // Obtaining source rank, worker ID, and destination sample from the message
    int source = status.MPI_SOURCE;
    int worker = _rankToWorkerMap[source];
    auto sample = _workerToSampleMap[worker];

    // Receiving message from the worker
    int messageSize = 0;
    MPI_Get_count(&status, MPI_UINT8_T, &messageSize);
    auto msgData = std::vector<std::uint8_t>(messageSize);
    MPI_Recv(msgData.data(), msgData.size(), MPI_UINT8_T, source, __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm, MPI_STATUS_IGNORE);
    auto message = knlohmann::json::from_cbor(msgData);

    // Storing message in the sample message queue
    sample->_messageQueue.push(message);
  }

#endif
}

void Distributed::sendMessageToSample(Sample &sample, knlohmann::json &message)
{
#ifdef _KORALI_USE_MPI

  // Serializing message in binary form
  std::vector<std::uint8_t> msgData = knlohmann::json::to_cbor(message);

  for (int i = 0; i < _ranksPerWorker; i++)
  {
    int rankId = _workerTeams[sample._workerId][i];
    MPI_Send(msgData.data(), msgData.size(), MPI_UINT8_T, rankId, __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm);
  }
#endif
}

void Distributed::stackEngine(Engine *engine)
{
#ifdef _KORALI_USE_MPI
  // (Engine-Side) Adding engine to the stack to support Korali-in-Korali execution
  _engineStack.push(engine);

  knlohmann::json engineJs;
  engineJs["Conduit Action"] = "Stack Engine";
  engine->serialize(engineJs["Engine"]);

  broadcastMessageToWorkers(engineJs);
#endif
}

void Distributed::popEngine()
{
#ifdef _KORALI_USE_MPI
  // (Engine-Side) Removing the current engine to the conduit's engine stack
  _engineStack.pop();

  auto popJs = knlohmann::json();
  popJs["Conduit Action"] = "Pop Engine";
  broadcastMessageToWorkers(popJs);
#endif
}

size_t Distributed::getProcessId() const
{
  return _rankId;
}

size_t Distributed::getWorkerCount() const
{
  return _workerCount;
}

void Distributed::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Ranks Per Worker"))
 {
 try { _ranksPerWorker = js["Ranks Per Worker"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ distributed ] \n + Key:    ['Ranks Per Worker']\n%s", e.what()); } 
   eraseValue(js, "Ranks Per Worker");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Ranks Per Worker'] required by distributed.\n"); 

 if (isDefined(js, "Engine Ranks"))
 {
 try { _engineRanks = js["Engine Ranks"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ distributed ] \n + Key:    ['Engine Ranks']\n%s", e.what()); } 
   eraseValue(js, "Engine Ranks");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Engine Ranks'] required by distributed.\n"); 

 Conduit::setConfiguration(js);
 _type = "distributed";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: distributed: \n%s\n", js.dump(2).c_str());
} 

void Distributed::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Ranks Per Worker"] = _ranksPerWorker;
   js["Engine Ranks"] = _engineRanks;
 Conduit::getConfiguration(js);
} 

void Distributed::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Ranks Per Worker\": 1, \"Engine Ranks\": 1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Conduit::applyModuleDefaults(js);
} 

void Distributed::applyVariableDefaults() 
{

 Conduit::applyVariableDefaults();
} 

;

} //conduit
} //korali
;
