/** \namespace conduit
* @brief Namespace declaration for modules of type: conduit.
*/

/** \file
* @brief Header file for module: Concurrent.
*/

/** \dir conduit/concurrent
* @brief Contains code, documentation, and scripts for module: Concurrent.
*/

#pragma once

#include "modules/conduit/conduit.hpp"
#include <chrono>
#include <map>
#include <vector>

namespace korali
{
namespace conduit
{
;

/**
* @brief Class declaration for module: Concurrent.
*/
class Concurrent : public Conduit
{
  public: 
  /**
  * @brief Specifies the number of worker processes (jobs) running concurrently.
  */
   size_t _concurrentJobs;
  
 
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
   * @brief PID of worker processes
   */
  std::vector<pid_t> _workerPids;

  /**
   * @brief Worker Id for current workers - 0 for the master process
   */
  int _workerId;

  /**
   * @brief OS Pipe to handle result contents communication coming from worker processes
   */
  std::vector<std::vector<int>> _resultContentPipe;

  /**
   * @brief OS Pipe to handle result size communication coming from worker processes
   */
  std::vector<std::vector<int>> _resultSizePipe;

  /**
   * @brief OS Pipe to handle sample parameter communication to worker processes
   */
  std::vector<std::vector<int>> _inputsPipe;

  bool isRoot() const override;
  void initServer() override;
  void initialize() override;
  void terminateServer() override;

  void stackEngine(Engine *engine) override;
  void popEngine() override;

  void listenWorkers() override;
  void broadcastMessageToWorkers(knlohmann::json &message) override;
  void sendMessageToEngine(knlohmann::json &message) override;
  knlohmann::json recvMessageFromEngine() override;
  void sendMessageToSample(Sample &sample, knlohmann::json &message) override;
  size_t getProcessId() const override;
  size_t getWorkerCount() const override;
};

} //conduit
} //korali
;
