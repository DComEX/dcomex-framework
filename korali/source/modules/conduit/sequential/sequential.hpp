/** \namespace conduit
* @brief Namespace declaration for modules of type: conduit.
*/

/** \file
* @brief Header file for module: Sequential.
*/

/** \dir conduit/sequential
* @brief Contains code, documentation, and scripts for module: Sequential.
*/

#pragma once

#include "auxiliar/libco/libco.h"
#include "modules/conduit/conduit.hpp"
#include <chrono>
#include <map>
#include <queue>
#include <vector>

namespace korali
{
namespace conduit
{
;

/**
* @brief Class declaration for module: Sequential.
*/
class Sequential : public Conduit
{
  public: 
  
 
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
   * @brief User-Level thread (coroutine) containing the CPU execution state of the single worker.
   */
  cothread_t _workerThread;

  /**
   * @brief Queue of messages sent from the engine to the worker
   */
  std::queue<knlohmann::json> _workerMessageQueue;

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
