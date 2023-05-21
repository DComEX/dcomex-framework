/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: Conduit.
*/

/** \dir conduit
* @brief Contains code, documentation, and scripts for module: Conduit.
*/

#pragma once

#include "modules/module.hpp"
#include <deque>
#include <vector>

namespace korali
{
;

/**
* @brief Class declaration for module: Conduit.
*/
class Engine;

/**
* @brief Class declaration for module: Conduit.
*/
class Conduit : public Module
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
   * @brief Lifetime function for korali workers.
   */
  void worker();

  /**
   * @brief Double ended queue to store idle workers to assign samples to
   */
  std::deque<size_t> _workerQueue;

  /**
   * @brief Map that links workers to their currently-executing sample
   */
  std::map<size_t, Sample *> _workerToSampleMap;

  /**
   * @brief Determines whether the caller rank/thread/process is root.
   * @return True, if it is root; false, otherwise.
   */
  virtual bool isRoot() const { return true; }

  /**
   * @brief Determines whether the caller rank is the leader of its worker root
   * @return True, if it is the worker leader rank; false, otherwise.
   */
  virtual bool isWorkerLeadRank() const { return true; }

  /**
   * @brief  (Worker Side) Starts the processing of a sample at the worker side
   * @param js Contains sample's input data and metadata
   */
  void workerProcessSample(const knlohmann::json &js);

  /**
   * @brief (Worker Side) Accepts and stacks an incoming Korali engine from the main process
   * @param js Contains Engine's input data and metadata
   */
  void workerStackEngine(const knlohmann::json &js);

  /**
   * @brief (Worker Side) Pops the top of the engine stack
   */
  void workerPopEngine();

  /**
   * @brief Starts the execution of the sample.
   * @param sample A Korali sample
   */
  void start(Sample &sample);

  /**
   * @brief Waits for a given sample to finish. The experiment will not continue until the sample has been evaluated.
   * @param sample A Korali sample
   */
  void wait(Sample &sample);

  /**
   * @brief Waits for a set of sample to finish. The experiment will not continue until all samples have been evaluated.
   * @param samples A list of Korali samples
   */
  void waitAll(std::vector<Sample> &samples);

  /**
   * @brief Waits for a set of sample to finish. The experiment will not continue until at least one of the samples have been evaluated.
   * @param samples A list of Korali samples
   * @return Position in the vector of the sample that has finished.
   */
  size_t waitAny(std::vector<Sample> &samples);

  /**
   * @brief Stacks a new Engine into the engine stack
   * @param engine A Korali Engine
   */
  virtual void stackEngine(Engine *engine) = 0;

  /**
   * @brief Pops the current Engine from the engine stack
   */
  virtual void popEngine() = 0;

  /**
   * @brief Starts the execution of a sample, given an Engine
   * @param sample the sample to execute
   * @param engine The Korali engine to use for its execution
   */
  void runSample(Sample *sample, Engine *engine);

  /**
   * @brief Wrapper function for the sample coroutine
   */
  static void coroutineWrapper();

  /**
   * @brief Initializes the worker/server bifurcation in the conduit
   */
  virtual void initServer() = 0;

  /**
   * @brief Finalizes the workers
   */
  virtual void terminateServer() = 0;

  /**
   * @brief (Engine -> Worker) Broadcasts a message to all workers
   * @param message JSON object with information to broadcast
   */
  virtual void broadcastMessageToWorkers(knlohmann::json &message) = 0;

  /**
   * @brief (Engine <- Worker) Receives all pending incoming messages and stores them into the corresponding sample's message queue.
   */
  virtual void listenWorkers() = 0;

  /**
   * @brief Start pending samples and retrieve any pending messages for them
   * @param samples The set of samples
   */
  void listen(std::vector<Sample> &samples);

  /**
   * @brief (Sample -> Engine) Sends an update to the engine to provide partial information while the sample is still active
   * @param message Message to send to engine
   */
  virtual void sendMessageToEngine(knlohmann::json &message) = 0;

  /**
   * @brief (Sample <- Engine) Blocking call that waits until any message incoming from the engine.
   * @return message from the engine.
   */
  virtual knlohmann::json recvMessageFromEngine() = 0;

  /**
   * @brief (Engine -> Sample) Sends an update to a still active sample
   * @param sample The sample from which to receive an update
   * @param message Message to send to the sample.
   */
  virtual void sendMessageToSample(Sample &sample, knlohmann::json &message) = 0;

  /**
   * @brief Returns the identifier corresponding to the executing process (to differentiate their random seeds)
   * @return The executing process id
   */
  virtual size_t getProcessId() const = 0;

  /**
   * @brief Get total Korali worker count in the conduit
   * @return The number of workers
   */
  virtual size_t getWorkerCount() const = 0;
};

} //korali
;
