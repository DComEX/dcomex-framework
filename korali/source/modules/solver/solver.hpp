/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: Solver.
*/

/** \dir solver
* @brief Contains code, documentation, and scripts for module: Solver.
*/

#pragma once

#include "auxiliar/libco/libco.h"
#include "modules/experiment/experiment.hpp"
#include "modules/module.hpp"
#include "sample/sample.hpp"
#include <string>
#include <vector>

/*! \namespace Korali
    \brief The Korali namespace includes all Korali-specific functions, variables, and modules.
*/
namespace korali
{
;

/**
 * @brief Macro to start the processing of a sample.
 */
#define KORALI_START(SAMPLE)                \
  {                                         \
    if (_k->_overrideEngine == false)       \
      _k->_engine->_conduit->start(SAMPLE); \
    else                                    \
      _k->_overrideFunction(SAMPLE);        \
  }

/**
 * @brief Macro to wait for the finishing of a sample.
 */
#define KORALI_WAIT(SAMPLE)                                                \
  {                                                                        \
    if (_k->_overrideEngine == false) _k->_engine->_conduit->wait(SAMPLE); \
  }

/**
 * @brief Macro to wait for any of the given samples.
 */
#define KORALI_WAITANY(SAMPLES) _k->_engine->_conduit->waitAny(SAMPLES);

/**
 * @brief Macro to wait for all of the given samples.
 */
#define KORALI_WAITALL(SAMPLES) _k->_engine->_conduit->waitAll(SAMPLES);

/**
 * @brief Macro to send a message to a sample
 */
#define KORALI_SEND_MSG_TO_SAMPLE(SAMPLE, MSG) _k->_engine->_conduit->sendMessageToSample(SAMPLE, MSG);

/**
 * @brief Macro to receive a message from a sample (blocking)
 */
#define KORALI_RECV_MSG_FROM_SAMPLE(SAMPLE) _k->_engine->_conduit->recvMessageFromSample(SAMPLE);

/**
 * @brief (Blocking) Receives all pending incoming messages (at least one) and stores them into the corresponding sample's message queue.
 */
#define KORALI_LISTEN(SAMPLES) _k->_engine->_conduit->listen(SAMPLES);

/**
* @brief Class declaration for module: Solver.
*/
class Solver : public Module
{
  public: 
  /**
  * @brief [Internal Use] Number of variables.
  */
   size_t _variableCount;
  /**
  * @brief [Internal Use] Keeps track on the number of calls to the computational model.
  */
   size_t _modelEvaluationCount;
  /**
  * @brief [Termination Criteria] Specifies the maximum allowed evaluations of the computational model.
  */
   size_t _maxModelEvaluations;
  /**
  * @brief [Termination Criteria] Determines how many solver generations to run before stopping execution. Execution can be resumed at a later moment.
  */
   size_t _maxGenerations;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
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
   * @brief Prints solver information before the execution of the current generation.
   */
  virtual void printGenerationBefore();

  /**
   * @brief Prints solver information after the execution of the current generation.
   */
  virtual void printGenerationAfter();

  /**
   * @brief Runs the current generation.
   */
  virtual void runGeneration() = 0;

  /**
   * @brief Initializes the solver with starting values for the first generation.
   */
  virtual void setInitialConfiguration();

  /**
   * @brief Stores termination criteria for the module.
   */
  std::vector<std::string> _terminationCriteria;
};

} //korali
;
