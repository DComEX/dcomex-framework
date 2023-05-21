/** \namespace problem
* @brief Namespace declaration for modules of type: problem.
*/

/** \file
* @brief Header file for module: ReinforcementLearning.
*/

/** \dir problem/reinforcementLearning
* @brief Contains code, documentation, and scripts for module: ReinforcementLearning.
*/

#pragma once

#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"
#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
;

/**
* @brief Class declaration for module: ReinforcementLearning.
*/
class ReinforcementLearning : public Problem
{
  public: 
  /**
  * @brief Number of agents in a given environment .
  */
   size_t _agentsPerEnvironment;
  /**
  * @brief Number of policies in a given environment. All agents share the same policy or all have individual policy.
  */
   size_t _policiesPerEnvironment;
  /**
  * @brief Maximum number of different types of environments.
  */
   size_t _environmentCount;
  /**
  * @brief Function to initialize and run an episode in the environment.
  */
   std::uint64_t _environmentFunction;
  /**
  * @brief Number of actions to take before requesting a new policy.
  */
   size_t _actionsBetweenPolicyUpdates;
  /**
  * @brief Number of episodes after which the policy will be tested.
  */
   size_t _testingFrequency;
  /**
  * @brief Number of test episodes to run the policy (without noise) for, for which the average sum of rewards will serve to evaluate the termination criteria.
  */
   size_t _policyTestingEpisodes;
  /**
  * @brief Any used-defined settings required by the environment.
  */
   knlohmann::json _customSettings;
  /**
  * @brief [Internal Use] Stores the dimension of the action space.
  */
   size_t _actionVectorSize;
  /**
  * @brief [Internal Use] Stores the dimension of the state space.
  */
   size_t _stateVectorSize;
  /**
  * @brief [Internal Use] Stores the indexes of the variables that constitute the action vector.
  */
   std::vector<size_t> _actionVectorIndexes;
  /**
  * @brief [Internal Use] Stores the indexes of the variables that constitute the action vector.
  */
   std::vector<size_t> _stateVectorIndexes;
  /**
  * @brief [Internal Use] The maximum number of actions an agent can take (only relevant for discrete).
  */
   size_t _actionCount;
  
 
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
  * @brief Runs the operation specified on the given sample. It checks recursively whether the function was found by the current module or its parents.
  * @param sample Sample to operate on. Should contain in the 'Operation' field an operation accepted by this module or its parents.
  * @param operation Should specify an operation type accepted by this module or its parents.
  * @return True, if operation found and executed; false, otherwise.
  */
  bool runOperation(std::string operation, korali::Sample& sample) override;
  

  void initialize() override;

  /**
   * @brief Runs an episode of the agent within the environment with actions produced by the policy + exploratory noise. If the reward exceeds the threshold, it also runs testing episodes.
   * @param agent Sample containing current agent/state information.
   */
  void runTrainingEpisode(korali::Sample &agent);

  /**
   * @brief Runs an episode of the agent within the environment with actions produced by the policy only.
   * @param agent Sample containing current agent/state information.
   */
  void runTestingEpisode(korali::Sample &agent);

  /**
   * @brief Initializes the environment and agent configuration
   * @param agent Sample containing current agent/state information.
   */
  void initializeEnvironment(korali::Sample &agent);

  /**
   * @brief Finalizes the environemnt (frees resources)
   */
  void finalizeEnvironment();

  /**
   * @brief Runs/resumes the execution of the environment
   * @param agent Sample containing current agent/state information.
   */
  void runEnvironment(Sample &agent);

  /**
   * @brief Communicates with the Engine to get the latest policy
   * @param agent Sample containing current agent/state information.
   */
  void requestNewPolicy(Sample &agent);

  /**
   * @brief Runs the policy on the current state to get the action
   * @param agent Sample containing current agent/state information.
   */
  void getAction(Sample &agent);

  /**
   * @brief Contains the state rescaling means
   */
  std::vector<std::vector<float>> _stateRescalingMeans;

  /**
   * @brief Contains the state rescaling sigmas
   */
  std::vector<std::vector<float>> _stateRescalingSdevs;

  /**
   * @brief [Profiling] Stores policy evaluation time per episode
   */
  double _agentPolicyEvaluationTime;

  /**
   * @brief [Profiling] Stores environment evaluation time per episode
   */
  double _agentComputationTime;

  /**
   * @brief [Profiling] Stores communication time per episode
   */
  double _agentCommunicationTime;
};

} //problem
} //korali
;
