/** \namespace discrete
* @brief Namespace declaration for modules of type: discrete.
*/

/** \file
* @brief Header file for module: dVRACER.
*/

/** \dir solver/agent/discrete/dVRACER
* @brief Contains code, documentation, and scripts for module: dVRACER.
*/

#pragma once

#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/problem/reinforcementLearning/discrete/discrete.hpp"
#include "modules/solver/agent/discrete/discrete.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
namespace discrete
{
;

/**
* @brief Class declaration for module: dVRACER.
*/
class dVRACER : public Discrete
{
  public: 
  /**
  * @brief Initial inverse temperature of the softmax distribution. Large values lead to a distribution that is more concentrated around the action with highes Q-value estimate.
  */
   float _initialInverseTemperature;
  /**
  * @brief [Internal Use] Measure of unlikeability for categorial data, approaches 1.0 for uniform behavior and 0. for deterministic case.
  */
   float _statisticsAverageInverseTemperature;
  /**
  * @brief [Internal Use] Measure of unlikeability for categorial data, approaches 1.0 for uniform behavior and 0. for deterministic case.
  */
   float _statisticsAverageActionUnlikeability;
  
 
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
   * @brief Update the V-target or current and previous experiences in the episode
   * @param expId Current Experience Id
   */
  void updateVtbc(size_t expId);

  /**
   * @brief Calculates the gradients for the policy/critic neural network
   * @param miniBatch The indexes of the experience mini batch
   * @param policyIdx The indexes of the policy to compute the gradient for
   */
  void calculatePolicyGradients(const std::vector<std::pair<size_t, size_t>> &miniBatch, const size_t policyIdx);

  /**
   * @brief Retreives the policy infos for the samples in the minibatch
   * @param miniBatch The indexes of the experience mini batch
   * @return A vector containing the policy infos in the order they are given in the miniBatch
   */
  std::vector<policy_t> getPolicyInfo(const std::vector<std::pair<size_t, size_t>> &miniBatch) const;

  float calculateStateValue(const std::vector<std::vector<float>> &stateSequence, size_t policyIdx = 0) override;
  void runPolicy(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &policy, size_t policyIdx = 0) override;
  knlohmann::json getPolicy() override;
  void setPolicy(const knlohmann::json &hyperparameters) override;
  void trainPolicy() override;
  void printInformation() override;
  void initializeAgent() override;
};

} //discrete
} //agent
} //solver
} //korali
;
