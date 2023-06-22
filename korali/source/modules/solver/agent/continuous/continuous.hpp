/** \namespace agent
* @brief Namespace declaration for modules of type: agent.
*/

/** \file
* @brief Header file for module: Continuous.
*/

/** \dir solver/agent/continuous
* @brief Contains code, documentation, and scripts for module: Continuous.
*/

#pragma once

#include "modules/distribution/univariate/beta/beta.hpp"
#include "modules/problem/reinforcementLearning/continuous/continuous.hpp"
#include "modules/solver/agent/agent.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
;

/**
* @brief Class declaration for module: Continuous.
*/
class Continuous : public Agent
{
  public: 
  /**
  * @brief Specifies which probability distribution to use for the policy.
  */
   std::string _policyDistribution;
  /**
  * @brief [Internal Use] Gaussian random number generator to generate the agent's action.
  */
   korali::distribution::univariate::Normal* _normalGenerator;
  /**
  * @brief [Internal Use] Shifts required for bounded actions.
  */
   std::vector<float> _actionShifts;
  /**
  * @brief [Internal Use] Scales required for bounded actions (half the action domain width).
  */
   std::vector<float> _actionScales;
  /**
  * @brief [Internal Use] Stores the transformations required for each parameter.
  */
   std::vector<std::string> _policyParameterTransformationMasks;
  /**
  * @brief [Internal Use] Stores the scaling required for the parameter after the transformation is applied.
  */
   std::vector<float> _policyParameterScaling;
  /**
  * @brief [Internal Use] Stores the shifting required for the parameter after the scaling is applied.
  */
   std::vector<float> _policyParameterShifting;
  
 
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
   * @brief Storage for the pointer to the (continuous) learning problem
   */
  problem::reinforcementLearning::Continuous *_problem;

  /**
   * @brief Calculates the gradient of teh importance weight  wrt to the parameter of the 2nd (current) distribution evaluated at old action.
   * @param action The action taken by the agent in the given experience
   * @param oldPolicy The policy for the given state used at the time the action was performed
   * @param curPolicy The current policy for the given state
   * @param importanceWeight The importance weight
   * @return gradient of policy wrt curParamsOne and curParamsTwo
   */
  std::vector<float> calculateImportanceWeightGradient(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy, const float importanceWeight);

  /**
   * @brief Calculates the gradient of KL(p_old, p_cur) wrt to the parameter of the 2nd (current) distribution.
   * @param oldPolicy The policy for the given state used at the time the action was performed
   * @param curPolicy The current policy for the given state
   * @return
   */
  std::vector<float> calculateKLDivergenceGradient(const policy_t &oldPolicy, const policy_t &curPolicy);

  /**
   * @brief Function to generate randomized actions from neural network output.
   * @param curPolicy The current policy for the given state
   * @return An action vector
   */
  std::vector<float> generateTrainingAction(policy_t &curPolicy);

  /**
   * @brief Function to generate deterministic actions from neural network output required for policy evaluation, respectively testing.
   * @param curPolicy The current policy for the given state
   * @return An action vector
   */
  std::vector<float> generateTestingAction(const policy_t &curPolicy);

  float calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy) override;
  virtual void getAction(korali::Sample &sample) override;
  virtual void initializeAgent() override;
};

} //agent
} //solver
} //korali
;
