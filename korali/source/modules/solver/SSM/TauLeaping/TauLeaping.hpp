/** \namespace ssm
* @brief Namespace declaration for modules of type: ssm.
*/

/** \file
* @brief Header file for module: TauLeaping.
*/

/** \dir solver/SSM/TauLeaping
* @brief Contains code, documentation, and scripts for module: TauLeaping.
*/

#pragma once

#include "modules/distribution/univariate/poisson/poisson.hpp"
#include "modules/solver/SSM/SSM.hpp"

namespace korali
{
namespace solver
{
namespace ssm
{
;

/**
* @brief Class declaration for module: TauLeaping.
*/
class TauLeaping : public SSM
{
  public: 
  /**
  * @brief TODO.
  */
   int _nc;
  /**
  * @brief Error control parameter, larger epsilon leads to larger tau steps and errors.
  */
   double _epsilon;
  /**
  * @brief Multiplicator of inverse total propensity, to calculate acceptance criterion of tau step.
  */
   double _acceptanceFactor;
  /**
  * @brief Number of intermediate SSA steps if leap step rejected.
  */
   int _numSSASteps;
  /**
  * @brief [Internal Use] Poisson random number generator.
  */
   korali::distribution::univariate::Poisson* _poissonGenerator;
  /**
  * @brief [Internal Use] Estimated means of the expected change of reactants.
  */
   std::vector<double> _mu;
  /**
  * @brief [Internal Use] Estimated variance of the expected change of reactants.
  */
   std::vector<double> _variance;
  
 
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
   * @brief Storage for propensities of reactions during each step
   */
  std::vector<double> _propensities;

  /**
   * @brief Storage for cumulative propensities during each SSA step
   */
  std::vector<double> _cumPropensities;

  /**
   * @brief Storage for number of firings per reaction per tau leap step
   */
  std::vector<int> _numFirings;

  /**
   * @brief Storage for critical reaction marker during each step
   */
  std::vector<double> _isCriticalReaction;

  /**
   * @brief Storage for candidate reactants per leap step
   */
  std::vector<int> _candidateNumReactants;

  /**
   * @brief Estimate time step such that that many reaction events occur, but small enough that no propensity functin changes significantly
   * @return tau time step duration
   */
  double estimateLargestTau();

  /**
   * @brief SSA advance step if leap step rejected.
   */
  void ssaAdvance();

  void setInitialConfiguration() override;

  void advance() override;
};

} //ssm
} //solver
} //korali
;
