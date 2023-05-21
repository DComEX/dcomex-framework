/** \namespace solver
* @brief Namespace declaration for modules of type: solver.
*/

/** \file
* @brief Header file for module: Optimizer.
*/

/** \dir solver/optimizer
* @brief Contains code, documentation, and scripts for module: Optimizer.
*/

#pragma once

#include "modules/solver/solver.hpp"

namespace korali
{
namespace solver
{
;

/**
* @brief Class declaration for module: Optimizer.
*/
class Optimizer : public Solver
{
  public: 
  /**
  * @brief [Internal Use] Best model evaluation from current generation.
  */
   double _currentBestValue;
  /**
  * @brief [Internal Use] Best model evaluation from previous generation.
  */
   double _previousBestValue;
  /**
  * @brief [Internal Use] Best ever model evaluation.
  */
   double _bestEverValue;
  /**
  * @brief [Internal Use] Variables associated to best ever value found.
  */
   std::vector<double> _bestEverVariables;
  /**
  * @brief [Internal Use] Keeps count of the number of infeasible samples.
  */
   size_t _infeasibleSampleCount;
  /**
  * @brief [Termination Criteria] Specifies the maximum target fitness to stop maximization.
  */
   double _maxValue;
  /**
  * @brief [Termination Criteria] Specifies the minimum fitness differential between two consecutive generations before stopping execution.
  */
   double _minValueDifferenceThreshold;
  /**
  * @brief [Termination Criteria] Maximum number of resamplings per candidate per generation if sample is outside of Lower and Upper Bound.
  */
   size_t _maxInfeasibleResamplings;
  
 
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
   * @brief Checks whether the proposed sample can be optimized
   * @param sample A Korali Sample
   * @return True, if feasible; false, otherwise.
   */
  bool isSampleFeasible(const std::vector<double> &sample);
};

} //solver
} //korali
;
