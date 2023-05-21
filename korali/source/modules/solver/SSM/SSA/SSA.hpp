/** \namespace ssm
* @brief Namespace declaration for modules of type: ssm.
*/

/** \file
* @brief Header file for module: SSA.
*/

/** \dir solver/SSM/SSA
* @brief Contains code, documentation, and scripts for module: SSA.
*/

#pragma once

#include "modules/solver/SSM/SSM.hpp"

namespace korali
{
namespace solver
{
namespace ssm
{
;

/**
* @brief Class declaration for module: SSA.
*/
class SSA : public SSM
{
  public: 
  
 
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
   * @brief Storage for cumulative propensities during each step
   */
  std::vector<double> _cumPropensities;

  void advance() override;
};

} //ssm
} //solver
} //korali
;
