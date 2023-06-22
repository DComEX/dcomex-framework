/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: Problem.
*/

/** \dir problem
* @brief Contains code, documentation, and scripts for module: Problem.
*/

#pragma once

#include "modules/experiment/experiment.hpp"
#include "modules/module.hpp"

namespace korali
{
;

/**
* @brief Class declaration for module: Problem.
*/
class Problem : public Module
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
   * @brief Evaluates whether at least one of constraints have been met.
   * @param operation Name of the operation
   * @param sample A Korali Sample
   */
};

} //korali
;
