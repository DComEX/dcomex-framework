/** \namespace distribution
* @brief Namespace declaration for modules of type: distribution.
*/

/** \file
* @brief Header file for module: Specific.
*/

/** \dir distribution/specific
* @brief Contains code, documentation, and scripts for module: Specific.
*/

#pragma once

#include "modules/distribution/distribution.hpp"

namespace korali
{
namespace distribution
{
;

/**
* @brief Class declaration for module: Specific.
*/
class Specific : public Distribution
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
  

};

} //distribution
} //korali
;
