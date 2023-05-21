/** \namespace specific
* @brief Namespace declaration for modules of type: specific.
*/

/** \file
* @brief Header file for module: Multinomial.
*/

/** \dir distribution/specific/multinomial
* @brief Contains code, documentation, and scripts for module: Multinomial.
*/

#pragma once

#include "modules/distribution/specific/specific.hpp"

namespace korali
{
namespace distribution
{
namespace specific
{
;

/**
* @brief Class declaration for module: Multinomial.
*/
class Multinomial : public Specific
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
   * @brief This function computes a random sample from the multinomial distribution.
   * @param p Underlying probability distributions
   * @param n Random sample to draw
   * @param N Number of trials
   */
  void getSelections(std::vector<double> &p, std::vector<unsigned int> &n, int N);
};

} //specific
} //distribution
} //korali
;
