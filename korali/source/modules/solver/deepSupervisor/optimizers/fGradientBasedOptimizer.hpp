/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: fGradientBasedOptimizer.
*/

/** \dir solver/deepSupervisor/optimizers
* @brief Contains code, documentation, and scripts for module: fGradientBasedOptimizer.
*/

#pragma once

#include "auxiliar/logger.hpp"
#include "modules/module.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace korali
{
;

/**
* @brief Class declaration for module: fGradientBasedOptimizer.
*/
class fGradientBasedOptimizer : public Module
{
  public: 
  /**
  * @brief Term to guard agains numerical instability.
  */
   float _epsilon;
  /**
  * @brief Size of variable space size(x) of f(x)
  */
   size_t _nVars;
  /**
  * @brief Step size/learning rate for current iterration.
  */
   float _eta;
  /**
  * @brief [Internal Use] Holds current values of the parameters.
  */
   std::vector<float> _currentValue;
  
 
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
   * @brief Takes a sample evaluation and its gradient and calculates the next set of parameters
   * @param gradient The gradient of the objective function at the current set of parameters
   */
  virtual void processResult(std::vector<float> &gradient) = 0;

  /**
   * @brief Checks size and values of gradient
   * @param gradient Gradient values to check
   */
  virtual void preProcessResult(std::vector<float> &gradient);

  /**
   * @brief Checks the result of the gradient update
   * @param parameters Parameter values to check
   */
  virtual void postProcessResult(std::vector<float> &parameters);

  /**
   * @brief Prints internals of solver
   */
  virtual void printInternals() = 0;

  /**
   * @brief Restores the optimizer to the initial state
   */
  virtual void reset() = 0;

  virtual void initialize() override;
};

} //korali
;