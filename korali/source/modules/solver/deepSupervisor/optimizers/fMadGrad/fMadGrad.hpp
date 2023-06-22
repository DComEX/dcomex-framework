/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: fMadGrad.
*/

/** \dir solver/deepSupervisor/optimizers/fMadGrad
* @brief Contains code, documentation, and scripts for module: fMadGrad.
*/

#pragma once

#include "modules/solver/deepSupervisor/optimizers/fGradientBasedOptimizer.hpp"

namespace korali
{
;

/**
* @brief Class declaration for module: fMadGrad.
*/
class fMadGrad : public fGradientBasedOptimizer
{
  public: 
  /**
  * @brief [Internal Use] Intitial value x0, currently set to 0.
  */
   std::vector<float> _initialValue;
  /**
  * @brief [Internal Use] Scaled gradient sum.
  */
   std::vector<float> _s;
  /**
  * @brief [Internal Use] Scaled digaonal sum of the outer products of the gradients diag(gg^T).
  */
   std::vector<float> _v;
  /**
  * @brief [Internal Use] Update rule.
  */
   std::vector<float> _z;
  /**
  * @brief [Internal Use] Momentum to be used.
  */
   float _momentum;
  
 
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
  

  virtual void initialize() override;
  virtual void processResult(std::vector<float> &gradient) override;
  virtual void reset() override;
  virtual void printInternals() override;
};

} //korali
;
