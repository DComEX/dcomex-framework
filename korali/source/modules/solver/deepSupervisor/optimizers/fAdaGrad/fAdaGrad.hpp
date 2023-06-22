/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: fAdaGrad.
*/

/** \dir solver/deepSupervisor/optimizers/fAdaGrad
* @brief Contains code, documentation, and scripts for module: fAdaGrad.
*/

#pragma once

#include "modules/solver/deepSupervisor/optimizers/fGradientBasedOptimizer.hpp"

namespace korali
{
;

/**
* @brief Class declaration for module: fAdaGrad.
*/
class fAdaGrad : public fGradientBasedOptimizer
{
  public: 
  /**
  * @brief [Internal Use] Digaonal sum of the outer products of the gradients diag(gg^T)
  */
   std::vector<float> _gdiag;
  
 
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
