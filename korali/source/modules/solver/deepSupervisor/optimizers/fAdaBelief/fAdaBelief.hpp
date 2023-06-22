/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: fAdaBelief.
*/

/** \dir solver/deepSupervisor/optimizers/fAdaBelief
* @brief Contains code, documentation, and scripts for module: fAdaBelief.
*/

#pragma once

#include "modules/solver/deepSupervisor/optimizers/fAdam/fAdam.hpp"
#include "modules/solver/deepSupervisor/optimizers/fGradientBasedOptimizer.hpp"

namespace korali
{
;

/**
* @brief Class declaration for module: fAdaBelief.
*/
class fAdaBelief : public fGradientBasedOptimizer
{
  public: 
  /**
  * @brief Term to guard agains numerical instability.
  */
   float _beta1;
  /**
  * @brief Term to guard agains numerical instability.
  */
   float _beta2;
  /**
  * @brief [Internal Use] First running powers of beta_1^t.
  */
   float _beta1Pow;
  /**
  * @brief [Internal Use] Second running powers of beta_2^t.
  */
   float _beta2Pow;
  /**
  * @brief [Internal Use] First moment of Gradient.
  */
   std::vector<double> _firstMoment;
  /**
  * @brief [Internal Use] Second central moment.
  */
   std::vector<double> _secondCentralMoment;
  
 
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
