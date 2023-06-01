/** \namespace integrator
* @brief Namespace declaration for modules of type: integrator.
*/

/** \file
* @brief Header file for module: MonteCarlo.
*/

/** \dir solver/integrator/montecarlo
* @brief Contains code, documentation, and scripts for module: MonteCarlo.
*/

#pragma once

#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/integrator/integrator.hpp"
#include "modules/solver/solver.hpp"

namespace korali
{
namespace solver
{
namespace integrator
{
;

/**
* @brief Class declaration for module: MonteCarlo.
*/
class MonteCarlo : public Integrator
{
  public: 
  /**
  * @brief Specifies the number of randomly generated parameter to evaluate.
  */
   size_t _numberOfSamples;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  
 
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
  

  void launchSample(size_t sampleIndex) override;
  void setInitialConfiguration() override;
};

} //integrator
} //solver
} //korali
;
