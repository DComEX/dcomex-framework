/** \namespace integrator
* @brief Namespace declaration for modules of type: integrator.
*/

/** \file
* @brief Header file for module: Quadrature.
*/

/** \dir solver/integrator/quadrature
* @brief Contains code, documentation, and scripts for module: Quadrature.
*/

#pragma once

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
* @brief Class declaration for module: Quadrature.
*/
class Quadrature : public Integrator
{
  public: 
  /**
  * @brief The name of the quadrature rule.
  */
   std::string _method;
  /**
  * @brief [Internal Use] Holds helper to calculate cartesian indices from linear index.
  */
   std::vector<size_t> _indicesHelper;
  
 
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
