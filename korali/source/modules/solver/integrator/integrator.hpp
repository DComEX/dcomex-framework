/** \namespace solver
* @brief Namespace declaration for modules of type: solver.
*/

/** \file
* @brief Header file for module: Integrator.
*/

/** \dir solver/integrator
* @brief Contains code, documentation, and scripts for module: Integrator.
*/

#pragma once

#include "modules/solver/solver.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
;

/**
* @brief Class declaration for module: Integrator.
*/
class Integrator : public Solver
{
  public: 
  /**
  * @brief Specifies the number of model executions per generation. By default this setting is 0, meaning that all executions will be performed in the first generation. For values greater 0, executions will be split into batches and split int generations for intermediate output.
  */
   size_t _executionsPerGeneration;
  /**
  * @brief [Internal Use] Current value of the integral.
  */
   double _accumulatedIntegral;
  /**
  * @brief [Internal Use] Gridpoints for quadrature.
  */
   std::vector<std::vector<float>> _gridPoints;
  /**
  * @brief [Internal Use] Precomputed weight for MC sample.
  */
   float _weight;
  
 
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
   * @brief Container for samples to be evaluated per generation
   */
  std::vector<Sample> _samples;

  /**
   * @brief Prepares and launches a sample to be evaluated
   * @param sampleIndex index of sample to be launched
   */
  virtual void launchSample(size_t sampleIndex) = 0;
  virtual void setInitialConfiguration() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;
  void finalize() override;
};

} //solver
} //korali
;
