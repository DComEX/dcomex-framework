/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: GridSearch.
*/

/** \dir solver/optimizer/gridSearch
* @brief Contains code, documentation, and scripts for module: GridSearch.
*/

#pragma once

#include "modules/solver/optimizer/optimizer.hpp"

namespace korali
{
namespace solver
{
namespace optimizer
{
;

/**
* @brief Class declaration for module: GridSearch.
*/
class GridSearch : public Optimizer
{
  public: 
  /**
  * @brief [Internal Use] Total number of parameter to evaluate (samples per generation).
  */
   size_t _numberOfValues;
  /**
  * @brief [Internal Use] Vector containing values of the objective function.
  */
   std::vector<double> _objective;
  /**
  * @brief [Internal Use] Holds helper to calculate cartesian indices from linear index.
  */
   std::vector<size_t> _indexHelper;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
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
  

  void finalize() override;
  void setInitialConfiguration() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;
};

} //optimizer
} //solver
} //korali
;
