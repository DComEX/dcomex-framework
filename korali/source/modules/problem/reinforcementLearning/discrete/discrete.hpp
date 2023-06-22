/** \namespace reinforcementLearning
* @brief Namespace declaration for modules of type: reinforcementLearning.
*/

/** \file
* @brief Header file for module: Discrete.
*/

/** \dir problem/reinforcementLearning/discrete
* @brief Contains code, documentation, and scripts for module: Discrete.
*/

#pragma once

#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"

namespace korali
{
namespace problem
{
namespace reinforcementLearning
{
;

/**
* @brief Class declaration for module: Discrete.
*/
class Discrete : public ReinforcementLearning
{
  public: 
  /**
  * @brief The set of all possible actions.
  */
   std::vector<std::vector<float>> _possibleActions;
  
 
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
  * @brief Runs the operation specified on the given sample. It checks recursively whether the function was found by the current module or its parents.
  * @param sample Sample to operate on. Should contain in the 'Operation' field an operation accepted by this module or its parents.
  * @param operation Should specify an operation type accepted by this module or its parents.
  * @return True, if operation found and executed; false, otherwise.
  */
  bool runOperation(std::string operation, korali::Sample& sample) override;
  

  void initialize() override;
};

} //reinforcementLearning
} //problem
} //korali
;
