/** \namespace problem
* @brief Namespace declaration for modules of type: problem.
*/

/** \file
* @brief Header file for module: Design.
*/

/** \dir problem/design
* @brief Contains code, documentation, and scripts for module: Design.
*/

#pragma once

#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
;

/**
* @brief Class declaration for module: Design.
*/
class Design : public Problem
{
  public: 
  /**
  * @brief Stores the model function.
  */
   std::uint64_t _model;
  /**
  * @brief [Internal Use] Stores the dimension of the parameter space.
  */
   size_t _parameterVectorSize;
  /**
  * @brief [Internal Use] Stores the dimension of the design space.
  */
   size_t _designVectorSize;
  /**
  * @brief [Internal Use] Stores the dimension of the design space.
  */
   size_t _measurementVectorSize;
  /**
  * @brief [Internal Use] Stores the indexes of the variables that constitute the parameter vector.
  */
   std::vector<size_t> _parameterVectorIndexes;
  /**
  * @brief [Internal Use] Stores the indexes of the variables that constitute the design vector.
  */
   std::vector<size_t> _designVectorIndexes;
  /**
  * @brief [Internal Use] Stores the indexes of the variables that constitute the design vector.
  */
   std::vector<size_t> _measurementVectorIndexes;
  
 
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

  /**
   * @brief Evaluates the model for a given sample from the prior distribution
   * @param sample A Korali Sample
   */
  void runModel(Sample &sample);
};

} //problem
} //korali
;
