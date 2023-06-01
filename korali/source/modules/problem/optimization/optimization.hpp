/** \namespace problem
* @brief Namespace declaration for modules of type: problem.
*/

/** \file
* @brief Header file for module: Optimization.
*/

/** \dir problem/optimization
* @brief Contains code, documentation, and scripts for module: Optimization.
*/

#pragma once

#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
;

/**
* @brief Class declaration for module: Optimization.
*/
class Optimization : public Problem
{
  private:
  public: 
  /**
  * @brief Number of return values to expect from objective function.
  */
   size_t _numObjectives;
  /**
  * @brief Stores the function to evaluate.
  */
   std::uint64_t _objectiveFunction;
  /**
  * @brief Stores constraints to the objective function.
  */
   std::vector<std::uint64_t> _constraints;
  /**
  * @brief [Internal Use] Flag indicating if at least one of the variables is discrete.
  */
   int _hasDiscreteVariables;
  
 
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
   * @brief Evaluates a single objective, given a set of parameters.
   * @param sample A sample to process
   */
  void evaluate(korali::Sample &sample);

  /**
   * @brief Evaluates multiple objectives, given a set of parameters.
   * @param sample A sample to process
   */
  void evaluateMultiple(korali::Sample &sample);

  /**
   * @brief Evaluates whether at least one of constraints have been met.
   * @param sample A Korali Sample
   */
  void evaluateConstraints(korali::Sample &sample);

  /**
   * @brief Evaluates the F(x) and Gradient(x) of a sample, given a set of parameters.
   * @param sample A sample to process
   */
  void evaluateWithGradients(korali::Sample &sample);
};

} //problem
} //korali
;
