/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: Rprop.
*/

/** \dir solver/optimizer/Rprop
* @brief Contains code, documentation, and scripts for module: Rprop.
*/

#pragma once

#include "modules/solver/optimizer/optimizer.hpp"
#include <vector>

namespace korali
{
namespace solver
{
namespace optimizer
{
;

/**
* @brief Class declaration for module: Rprop.
*/
class Rprop : public Optimizer
{
  private:
  void evaluateFunctionAndGradient(Sample &sample);

  void performUpdate(void); // iRprop_minus

  public: 
  /**
  * @brief Initial Delta.
  */
   double _delta0;
  /**
  * @brief Minimum Delta, parameter for step size calibration.
  */
   double _deltaMin;
  /**
  * @brief Maximum Delta, parameter for step size calibration.
  */
   double _deltaMax;
  /**
  * @brief Parameter for step size calibration.
  */
   double _etaMinus;
  /**
  * @brief Parameter for step size calibration.
  */
   double _etaPlus;
  /**
  * @brief [Internal Use] Current value of parameters.
  */
   std::vector<double> _currentVariable;
  /**
  * @brief [Internal Use] Best value of parameters.
  */
   std::vector<double> _bestEverVariable;
  /**
  * @brief [Internal Use] Gradient scaling factor
  */
   std::vector<double> _delta;
  /**
  * @brief [Internal Use] Gradient of parameters.
  */
   std::vector<double> _currentGradient;
  /**
  * @brief [Internal Use] Old gradient of parameters.
  */
   std::vector<double> _previousGradient;
  /**
  * @brief [Internal Use] Gradient of function with respect to Best Ever Variables.
  */
   std::vector<double> _bestEverGradient;
  /**
  * @brief [Internal Use] Norm of old gradient.
  */
   double _normPreviousGradient;
  /**
  * @brief [Internal Use] Counts the number the algorithm has been stalled in function evaluation bigger than the best one.
  */
   double _maxStallCounter;
  /**
  * @brief [Internal Use] Norm of variable update.
  */
   double _xDiff;
  /**
  * @brief [Termination Criteria] Maximum value of the norm of the gradient.
  */
   double _maxGradientNorm;
  /**
  * @brief [Termination Criteria] Maximum times stalled with function evaluation bigger than the best one.
  */
   size_t _maxStallGenerations;
  /**
  * @brief [Termination Criteria] Relative tolerance in parameter difference between generations.
  */
   double _parameterRelativeTolerance;
  
 
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
  

  void setInitialConfiguration() override;
  void finalize() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;
};

} //optimizer
} //solver
} //korali
;
