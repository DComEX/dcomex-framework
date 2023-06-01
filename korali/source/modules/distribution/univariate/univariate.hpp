/** \namespace distribution
* @brief Namespace declaration for modules of type: distribution.
*/

/** \file
* @brief Header file for module: Univariate.
*/

/** \dir distribution/univariate
* @brief Contains code, documentation, and scripts for module: Univariate.
*/

#pragma once

#include "modules/distribution/distribution.hpp"

namespace korali
{
namespace distribution
{
;

/**
* @brief Class declaration for module: Univariate.
*/
class Univariate : public Distribution
{
  public: 
  
 
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
   * @brief Gets the probability density of the distribution at point x.
   * @param x point to evaluate P(x)
   * @return Value of the probability density.
   */
  virtual double getDensity(const double x) const = 0;

  /**
   * @brief Gets the log probability density of the distribution at point x.
   * @param x point to evaluate log(P(x))
   * @return Log of probability density.
   */
  virtual double getLogDensity(const double x) const = 0;

  /**
   * @brief Gets the gradient of the log probability density of the distribution wrt. to x.
   * @param x point to evaluate grad(log(P(x)))
   * @return Gradient of log of probability density.
   */
  virtual double getLogDensityGradient(const double x) const { KORALI_LOG_ERROR("Gradient for prior not yet implemented\n"); };

  /**
   * @brief Gets the second derivative of the log probability density of the distribution wrt. to x.
   * @param x point to evaluate H(log(P(x)))
   * @return Hessian of log of probability density.
   */
  virtual double getLogDensityHessian(const double x) const { KORALI_LOG_ERROR("Hessian for prior not yet implemented\n"); };

  /**
   * @brief Draws and returns a random number from the distribution.
   * @return Random real number.
   */
  virtual double getRandomNumber() = 0;
};

} //distribution
} //korali
;
