/** \namespace univariate
* @brief Namespace declaration for modules of type: univariate.
*/

/** \file
* @brief Header file for module: Geometric.
*/

/** \dir distribution/univariate/geometric
* @brief Contains code, documentation, and scripts for module: Geometric.
*/

#pragma once

#include "modules/distribution/univariate/univariate.hpp"

namespace korali
{
namespace distribution
{
namespace univariate
{
;

/**
* @brief Class declaration for module: Geometric.
*/
class Geometric : public Univariate
{
  private:
  public: 
  /**
  * @brief [Conditional Variable Value] Probability of success of an individual trial.
  */
   double _successProbability;
  /**
  * @brief [Conditional Variable Reference] Probability of success of an individual trial.
  */
   std::string _successProbabilityConditional;
  
 
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
  * @brief Retrieves the pointer of a conditional value of a distribution property.
  * @param property Name of the property to find.
  * @return The pointer to the property..
  */
  double* getPropertyPointer(const std::string& property) override;
  

  /*
   * @brief Not used.
   */
  void updateDistribution() override;

  /**
   * @brief Gets the probability density of the distribution at point x.
   * @param x point to evaluate P(x)
   * @return Value of the probability density.
   */
  double getDensity(const double x) const override;

  /**
   * @brief Gets the Log probability density of the distribution at point x.
   * @param x point to evaluate log(P(x))
   * @return Log of probability density.
   */
  double getLogDensity(const double x) const override;

  /**
   * @brief Gets the Gradient of the log probability density of the distribution wrt. to x.
   * @param x point to evaluate grad(log(P(x)))
   * @return Gradient of log of probability density.
   */
  double getLogDensityGradient(double x) const override;

  /**
   * @brief Gets the second derivative of the log probability density of the distribution wrt. to x.
   * @param x point to evaluate H(log(P(x)))
   * @return Hessian of log of probability density.
   */
  double getLogDensityHessian(double x) const override;

  /**
   * @brief Draws and returns a random number from the distribution.
   * @return Random real number.
   */
  double getRandomNumber() override;
};

} //univariate
} //distribution
} //korali
;
