/** \namespace univariate
* @brief Namespace declaration for modules of type: univariate.
*/

/** \file
* @brief Header file for module: Normal.
*/

/** \dir distribution/univariate/normal
* @brief Contains code, documentation, and scripts for module: Normal.
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
* @brief Class declaration for module: Normal.
*/
class Normal : public Univariate
{
  private:
  double _normalization;
  double _logNormalization;

  public: 
  /**
  * @brief [Conditional Variable Value] The mean of the Normal (Gaussian) distribution.
  */
   double _mean;
  /**
  * @brief [Conditional Variable Reference] The mean of the Normal (Gaussian) distribution.
  */
   std::string _meanConditional;
  /**
  * @brief [Conditional Variable Value] The standard deviation of the Normal (Gaussian) distribution.
  */
   double _standardDeviation;
  /**
  * @brief [Conditional Variable Reference] The standard deviation of the Normal (Gaussian) distribution.
  */
   std::string _standardDeviationConditional;
  
 
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
  * @brief Updates distribution with new parameter (here standard deviation).
    Call this after changing a distribution's parameter, for example:
        _myGenerator->standardDeviation = newValue;
        _myGenerator->updateDistribution();
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
