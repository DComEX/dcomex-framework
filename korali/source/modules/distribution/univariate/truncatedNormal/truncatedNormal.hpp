/** \namespace univariate
* @brief Namespace declaration for modules of type: univariate.
*/

/** \file
* @brief Header file for module: TruncatedNormal.
*/

/** \dir distribution/univariate/truncatedNormal
* @brief Contains code, documentation, and scripts for module: TruncatedNormal.
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
* @brief Class declaration for module: TruncatedNormal.
*/
class TruncatedNormal : public Univariate
{
  private:
  double _normalization;
  double _logNormalization;

  public: 
  /**
  * @brief [Conditional Variable Value] The mean of the untruncated Normal distribution.
  */
   double _mu;
  /**
  * @brief [Conditional Variable Reference] The mean of the untruncated Normal distribution.
  */
   std::string _muConditional;
  /**
  * @brief [Conditional Variable Value] The standard deviation of the untruncated Normal distribution.
  */
   double _sigma;
  /**
  * @brief [Conditional Variable Reference] The standard deviation of the untruncated Normal distribution.
  */
   std::string _sigmaConditional;
  /**
  * @brief [Conditional Variable Value] The lower bound of the truncated Normal distribution.
  */
   double _minimum;
  /**
  * @brief [Conditional Variable Reference] The lower bound of the truncated Normal distribution.
  */
   std::string _minimumConditional;
  /**
  * @brief [Conditional Variable Value] The upper bound of the truncated Normal distribution.
  */
   double _maximum;
  /**
  * @brief [Conditional Variable Reference] The upper bound of the truncated Normal distribution.
  */
   std::string _maximumConditional;
  
 
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
   * @brief Updates distribution with new parameter (here upper and lower boundaries, and standard deviation of a normal distribution).
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
   * @brief Gets the Gradient of the log probability density of the distribution wrt. to x.
   * @param x point to evaluate grad(log(P(x)))
   * @return Gradient of log of probability density.
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
