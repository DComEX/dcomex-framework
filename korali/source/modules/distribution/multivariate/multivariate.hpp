/** \namespace distribution
* @brief Namespace declaration for modules of type: distribution.
*/

/** \file
* @brief Header file for module: Multivariate.
*/

/** \dir distribution/multivariate
* @brief Contains code, documentation, and scripts for module: Multivariate.
*/

#pragma once

#include "modules/distribution/distribution.hpp"

namespace korali
{
namespace distribution
{
;

/**
* @brief Class declaration for module: Multivariate.
*/
class Multivariate : public Distribution
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
   * @brief Updates a specific property with a vector of values.
   * @param propertyName The name of the property to update
   * @param values double Numerical values to assign.
   */
  virtual void setProperty(const std::string &propertyName, const std::vector<double> &values) = 0;

  /**
   * @brief Gets the probability density of the distribution at points x.
   * @param x points to evaluate
   * @param result P(x) at the given points
   * @param n number of points
   */
  virtual void getDensity(double *x, double *result, const size_t n) = 0;

  /**
   * @brief Gets Log probability density of the distribution at points x.
   * @param x points to evaluate
   * @param result log(P(x)) at the given points
   * @param n number of points
   */
  virtual void getLogDensity(double *x, double *result, const size_t n) = 0;

  /**
   * @brief Draws and returns a random number vector from the distribution.
   * @param x Random real number vector.
   * @param n Vector size
   */
  virtual void getRandomVector(double *x, const size_t n) = 0;
};

} //distribution
} //korali
;
