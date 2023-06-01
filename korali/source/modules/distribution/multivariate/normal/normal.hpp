/** \namespace multivariate
* @brief Namespace declaration for modules of type: multivariate.
*/

/** \file
* @brief Header file for module: Normal.
*/

/** \dir distribution/multivariate/normal
* @brief Contains code, documentation, and scripts for module: Normal.
*/

#pragma once

#include "modules/distribution/multivariate/multivariate.hpp"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace korali
{
namespace distribution
{
namespace multivariate
{
;

/**
* @brief Class declaration for module: Normal.
*/
class Normal : public Multivariate
{
  private:
  /**
   * @brief Temporal storage for covariance matrix
   */
  gsl_matrix_view _sigma_view;

  /**
   * @brief Temporal storage for variable means
   */
  gsl_vector_view _mean_view;

  /**
   * @brief Temporal storage for work
   */
  gsl_vector_view _work_view;

  public: 
  /**
  * @brief Means of the variables.
  */
   std::vector<double> _meanVector;
  /**
  * @brief Cholesky Decomposition of the covariance matrix.
  */
   std::vector<double> _sigma;
  /**
  * @brief [Internal Use] Auxiliary work vector.
  */
   std::vector<double> _workVector;
  
 
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
  

  /*
   * @brief Updates distribution to new covariance matrix.
   */
  void updateDistribution() override;

  /**
   * @brief Updates a specific property with a vector of values.
   * @param propertyName The name of the property to update
   * @param values double Numerical values to assign.
   */
  void setProperty(const std::string &propertyName, const std::vector<double> &values) override;

  /**
   * @brief Gets the probability density of the distribution at points x.
   * @param x points to evaluate
   * @param result P(x) at the given points
   * @param n number of points
   */
  void getDensity(double *x, double *result, const size_t n) override;

  /**
   * @brief Gets Log probability density of the distribution at points x.
   * @param x points to evaluate
   * @param result log(P(x)) at the given points
   * @param n number of points
   */
  void getLogDensity(double *x, double *result, const size_t n) override;

  /**
   * @brief Draws and returns a random number vector from the distribution.
   * @param x Random real number vector.
   * @param n Vector size
   */
  void getRandomVector(double *x, const size_t n) override;
};

} //multivariate
} //distribution
} //korali
;
