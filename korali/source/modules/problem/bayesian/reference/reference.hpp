/** \namespace bayesian
* @brief Namespace declaration for modules of type: bayesian.
*/

/** \file
* @brief Header file for module: Reference.
*/

/** \dir problem/bayesian/reference
* @brief Contains code, documentation, and scripts for module: Reference.
*/

#pragma once

#include "modules/problem/bayesian/bayesian.hpp"
#include <vector>

namespace korali
{
namespace problem
{
namespace bayesian
{
;

/**
* @brief Class declaration for module: Reference.
*/
class Reference : public Bayesian
{
  private:
  const double _log2pi = 1.83787706640934533908193770912476;

  /**
   * @brief Precomputes the square distance between two vectors (f and y) of the same size normalized by a third vector (g)
   * @param f Vector f
   * @param g Vector g, the normalization vector
   * @param y Vector y
   * @return Normalized square distance of the vectors
   */
  double compute_normalized_sse(std::vector<double> f, std::vector<double> g, std::vector<double> y);

  /**
   * @brief An implementation of the normal likelihood y~N(f,g), where f ang g are provided by the user.
   * @param sample A Korali Sample
   */
  void loglikelihoodNormal(korali::Sample &sample);

  /**
   * @brief An implementation of the normal likelihood y~N(f,g) truncated at zero, where f ang g are provided by the user.
   * @param sample A Korali Sample
   */
  void loglikelihoodPositiveNormal(korali::Sample &sample);

  /**
   * @brief An implementation of the student's t loglikelihood y~T(v), where v>0 (degrees of freedom) is provided by the user.
   * @param sample A Korali Sample
   */
  void loglikelihoodStudentT(korali::Sample &sample);

  /**
   * @brief An implementation of the student's t loglikelihood y~T(v) truncated at zero, where v>0 (degrees of freedom) is provided by the user.
   * @param sample A Korali Sample
   */
  void loglikelihoodPositiveStudentT(Sample &sample);

  /**
   * @brief Poisson likelihood parametrized by mean.
   * @param sample A Korali Sample
   */
  void loglikelihoodPoisson(korali::Sample &sample);

  /**
   * @brief Geometric likelihood parametrized by mean. Parametrization of number of trials before success used.
   * @param sample A Korali Sample
   */
  void loglikelihoodGeometric(korali::Sample &sample);

  /**
   * @brief Negative Binomial likelihood parametrized by mean and dispersion.
   * @param sample A Korali Sample
   */
  void loglikelihoodNegativeBinomial(korali::Sample &sample);

  /**
   * @brief Calculates the gradient of the Normal loglikelihood model.
   * @param sample A Korali Sample
   */
  void gradientLoglikelihoodNormal(korali::Sample &sample);

  /**
   * @brief Calculates the gradient of the Positive Normal (truncated at 0) loglikelihood model.
   * @param sample A Korali Sample
   */
  void gradientLoglikelihoodPositiveNormal(korali::Sample &sample);

  /**
   * @brief Calculates the gradient of the Negative Binomial loglikelihood model.
   * @param sample A Korali Sample
   */
  void gradientLoglikelihoodNegativeBinomial(korali::Sample &sample);

  /**
   * @brief Calculates the Hessian of the Normal logLikelihood model.
   * @param sample A Korali Sample
   */
  void hessianLogLikelihoodNormal(korali::Sample &sample);

  /**
   * @brief Calculates the Hessian of the Positive Normal logLikelihood model.
   * @param sample A Korali Sample
   */
  void hessianLogLikelihoodPositiveNormal(korali::Sample &sample);

  /**
   * @brief Calculates the Hessian of the Negative Binomial logLikelihood model.
   * @param sample A Korali Sample
   */
  void hessianLogLikelihoodNegativeBinomial(korali::Sample &sample);

  /**
   * @brief Calculates the Fisher information matrix of the Normal likelihood model.
   * @param sample A Korali Sample
   */
  void fisherInformationLoglikelihoodNormal(korali::Sample &sample);

  /**
   * @brief Calculates the Fisher information matrix of the Positive Normal (truncated at 0) likelihood model.
   * @param sample A Korali Sample
   */
  void fisherInformationLoglikelihoodPositiveNormal(korali::Sample &sample);

  /**
   * @brief Calculates the Fisher information matrix of the Negative Binomial likelihood model.
   * @param sample A Korali Sample
   */
  void fisherInformationLoglikelihoodNegativeBinomial(korali::Sample &sample);

  public: 
  /**
  * @brief Stores the computational model. It should the evaluation of the model at the given reference data points.
  */
   std::uint64_t _computationalModel;
  /**
  * @brief Reference data required to calculate likelihood. Model evaluations are compared against these data.
  */
   std::vector<double> _referenceData;
  /**
  * @brief Specifies the likelihood model to approximate the reference data to.
  */
   std::string _likelihoodModel;
  
 
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
  

  void initialize() override;
  void evaluateLoglikelihood(korali::Sample &sample) override;
  void evaluateLoglikelihoodGradient(korali::Sample &sample) override;
  void evaluateLogLikelihoodHessian(korali::Sample &sample) override;
  void evaluateFisherInformation(korali::Sample &sample) override;
};

} //bayesian
} //problem
} //korali
;
