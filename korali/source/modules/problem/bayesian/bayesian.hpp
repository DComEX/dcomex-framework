/** \namespace problem
* @brief Namespace declaration for modules of type: problem.
*/

/** \file
* @brief Header file for module: Bayesian.
*/

/** \dir problem/bayesian
* @brief Contains code, documentation, and scripts for module: Bayesian.
*/

#pragma once

#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
;

/**
* @brief Class declaration for module: Bayesian.
*/
class Bayesian : public Problem
{
  private:
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
  * @brief Runs the operation specified on the given sample. It checks recursively whether the function was found by the current module or its parents.
  * @param sample Sample to operate on. Should contain in the 'Operation' field an operation accepted by this module or its parents.
  * @param operation Should specify an operation type accepted by this module or its parents.
  * @return True, if operation found and executed; false, otherwise.
  */
  bool runOperation(std::string operation, korali::Sample& sample) override;
  

  void initialize() override;

  /**
   * @brief Produces a generic evaluation from the Posterior distribution of the sample, for optimization with CMAES, DEA, storing it in and stores it in sample["F(x)"].
   * @param sample A Korali Sample
   */
  virtual void evaluate(korali::Sample &sample);

  /**
   * @brief Evaluates the log prior of the given sample, and stores it in sample["Log Prior"]
   * @param sample A Korali Sample
   */
  void evaluateLogPrior(korali::Sample &sample);

  /**
   * @brief Evaluates the log likelihood of the given sample, and stores it in sample["Log Likelihood"]
   * @param sample A Korali Sample
   */
  virtual void evaluateLoglikelihood(korali::Sample &sample) = 0;

  /**
   * @brief Evaluates the log posterior of the given sample, and stores it in sample["Log Posterior"]
   * @param sample A Korali Sample
   */
  void evaluateLogPosterior(korali::Sample &sample);

  /**
   * @brief Evaluates the gradient of the objective w.r.t. to the variables, and stores it in sample["Gradient"]
   * @param sample A Korali Sample
   */
  virtual void evaluateGradient(korali::Sample &sample);

  /**
   * @brief Evaluates the gradient of the logPrior w.r.t. to the variables, and stores it in sample["logPrior Gradient"]
   * @param sample A Korali Sample
   */
  void evaluateLogPriorGradient(korali::Sample &sample);

  /**
   * @brief Evaluates the gradient of the logLikelikood w.r.t. to the variables, and stores it in sample["logLikelihood Gradient"]
   * @param sample A Korali Sample
   */
  virtual void evaluateLoglikelihoodGradient(korali::Sample &sample) { KORALI_LOG_ERROR("Gradient not yet implemented for selected Bayesian problem and logLikelihood model."); }

  /**
   * @brief Evaluates the hessian of the objective w.r.t. to the variables, and stores it in sample["Hessian"]
   * @param sample A Korali Sample
   */
  virtual void evaluateHessian(korali::Sample &sample);

  /**
   * @brief Evaluates the gradient of the logPrior w.r.t. to the variables, and stores it in sample["logPrior Hessian"]
   * @param sample A Korali Sample
   */
  void evaluateLogPriorHessian(korali::Sample &sample);

  /**
   * @brief Evaluates the gradient of the logLikelikood w.r.t. to the variables, and stores it in sample["logLikelihood Hessian"]
   * @param sample A Korali Sample
   */
  virtual void evaluateLogLikelihoodHessian(korali::Sample &sample) { KORALI_LOG_ERROR("Hessian not yet implemented for selected Bayesian problem and logLikelihood model."); }

  /**
   * @brief Evaluates the empirical Fisher information.
   * @param sample A Korali Sample
   */
  virtual void evaluateFisherInformation(korali::Sample &sample) { KORALI_LOG_ERROR("Fisher information not yet implemented for selected Bayesian problem and log likelihood model."); }
};

} //problem
} //korali
;
