/** \namespace hierarchical
* @brief Namespace declaration for modules of type: hierarchical.
*/

/** \file
* @brief Header file for module: Theta.
*/

/** \dir problem/hierarchical/theta
* @brief Contains code, documentation, and scripts for module: Theta.
*/

#pragma once

#include "modules/problem/bayesian/bayesian.hpp"
#include "modules/problem/hierarchical/psi/psi.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

/**
* @brief Class declaration for module: Theta.
*/
class Theta : public Hierarchical
{
  private:
  /**
   * @brief Stores the actual Korali object for the psi experiment
   */
  korali::Experiment _psiExperimentObject;

  /**
   * @brief Stores the actual Korali object for the theta experiment
   */
  korali::Experiment _subExperimentObject;

  /**
   * @brief Stores the number of variables defined in the Psi problem
   */
  size_t _psiVariableCount;

  /**
   * @brief Stores the number of samples in the Psi problem experiment to use as input
   */
  size_t _psiProblemSampleCount;

  /**
   * @brief Stores the sample coordinates of the Psi Problem
   */
  std::vector<std::vector<double>> _psiProblemSampleCoordinates;

  /**
   * @brief Stores the sample logLikelihoods of the Psi Problem
   */
  std::vector<double> _psiProblemSampleLogLikelihoods;

  /**
   * @brief Stores the sample logPriors of the Psi Problem
   */
  std::vector<double> _psiProblemSampleLogPriors;

  /**
   * @brief Stores the Problem module of the Psi problem experiment to use as input
   */
  korali::problem::hierarchical::Psi *_psiProblem;

  /**
   * @brief Stores the number of variables defined in the Sub problem
   */
  size_t _subProblemVariableCount;

  /**
   * @brief Stores the number of samples in the sub problem experiment to use as input
   */
  size_t _subProblemSampleCount;

  /**
   * @brief Stores the sample coordinates of the sub Problem
   */
  std::vector<std::vector<double>> _subProblemSampleCoordinates;

  /**
   * @brief Stores the sample logLikelihoods of the sub Problem
   */
  std::vector<double> _subProblemSampleLogLikelihoods;

  /**
   * @brief Stores the sample logPriors of the sub Problem
   */
  std::vector<double> _subProblemSampleLogPriors;

  /**
   * @brief Stores the precomputed log denomitator to speed up calculations
   */
  std::vector<double> _precomputedLogDenominator;

  public: 
  /**
  * @brief Results from one previously executed Bayesian experiment.
  */
   knlohmann::json _subExperiment;
  /**
  * @brief Results from the hierarchical problem (Psi).
  */
   knlohmann::json _psiExperiment;
  
 
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
  

  void evaluateLogLikelihood(korali::Sample &sample) override;
  void initialize() override;
};

} //hierarchical
} //problem
} //korali
;
