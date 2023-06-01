/** \namespace hierarchical
* @brief Namespace declaration for modules of type: hierarchical.
*/

/** \file
* @brief Header file for module: ThetaNew.
*/

/** \dir problem/hierarchical/thetaNew
* @brief Contains code, documentation, and scripts for module: ThetaNew.
*/

#pragma once

#include "modules/problem/hierarchical/hierarchical.hpp"
#include "modules/problem/hierarchical/psi/psi.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

/**
* @brief Class declaration for module: ThetaNew.
*/
class ThetaNew : public Hierarchical
{
  private:
  /**
   * @brief Stores the actual Korali object for the psi experiment
   */
  korali::Experiment _psiExperimentObject;

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

  public: 
  /**
  * @brief Results from the hierarchical Psi experiment.
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

  /**
   * @brief Evaluates the theta log likelihood of the given sample.
   * @param sample A Korali Sample
   */
  void evaluateThetaLikelihood(korali::Sample &sample);
  void initialize() override;
};

} //hierarchical
} //problem
} //korali
;
