/** \namespace hierarchical
* @brief Namespace declaration for modules of type: hierarchical.
*/

/** \file
* @brief Header file for module: Psi.
*/

/** \dir problem/hierarchical/psi
* @brief Contains code, documentation, and scripts for module: Psi.
*/

#pragma once

#include "modules/distribution/distribution.hpp"
#include "modules/problem/hierarchical/hierarchical.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{
;

/**
* @brief Class declaration for module: Psi.
*/
class Psi : public Hierarchical
{
  private:
  /**
   * @brief Stores the pre-computed positions (pointers) of the conditional priors to evaluate for performance
   */
  struct conditionalPriorInfo
  {
    /**
     * @brief Stores the position of the conditional prior
     */
    std::vector<size_t> _samplePositions;

    /**
     * @brief Stores the pointer of the conditional prior
     */
    std::vector<double *> _samplePointers;
  };

  /**
   * @brief Stores the number of subproblems
   */
  size_t _subProblemsCount;

  /**
   * @brief Stores the number of variables in the subproblems (all must be the same)
   */
  size_t _subProblemsVariablesCount;

  /**
   * @brief Stores the sample coordinates of all the subproblems
   */
  std::vector<std::vector<std::vector<double>>> _subProblemsSampleCoordinates;

  /**
   * @brief Stores the sample logLikelihoods of all the subproblems
   */
  std::vector<std::vector<double>> _subProblemsSampleLogLikelihoods;

  /**
   * @brief Stores the sample logPriors of all the subproblems
   */
  std::vector<std::vector<double>> _subProblemsSampleLogPriors;

  /**
   * @brief Stores the precomputed conditional prior information, for performance
   */
  std::vector<conditionalPriorInfo> _conditionalPriorInfos;

  public: 
  /**
  * @brief Provides results from previous Bayesian Inference sampling experiments.
  */
   std::vector<knlohmann::json> _subExperiments;
  /**
  * @brief List of conditional priors to use in the hierarchical problem.
  */
   std::vector<std::string> _conditionalPriors;
  
 
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
   * @brief Stores the indexes of conditional priors to Experiment variables
   */
  std::vector<size_t> _conditionalPriorIndexes;

  /**
   * @brief Updates the distribution parameters for the conditional priors, given variable values in the sample.
   * @param sample A Korali Sample
   */
  void updateConditionalPriors(korali::Sample &sample);
  void evaluateLogLikelihood(korali::Sample &sample) override;
  void initialize() override;
};

} //hierarchical
} //problem
} //korali
;
