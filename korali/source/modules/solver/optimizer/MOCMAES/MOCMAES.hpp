/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: MOCMAES.
*/

/** \dir solver/optimizer/MOCMAES
* @brief Contains code, documentation, and scripts for module: MOCMAES.
*/

#pragma once

#include "modules/distribution/multivariate/normal/normal.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/optimizer/optimizer.hpp"
#include <vector>

namespace korali
{
namespace solver
{
namespace optimizer
{
;

/**
* @brief Class declaration for module: MOCMAES.
*/
class MOCMAES : public Optimizer
{
  public: 
  /**
  * @brief Specifies the number of samples to evaluate per generation (preferably $4+3*log(N)$, where $N$ is the number of variables).
  */
   size_t _populationSize;
  /**
  * @brief Number of best samples (offspring) advancing to the next generation (by default it is half the Sample Count).
  */
   size_t _muValue;
  /**
  * @brief Controls the learning rate of the conjugate evolution path (must be in (0,1], by default this variable is internally calibrated, variable Cc in reference).
  */
   double _evolutionPathAdaptionStrength;
  /**
  * @brief Controls the learning rate of the covariance matrices (must be in (0,1], by default this variable is internally calibrated, variable Ccov in reference).
  */
   double _covarianceLearningRate;
  /**
  * @brief Value that controls the updates of the covariance matrix and the evolution path (must be in (0,1], variable Psucc in reference).
  */
   double _targetSuccessRate;
  /**
  * @brief Threshold that defines update scheme for the covariance matrix and the evolution path (must be in (0,1], variable Pthresh in reference).
  */
   double _thresholdProbability;
  /**
  * @brief Learning Rate of success rates (must be in (0,1], by default this variable is internally calibrated, variable Cp in reference).
  */
   double _successLearningRate;
  /**
  * @brief [Internal Use] The number of objective functions to optimize.
  */
   size_t _numObjectives;
  /**
  * @brief [Internal Use] Multinormal random number generator.
  */
   korali::distribution::multivariate::Normal* _multinormalGenerator;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] Number of non dominated samples of current generation.
  */
   size_t _currentNonDominatedSampleCount;
  /**
  * @brief [Internal Use] Objective function values.
  */
   std::vector<std::vector<double>> _currentValues;
  /**
  * @brief [Internal Use] Objective function values from previous generation.
  */
   std::vector<std::vector<double>> _previousValues;
  /**
  * @brief [Internal Use] Tracking index of parent samples.
  */
   std::vector<size_t> _parentIndex;
  /**
  * @brief [Internal Use] Sample coordinate information of parents.
  */
   std::vector<std::vector<double>> _parentSamplePopulation;
  /**
  * @brief [Internal Use] Sample coordinate information.
  */
   std::vector<std::vector<double>> _currentSamplePopulation;
  /**
  * @brief [Internal Use] Sample coordinate information of previous offsprint.
  */
   std::vector<std::vector<double>> _previousSamplePopulation;
  /**
  * @brief [Internal Use] Step size of parent.
  */
   std::vector<double> _parentSigma;
  /**
  * @brief [Internal Use] Determines the step size.
  */
   std::vector<double> _currentSigma;
  /**
  * @brief [Internal Use] Previous step size.
  */
   std::vector<double> _previousSigma;
  /**
  * @brief [Internal Use] (Unscaled) covariance matrices of parents.
  */
   std::vector<std::vector<double>> _parentCovarianceMatrix;
  /**
  * @brief [Internal Use] (Unscaled) covariance matrices of proposal distributions.
  */
   std::vector<std::vector<double>> _currentCovarianceMatrix;
  /**
  * @brief [Internal Use] (Unscaled) covariance matrices of proposal distributions from previous offspring.
  */
   std::vector<std::vector<double>> _previousCovarianceMatrix;
  /**
  * @brief [Internal Use] Evolution path of parents.
  */
   std::vector<std::vector<double>> _parentEvolutionPaths;
  /**
  * @brief [Internal Use] Evolution path of samples.
  */
   std::vector<std::vector<double>> _currentEvolutionPaths;
  /**
  * @brief [Internal Use] Evolution path of samples of previous offspring.
  */
   std::vector<std::vector<double>> _previousEvolutionPaths;
  /**
  * @brief [Internal Use] Smoothed success probabilities of parents.
  */
   std::vector<double> _parentSuccessProbabilities;
  /**
  * @brief [Internal Use] Smoothed success probabilities.
  */
   std::vector<double> _currentSuccessProbabilities;
  /**
  * @brief [Internal Use] Smoothed success probabilities of previous generation.
  */
   std::vector<double> _previousSuccessProbabilities;
  /**
  * @brief [Internal Use] Counter of evaluated samples to terminate evaluation.
  */
   size_t _finishedSampleCount;
  /**
  * @brief [Internal Use] Best value of each objective.
  */
   std::vector<double> _bestEverValues;
  /**
  * @brief [Internal Use] Samples associated with  best ever objective values.
  */
   std::vector<std::vector<double>> _bestEverVariablesVector;
  /**
  * @brief [Internal Use] Best objectives from previous generation.
  */
   std::vector<double> _previousBestValues;
  /**
  * @brief [Internal Use] Samples associated with previous best objective values.
  */
   std::vector<std::vector<double>> _previousBestVariablesVector;
  /**
  * @brief [Internal Use] Best objectives from current generation.
  */
   std::vector<double> _currentBestValues;
  /**
  * @brief [Internal Use] Samples associated with current best objective values.
  */
   std::vector<std::vector<double>> _currentBestVariablesVector;
  /**
  * @brief [Internal Use] Candidate pareto optimal samples. Samples will be finalized at termination.
  */
   std::vector<std::vector<double>> _sampleCollection;
  /**
  * @brief [Internal Use] Model evaluations of pareto candidates.
  */
   std::vector<std::vector<double>> _sampleValueCollection;
  /**
  * @brief [Internal Use] Keeps count of the number of infeasible samples.
  */
   size_t _infeasibleSampleCount;
  /**
  * @brief [Internal Use] Value differences of current and previous best values found.
  */
   std::vector<double> _currentBestValueDifferences;
  /**
  * @brief [Internal Use] L2 norm of previous and current best variable for each objective.
  */
   std::vector<double> _currentBestVariableDifferences;
  /**
  * @brief [Internal Use] Current minimum of any standard devs of a sample.
  */
   std::vector<double> _currentMinStandardDeviations;
  /**
  * @brief [Internal Use] Current maximum of any standard devs of a sample.
  */
   std::vector<double> _currentMaxStandardDeviations;
  /**
  * @brief [Termination Criteria] Specifies the min max fitness differential between two consecutive generations before stopping execution.
  */
   double _minMaxValueDifferenceThreshold;
  /**
  * @brief [Termination Criteria] Specifies the min L2 norm of the best samples between two consecutive generations before stopping execution.
  */
   double _minVariableDifferenceThreshold;
  /**
  * @brief [Termination Criteria] Specifies the minimal standard deviation.
  */
   double _minStandardDeviation;
  /**
  * @brief [Termination Criteria] Specifies the maximal standard deviation.
  */
   double _maxStandardDeviation;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
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
   * @brief Prepares generation for the next set of evaluations
   */
  void prepareGeneration();

  /**
   * @brief Evaluates a single sample
   * @param sampleIdx Index of the sample to evaluate
   */
  void sampleSingle(size_t sampleIdx);

  /**
   * @brief Sort sample indeces based on non-dominance (primary) and contribution and contributing hypervolume (secondary).
   * @param values Values to sort
   * @return sorted indices
   */
  std::vector<int> sortSampleIndices(const std::vector<std::vector<double>> &values) const;

  /**
   * @brief Updates mean and covariance of Gaussian proposal distribution.
   */
  void updateDistribution();

  /**
   * @brief Update statistics mostly for analysis.
   */
  void updateStatistics();

  /**
   * @brief Configures CMA-ES.
   */
  void setInitialConfiguration() override;

  /**
   * @brief Executes sampling & evaluation generation.
   */
  void runGeneration() override;

  /**
   * @brief Console Output before generation runs.
   */
  void printGenerationBefore() override;

  /**
   * @brief Console output after generation.
   */
  void printGenerationAfter() override;

  /**
   * @brief Final console output at termination.
   */
  void finalize() override;
};

} //optimizer
} //solver
} //korali
;
