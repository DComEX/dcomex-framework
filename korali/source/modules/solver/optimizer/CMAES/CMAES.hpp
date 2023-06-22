/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: CMAES.
*/

/** \dir solver/optimizer/CMAES
* @brief Contains code, documentation, and scripts for module: CMAES.
*/

#pragma once

#include "modules/distribution/univariate/normal/normal.hpp"
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
* @brief Class declaration for module: CMAES.
*/
class CMAES : public Optimizer
{
  public: 
  /**
  * @brief Specifies the number of samples to evaluate per generation (preferably $4+3*log(N)$, where $N$ is the number of variables).
  */
   size_t _populationSize;
  /**
  * @brief Number of best samples (offspring samples) used to update the covariance matrix and the mean (by default it is half the Sample Count).
  */
   size_t _muValue;
  /**
  * @brief Weights given to the Mu best values to update the covariance matrix and the mean.
  */
   std::string _muType;
  /**
  * @brief Controls the learning rate of the conjugate evolution path (by default this variable is internally calibrated).
  */
   double _initialSigmaCumulationFactor;
  /**
  * @brief Controls the updates of the covariance matrix scaling factor (by default this variable is internally calibrated).
  */
   double _initialDampFactor;
  /**
  * @brief Include gradient information for proposal distribution update.
  */
   int _useGradientInformation;
  /**
  * @brief Scaling factor for gradient step, only relevant if gradient information used.
  */
   float _gradientStepSize;
  /**
  * @brief Sets an upper bound for the covariance matrix scaling factor. The upper bound is given by the average of the initial standard deviation of the variables.
  */
   int _isSigmaBounded;
  /**
  * @brief Controls the learning rate of the evolution path for the covariance update (must be in (0,1], by default this variable is internally calibrated).
  */
   double _initialCumulativeCovariance;
  /**
  * @brief Covariance matrix updates will be optimized for diagonal matrices.
  */
   int _diagonalCovariance;
  /**
  * @brief Generate the negative counterpart of each random number during sampling.
  */
   int _mirroredSampling;
  /**
  * @brief Specifies the number of samples per generation during the viability regime, i.e. during the search for a parameter vector not violating the constraints.
  */
   size_t _viabilityPopulationSize;
  /**
  * @brief Number of best samples used to update the covariance matrix and the mean during the viability regime (by default this variable is half the Viability Sample Count).
  */
   size_t _viabilityMuValue;
  /**
  * @brief Max number of covairance matrix adaptions per generation during the constraint handling loop.
  */
   size_t _maxCovarianceMatrixCorrections;
  /**
  * @brief Controls the updates of the covariance matrix scaling factor during the viability regime.
  */
   double _targetSuccessRate;
  /**
  * @brief Controls the covariane matrix adaption strength if samples violate constraints.
  */
   double _covarianceMatrixAdaptionStrength;
  /**
  * @brief Learning rate of constraint normal vectors (must be in (0, 1], by default this variable is internally calibrated).
  */
   double _normalVectorLearningRate;
  /**
  * @brief Learning rate of success probability of objective function improvements.
  */
   double _globalSuccessLearningRate;
  /**
  * @brief [Internal Use] Normal random number generator.
  */
   korali::distribution::univariate::Normal* _normalGenerator;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] True if mean is outside feasible domain. During viability regime CMA-ES is working with relaxed constraint boundaries that contract towards the true constraint boundaries.
  */
   int _isViabilityRegime;
  /**
  * @brief [Internal Use] Objective function values.
  */
   std::vector<double> _valueVector;
  /**
  * @brief [Internal Use] Gradients of objective function evaluations.
  */
   std::vector<std::vector<double>> _gradients;
  /**
  * @brief [Internal Use] Actual number of samples used per generation (Population Size or Viability Population Size).
  */
   size_t _currentPopulationSize;
  /**
  * @brief [Internal Use] Actual value of mu (Mu Value or Viability Mu Value).
  */
   size_t _currentMuValue;
  /**
  * @brief [Internal Use] Calibrated Weights for each of the Mu offspring samples.
  */
   std::vector<double> _muWeights;
  /**
  * @brief [Internal Use] Variance effective selection mass.
  */
   double _effectiveMu;
  /**
  * @brief [Internal Use] Increment for sigma, calculated from muEffective and dimension.
  */
   double _sigmaCumulationFactor;
  /**
  * @brief [Internal Use] Dampening parameter controls step size adaption.
  */
   double _dampFactor;
  /**
  * @brief [Internal Use] Controls the step size adaption.
  */
   double _cumulativeCovariance;
  /**
  * @brief [Internal Use] Expectation of $||N(0,I)||^2$.
  */
   double _chiSquareNumber;
  /**
  * @brief [Internal Use] Establishes how frequently the eigenvalues are updated.
  */
   size_t _covarianceEigenvalueEvaluationFrequency;
  /**
  * @brief [Internal Use] Determines the step size.
  */
   double _sigma;
  /**
  * @brief [Internal Use] The trace of the initial covariance matrix.
  */
   double _trace;
  /**
  * @brief [Internal Use] Sample coordinate information.
  */
   std::vector<std::vector<double>> _samplePopulation;
  /**
  * @brief [Internal Use] Counter of evaluated samples to terminate evaluation.
  */
   size_t _finishedSampleCount;
  /**
  * @brief [Internal Use] Best variables of current generation.
  */
   std::vector<double> _currentBestVariables;
  /**
  * @brief [Internal Use] Best ever model evaluation as of previous generation.
  */
   double _previousBestEverValue;
  /**
  * @brief [Internal Use] Sorted indeces of samples according to their model evaluation.
  */
   std::vector<size_t> _sortingIndex;
  /**
  * @brief [Internal Use] (Unscaled) covariance Matrix of proposal distribution.
  */
   std::vector<double> _covarianceMatrix;
  /**
  * @brief [Internal Use] Temporary Storage for Covariance Matrix.
  */
   std::vector<double> _auxiliarCovarianceMatrix;
  /**
  * @brief [Internal Use] Matrix with eigenvectors in columns.
  */
   std::vector<double> _covarianceEigenvectorMatrix;
  /**
  * @brief [Internal Use] Temporary Storage for Matrix with eigenvectors in columns.
  */
   std::vector<double> _auxiliarCovarianceEigenvectorMatrix;
  /**
  * @brief [Internal Use] Axis lengths (sqrt(Evals))
  */
   std::vector<double> _axisLengths;
  /**
  * @brief [Internal Use] Temporary storage for Axis lengths.
  */
   std::vector<double> _auxiliarAxisLengths;
  /**
  * @brief [Internal Use] Temporary storage.
  */
   std::vector<double> _bDZMatrix;
  /**
  * @brief [Internal Use] Temporary storage.
  */
   std::vector<double> _auxiliarBDZMatrix;
  /**
  * @brief [Internal Use] Current mean of proposal distribution.
  */
   std::vector<double> _currentMean;
  /**
  * @brief [Internal Use] Previous mean of proposal distribution.
  */
   std::vector<double> _previousMean;
  /**
  * @brief [Internal Use] Update differential from previous to current mean.
  */
   std::vector<double> _meanUpdate;
  /**
  * @brief [Internal Use] Evolution path for Covariance Matrix update.
  */
   std::vector<double> _evolutionPath;
  /**
  * @brief [Internal Use] Conjugate evolution path for sigma update.
  */
   std::vector<double> _conjugateEvolutionPath;
  /**
  * @brief [Internal Use] L2 Norm of the conjugate evolution path.
  */
   double _conjugateEvolutionPathL2Norm;
  /**
  * @brief [Internal Use] Maximum diagonal element of the Covariance Matrix.
  */
   double _maximumDiagonalCovarianceMatrixElement;
  /**
  * @brief [Internal Use] Minimum diagonal element of the Covariance Matrix.
  */
   double _minimumDiagonalCovarianceMatrixElement;
  /**
  * @brief [Internal Use] Maximum Covariance Matrix Eigenvalue.
  */
   double _maximumCovarianceEigenvalue;
  /**
  * @brief [Internal Use] Minimum Covariance Matrix Eigenvalue.
  */
   double _minimumCovarianceEigenvalue;
  /**
  * @brief [Internal Use] Flag determining if the covariance eigensystem is up to date.
  */
   int _isEigensystemUpdated;
  /**
  * @brief [Internal Use] Evaluation of each constraint for each sample.
  */
   std::vector<std::vector<int>> _viabilityIndicator;
  /**
  * @brief [Internal Use] True if the number of constraints is higher than zero.
  */
   int _hasConstraints;
  /**
  * @brief [Internal Use] This is the beta factor that indicates how fast the covariance matrix is adapted.
  */
   double _covarianceMatrixAdaptionFactor;
  /**
  * @brief [Internal Use] Index of best sample without constraint violations (otherwise -1).
  */
   int _bestValidSample;
  /**
  * @brief [Internal Use] Estimated Global Success Rate, required for calibration of covariance matrix scaling factor updates.
  */
   double _globalSuccessRate;
  /**
  * @brief [Internal Use] Viability Function Value.
  */
   double _viabilityFunctionValue;
  /**
  * @brief [Internal Use] Number of resampled parameters due constraint violation.
  */
   size_t _resampledParameterCount;
  /**
  * @brief [Internal Use] Number of Covariance Matrix Adaptations.
  */
   size_t _covarianceMatrixAdaptationCount;
  /**
  * @brief [Internal Use] Viability Boundaries.
  */
   std::vector<double> _viabilityBoundaries;
  /**
  * @brief [Internal Use] Sample evaluations larger than fviability.
  */
   std::vector<int> _viabilityImprovement;
  /**
  * @brief [Internal Use] Temporary counter of maximal amount of constraint violations attained by a sample (must be 0).
  */
   size_t _maxConstraintViolationCount;
  /**
  * @brief [Internal Use] Maximal amount of constraint violations.
  */
   std::vector<size_t> _sampleConstraintViolationCounts;
  /**
  * @brief [Internal Use] Functions to be evaluated as constraint evaluations, if the return from any of them is > 0, then the constraint is met.
  */
   std::vector<std::vector<double>> _constraintEvaluations;
  /**
  * @brief [Internal Use] Normal approximation of constraints.
  */
   std::vector<std::vector<double>> _normalConstraintApproximation;
  /**
  * @brief [Internal Use] Constraint evaluations for best ever.
  */
   std::vector<double> _bestConstraintEvaluations;
  /**
  * @brief [Internal Use] Flag indicating if at least one of the variables is discrete.
  */
   int _hasDiscreteVariables;
  /**
  * @brief [Internal Use] Vector storing discrete mutations, required for covariance matrix update.
  */
   std::vector<double> _discreteMutations;
  /**
  * @brief [Internal Use] Number of discrete mutations in current generation.
  */
   size_t _numberOfDiscreteMutations;
  /**
  * @brief [Internal Use] Number of nonzero entries on diagonal in Masking Matrix.
  */
   size_t _numberMaskingMatrixEntries;
  /**
  * @brief [Internal Use] Diagonal Matrix signifying where an integer mutation may be conducted.
  */
   std::vector<double> _maskingMatrix;
  /**
  * @brief [Internal Use] Sigma of the Masking Matrix.
  */
   std::vector<double> _maskingMatrixSigma;
  /**
  * @brief [Internal Use] Expectation of $||N(0,I^S)||^2$ for discrete mutations.
  */
   double _chiSquareNumberDiscreteMutations;
  /**
  * @brief [Internal Use] Current minimum standard deviation of any variable.
  */
   double _currentMinStandardDeviation;
  /**
  * @brief [Internal Use] Current maximum standard deviation of any variable.
  */
   double _currentMaxStandardDeviation;
  /**
  * @brief [Internal Use] Number of Constraint Evaluations.
  */
   size_t _constraintEvaluationCount;
  /**
  * @brief [Termination Criteria] Specifies the maximum condition of the covariance matrix.
  */
   double _maxConditionCovarianceMatrix;
  /**
  * @brief [Termination Criteria] Specifies the minimal standard deviation for any variable in any proposed sample.
  */
   double _minStandardDeviation;
  /**
  * @brief [Termination Criteria] Specifies the maximal standard deviation for any variable in any proposed sample.
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
   * @param randomNumbers Random numbers to generate sample
   */
  void sampleSingle(size_t sampleIdx, const std::vector<double> &randomNumbers);

  /**
   * @brief Adapts the covariance matrix.
   * @param hsig Sign
   */
  void adaptC(int hsig);

  /**
   * @brief Updates scaling factor of covariance matrix.
   */
  void updateSigma(); /* update Sigma */

  /**
   * @brief Updates mean and covariance of Gaussian proposal distribution.
   */
  void updateDistribution();

  /**
   * @brief Updates the system of eigenvalues and eigenvectors
   * @param M Input matrix
   */
  void updateEigensystem(const std::vector<double> &M);

  /**
   * @brief Method that checks potential numerical issues and does correction. Not yet implemented.
   */
  void numericalErrorTreatment();

  /**
   * @brief Function for eigenvalue decomposition.
   * @param N Matrix size
   * @param C Input matrix
   * @param diag Sorted eigenvalues
   * @param Q eingenvectors of C
   */
  void eigen(size_t N, const std::vector<double> &C, std::vector<double> &diag, std::vector<double> &Q) const;

  /**
   * @brief Descending sort of vector elements, stores ordering in _sortingIndex.
   * @param _sortingIndex Ordering of elements in vector
   * @param vec Vector to sort
   * @param N Number of current samples.
   */
  void sort_index(const std::vector<double> &vec, std::vector<size_t> &_sortingIndex, size_t N) const;

  /**
   * @brief Initializes the weights of the mu vector
   * @param numsamples Length of mu vector
   */
  void initMuWeights(size_t numsamples); /* init _muWeights and dependencies */

  /**
   * @brief Initialize Covariance Matrix and Cholesky Decomposition
   */
  void initCovariance(); /* init sigma, C and B */

  /**
   * @brief Check if mean of proposal distribution is inside of valid domain (does not violate constraints), if yes, re-initialize internal vars. Method for CCMA-ES.
   */
  void checkMeanAndSetRegime();

  /**
   * @brief Update constraint evaluationsa. Method for CCMA-ES.
   */
  void updateConstraints();

  /**
   * @brief Update viability boundaries. Method for CCMA-ES.
   */
  void updateViabilityBoundaries();

  /**
   * @brief Process samples that violate constraints. Method for CCMA-ES.
   */
  void handleConstraints();

  /**
   * @brief Reevaluate constraint evaluations. Called in handleConstraints. Method for CCMA-ES.
   */
  void reEvaluateConstraints();

  /**
   * @brief Update mutation matrix for discrete variables. Method for discrete/integer optimization.
   */
  void updateDiscreteMutationMatrix();

  /**
   * @brief Discretize variables to given granularity using arithmetic rounding.
   * @param sample Sample to discretize
   */
  void discretize(std::vector<double> &sample);

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
