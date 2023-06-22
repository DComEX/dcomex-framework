/** \namespace sampler
* @brief Namespace declaration for modules of type: sampler.
*/

/** \file
* @brief Header file for module: HMC.
*/

/** \dir solver/sampler/HMC
* @brief Contains code, documentation, and scripts for module: HMC.
*/

#pragma once

// TODO: REMOVE normal/normal.hpp
#include "modules/distribution/multivariate/normal/normal.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/sampler/sampler.hpp"
#include <string>
#include <vector>

#include "modules/solver/sampler/HMC/helpers/hamiltonian_euclidean_dense.hpp"
#include "modules/solver/sampler/HMC/helpers/hamiltonian_euclidean_diag.hpp"
#include "modules/solver/sampler/HMC/helpers/hamiltonian_riemannian_const_dense.hpp"
#include "modules/solver/sampler/HMC/helpers/hamiltonian_riemannian_const_diag.hpp"
#include "modules/solver/sampler/HMC/helpers/hamiltonian_riemannian_diag.hpp"
#include "modules/solver/sampler/HMC/helpers/leapfrog_explicit.hpp"
#include "modules/solver/sampler/HMC/helpers/leapfrog_implicit.hpp"
#include "modules/solver/sampler/HMC/helpers/tree_helper_euclidean.hpp"
#include "modules/solver/sampler/HMC/helpers/tree_helper_riemannian.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
;

/**
 * @brief Enum to set metric type.
 */
enum Metric
{
  /**
   * @brief Static Metric type.
   */
  Static = 0,

  /**
   * @brief Euclidean Metric type.
   */
  Euclidean = 1,

  /**
   * @brief Riemannian Metric type.
   */
  Riemannian = 2,

  /**
   * @brief Const Riemannian Metric type.
   */
  Riemannian_Const = 3,
};

/**
* @brief Class declaration for module: HMC.
*/
class HMC : public Sampler
{
  std::shared_ptr<Hamiltonian> _hamiltonian;
  std::unique_ptr<Leapfrog> _integrator;

  /**
   * @brief Updates internal state such as mean, Metric and InverseMetric.
   */
  void updateState();

  /**
   * @brief Process sample after evaluation.
   */
  void finishSample(size_t sampleId);

  /**
   * @brief Runs generation of HMC sampler.
   * @param logUniSample Log of uniform sample needed for Metropolis accepance / rejection step.
   */
  void runGenerationHMC(const double logUniSample);

  /**
   * @brief Runs NUTS algorithm with buildTree.
   * @param logUniSample Log of uniform sample needed for Metropolis accepance / rejection step.
   */
  void runGenerationNUTS(const double logUniSample);

  /**
   * @brief Runs NUTS algorithm with buildTree.
   * @param logUniSample Log of uniform sample needed for Metropolis accepance / rejection step.
   */
  void runGenerationNUTSRiemannian(const double logUniSample);

  /**
   * @brief Saves sample.
   */
  void saveSample();

  /**
   * @brief Updates Step Size for Adaptive Step Size.
   */
  void updateStepSize();

  /**
   * @brief Recursive binary tree building algorithm. Applied if configuration 'Use NUTS' is set to True.
   * @param helper Helper struct for large argument list.
   * @param depth Current depth of binary tree.
   */
  void buildTree(std::shared_ptr<TreeHelperEuclidean> helper, const size_t depth);

  /**
   * @brief Recursive binary tree building algorithm. Applied if configuration 'Use NUTS' is set to True.
   * @param helper Helper struct for large argument list.
   * @param rho Sum of momenta encountered along path.
   * @param depth Current depth of binary tree.
   */
  void buildTreeIntegration(std::shared_ptr<TreeHelperRiemannian> helper, std::vector<double> &rho, const size_t depth);

  public: 
  /**
  * @brief Specifies the number of preliminary HMC steps before samples are being drawn. This may reduce effects from improper initialization.
  */
   size_t _burnIn;
  /**
  * @brief Specifies if Metric is restricted to be diagonal.
  */
   int _useDiagonalMetric;
  /**
  * @brief Number of Integration steps used in Leapfrog scheme. Only relevant if Adaptive Step Size not used.
  */
   size_t _numIntegrationSteps;
  /**
  * @brief Number of Integration steps used in Leapfrog scheme. Only relevant if Adaptive Step Size is used.
  */
   size_t _maxIntegrationSteps;
  /**
  * @brief Specifies if No-U-Turn Sampler (NUTS) is used.
  */
   int _useNUTS;
  /**
  * @brief Step size used in Leapfrog scheme.
  */
   double _stepSize;
  /**
  * @brief Controls whether dual averaging technique for adaptive step size calibration is used.
  */
   int _useAdaptiveStepSize;
  /**
  * @brief Desired Acceptance Rate for Adaptive Step Size calibration.
  */
   double _targetAcceptanceRate;
  /**
  * @brief Learning rate of running acceptance rate estimate.
  */
   double _acceptanceRateLearningRate;
  /**
  * @brief Targeted Integration Time for Leapfrog scheme. Only relevant if Adaptive Step Size used.
  */
   double _targetIntegrationTime;
  /**
  * @brief Controls how fast the step size is adapted. Only relevant if Adaptive Step Size used.
  */
   double _adaptiveStepSizeSpeedConstant;
  /**
  * @brief Controls stability of adaptive step size calibration during the inital iterations. Only relevant if Adaptive Step Size used.
  */
   double _adaptiveStepSizeStabilizationConstant;
  /**
  * @brief Controls the weight of the previous step sizes. Only relevant if Adaptive Step Size used. The smaller the higher the weight.
  */
   double _adaptiveStepSizeScheduleConstant;
  /**
  * @brief Sets the maximum depth of NUTS binary tree.
  */
   size_t _maxDepth;
  /**
  * @brief Metric can be set to 'Static', 'Euclidean' or 'Riemannian'.
  */
   std::string _version;
  /**
  * @brief Controls hardness of inverse metric approximation: For large values the Inverse Metric is closer the to Hessian (and therefore closer to degeneracy in certain cases).
  */
   double _inverseRegularizationParameter;
  /**
  * @brief Max number of fixed point iterations during implicit leapfrog scheme.
  */
   size_t _maxFixedPointIterations;
  /**
  * @brief Step Size Jitter to vary trajectory length. Number must be in the interval [0.0. 1.0]. A uniform realization between [-(Step Size Jitter) * (Step Size), (Step Size Jitter) * (Step Size)) is sampled and added to the current Step Size.
  */
   double _stepSizeJitter;
  /**
  * @brief Initial warm-up interval during which step size is adaptively adjusted.
  */
   size_t _initialFastAdaptionInterval;
  /**
  * @brief Final warm-up interval during which step size is adaptively adjusted.
  */
   size_t _finalFastAdaptionInterval;
  /**
  * @brief Lenght of first (out of 5) warm-up intervals during which euclidean metric is adapted. The length of each following slow adaption intervals is doubled.
  */
   size_t _initialSlowAdaptionInterval;
  /**
  * @brief [Internal Use] Metric Type can be set to 'Static', 'Euclidean' or 'Riemannian'.
  */
   Metric _metricType;
  /**
  * @brief [Internal Use] Normal random number generator.
  */
   korali::distribution::univariate::Normal* _normalGenerator;
  /**
  * @brief [Internal Use] Random number generator with a multivariate normal distribution.
  */
   korali::distribution::multivariate::Normal* _multivariateGenerator;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] Ratio proposed to accepted samples (including Burn In period).
  */
   double _acceptanceRate;
  /**
  * @brief [Internal Use] Running estimate of current acceptance rate.
  */
   double _runningAcceptanceRate;
  /**
  * @brief [Internal Use] Number of accepted samples (including Burn In period).
  */
   size_t _acceptanceCount;
  /**
  * @brief [Internal Use] Number of proposed samples.
  */
   size_t _proposedSampleCount;
  /**
  * @brief [Internal Use] Parameters generated by HMC and stored in the database.
  */
   std::vector<std::vector<double>> _sampleDatabase;
  /**
  * @brief [Internal Use] Parameters generated during warmup. Used for Euclidean Metric approximation.
  */
   std::vector<std::vector<double>> _euclideanWarmupSampleDatabase;
  /**
  * @brief [Internal Use] Sample evaluations coresponding to the samples stored in Sample Databse.
  */
   std::vector<double> _sampleEvaluationDatabase;
  /**
  * @brief [Internal Use] Current Chain Length (including Burn In and Leaped Samples).
  */
   size_t _chainLength;
  /**
  * @brief [Internal Use] Evaluation of leader.
  */
   double _leaderEvaluation;
  /**
  * @brief [Internal Use] Evaluation of candidate.
  */
   double _candidateEvaluation;
  /**
  * @brief [Internal Use] Variables of the newest position/sample in the Markov chain.
  */
   std::vector<double> _positionLeader;
  /**
  * @brief [Internal Use] Candidate position to be accepted or rejected.
  */
   std::vector<double> _positionCandidate;
  /**
  * @brief [Internal Use] Latest momentum sample.
  */
   std::vector<double> _momentumLeader;
  /**
  * @brief [Internal Use] Proposed momentum after propagating Chain Leader and Momentum Leader according to Hamiltonian dynamics.
  */
   std::vector<double> _momentumCandidate;
  /**
  * @brief [Internal Use] Logarithm of smoothed average step size. Step size that is used after burn in period. Only relevant if adaptive step size used.
  */
   double _logDualStepSize;
  /**
  * @brief [Internal Use] Constant used for Adaptive Step Size option.
  */
   double _mu;
  /**
  * @brief [Internal Use] Constant used for Adaptive Step Size option.
  */
   double _hBar;
  /**
  * @brief [Internal Use] TODO: is this the number of accepted proposals?
  */
   double _acceptanceCountNUTS;
  /**
  * @brief [Internal Use] Depth of NUTS binary tree in current generation.
  */
   size_t _currentDepth;
  /**
  * @brief [Internal Use] Metropolis update acceptance probability - usually denoted with alpha - needed due to numerical error during integration.
  */
   double _acceptanceProbability;
  /**
  * @brief [Internal Use] Accumulated differences of Acceptance Probability and Target Acceptance Rate.
  */
   double _acceptanceRateError;
  /**
  * @brief [Internal Use] Metric for proposal distribution.
  */
   std::vector<double> _metric;
  /**
  * @brief [Internal Use] Inverse Metric for proposal distribution.
  */
   std::vector<double> _inverseMetric;
  /**
  * @brief [Termination Criteria] Number of Samples to Generate.
  */
   size_t _maxSamples;
  
 
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
   * @brief Configures HMC.
   */
  void setInitialConfiguration() override;

  /**
   * @brief Final console output at termination.
   */
  void finalize() override;

  /**
   * @brief Generate a sample and evaluate it.
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
};

} //sampler
} //solver
} //korali
;
