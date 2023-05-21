/** \namespace sampler
* @brief Namespace declaration for modules of type: sampler.
*/

/** \file
* @brief Header file for module: Nested.
*/

/** \dir solver/sampler/Nested
* @brief Contains code, documentation, and scripts for module: Nested.
*/

#pragma once

#include "modules/distribution/multivariate/normal/normal.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/sampler/sampler.hpp"
#include <vector>

namespace korali
{
namespace solver
{
namespace sampler
{
;

/**
 * @brief Ellipse object to generate bounds.
 */
struct ellipse_t
{
  /**
   * @brief Default c-tor (avoid empty initialization).
   */
  ellipse_t() = delete;

  /**
   * @brief Init d-dimensional ellipse without covariance.
   * @param dim Dimension of ellipsoid.
   */
  ellipse_t(size_t dim) : dim(dim), num(0), det(0.0), sampleIdx(0), mean(dim, 0.0), cov(dim * dim, 0.0), invCov(dim * dim, 0.0), axes(dim * dim, 0.0), evals(dim, 0.0), paxes(dim * dim, 0.0), volume(0.0), pointVolume(0.0){};

  /**
   * @brief Init d-dimensional unit sphere.
   */
  void initSphere();

  /**
   * @brief Scale volume.
   * @param factor Volume multiplicator.
   */
  void scaleVolume(double factor);

  /**
   * @brief Dimension of ellipsoid.
   */
  size_t dim;

  /**
   * @brief Number samples in ellipse.
   */
  size_t num;

  /**
   * @brief Determinant of covariance.
   */
  double det;

  /**
   * @brief Indices of samples from live data set.
   */
  std::vector<size_t> sampleIdx;

  /**
   * @brief Mean vector of samples in ellipse.
   */
  std::vector<double> mean;

  /**
   * @brief Covariance Matrix of samples in ellipse.
   */
  std::vector<double> cov;

  /**
   * @brief Inverse of Covariance Matrix.
   */
  std::vector<double> invCov;

  /**
   * @brief Axes of the ellipse.
   */
  std::vector<double> axes;

  /**
   * @brief Eigenvalues of the ellipse.
   */
  std::vector<double> evals;

  /**
   * @brief Principal axes of the ellipse.
   */
  std::vector<double> paxes;

  /**
   * @brief Volume estimated from covariance.
   */
  double volume;

  /**
   * @brief 'True' volume from which the subset of samples were sampled from.
   */
  double pointVolume;
};

/**
* @brief Class declaration for module: Nested.
*/
class Nested : public Sampler
{
  private:
  /*
   * @brief Seed for the shuffle randomizer
   */
  size_t _shuffleSeed;

  /*
   * @brief Storing ellipses (only relevant for Multi Ellipsoidal sampling).
   */
  std::vector<ellipse_t> _ellipseVector;

  /*
   * @brief Init and run first Generation.
   */
  void runFirstGeneration();

  /*
   * @brief Update the Bounds (Box, Ellipsoid or Multi Ellipsoid).
   */
  void updateBounds();

  /*
   * @brief Transformation from unit domain into prior domain.
   * @param sample Sample to transform.
   */
  void priorTransform(std::vector<double> &sample) const;

  /*
   * @brief Generate new candidates to evaluate.
   */
  void generateCandidates();

  /**
   * @brief Generate new samples uniformly in Box
   */
  void generateCandidatesFromBox();

  /**
   * @brief Generates a sample uniformly in Ellipse
   * @param ellipse Bounding ellipsoid from which to sample.
   * @param sample Generated sample.
   */
  void generateSampleFromEllipse(const ellipse_t &ellipse, std::vector<double> &sample) const;

  /**
   * @brief Generate new samples uniformly in Ellipse
   */
  void generateCandidatesFromEllipse();

  /**
   * @brief Generate new samples uniformly from multiple Ellipses
   */
  void generateCandidatesFromMultiEllipse();

  /*
   * @brief Process Generation after receiving all results.
   */
  bool processGeneration();

  /*
   * @brief Calculates the log prior weight.
   */
  double logPriorWeight(std::vector<double> &sample);

  /*
   * @brief Caclculate volume of bounds.
   */
  void setBoundsVolume();

  /*
   * @brief Add all live samples to sample data base.
   */
  void consumeLiveSamples();

  /*
   * @brief Updates bounding Box.
   */
  void updateBox();

  /*
   * @brief Sorts live sample ranks ascending based on loglikelihood and prior weight evaluation.
   */
  void sortLiveSamplesAscending();

  /*
   * @brief Remove sample from live samples and move it to dead samples.
   * @param sampleIdx Index of sample in live samples to process.
   */
  void updateDeadSamples(size_t sampleIdx);

  /*
   * @brief Generate posterior distribution from sample data base.
   */
  void generatePosterior();

  /*
   * @brief Calculate L2 distance between two vectors.
   * @param sampleOne Vector one
   * @param sampleTwo Vector two
   */
  double l2distance(const std::vector<double> &sampleOne, const std::vector<double> &sampleTwo) const;

  /*
   * @brief Updates bounding Ellipse (mean, cov and volume).
   * @param ellipse Ellipse to be updated.
   */
  bool updateEllipse(ellipse_t &ellipse) const;

  /*
   * @brief Updates ellipses based on Multi Nest.
   */
  void updateMultiEllipse();

  /*
   * @brief Initializes the ellipse vector with one ellipse..
   */
  void initEllipseVector();

  /*
   * @brief Applies k-means clustering (k=2) and fills cluster vectors with samples.
   * @param parent Parent ellipse to be split.
   * @param childOne Cluster one
   * @param childTwo Cluster two
   */
  bool kmeansClustering(const ellipse_t &parent, size_t maxIter, ellipse_t &childOne, ellipse_t &childTwo) const;

  /*
   * @brief Udates the mean vector of ellipse argument.
   * @param ellipse Ellipse to be updated.
   */
  void updateEllipseMean(ellipse_t &ellipse) const;

  /*
   * @brief Udates the covariance matrix of input ellipse.
   * @param ellipse Ellipse to be updated.
   */
  bool updateEllipseCov(ellipse_t &ellipse) const;

  /*
   * @brief Updates volume and the axes of the ellipse.
   * @param ellipse Ellipse to be updated.
   */
  bool updateEllipseVolume(ellipse_t &ellipse) const;

  /*
   * @brief Calculates Mahalanobis metric of sample and ellipse.
   * @param sample Sample.
   * @param ellipse Ellipse.
   */
  double mahalanobisDistance(const std::vector<double> &sample, const ellipse_t &ellipse) const;

  /*
   * @brief Calculate effective number of samples.
   * @return the number of effective samples
   */
  void updateEffectiveSamples();

  /*
   * @brief Checks if sample is inside d dimensional unit cube.
   * @param sample Sample to be checked.
   */
  bool insideUnitCube(const std::vector<double> &sample) const;

  public: 
  /**
  * @brief Number of live samples.
  */
   size_t _numberLivePoints;
  /**
  * @brief Number of samples to discard and replace per generation, maximal number of parallel sample evaluation.
  */
   size_t _batchSize;
  /**
  * @brief Add live points to dead points.
  */
   int _addLivePoints;
  /**
  * @brief Method to generate new candidates (can be set to either 'Box' or 'Ellipse', 'Multi Ellipse').
  */
   std::string _resamplingMethod;
  /**
  * @brief Frequency of resampling distribution update (e.g. ellipse rescaling for Ellipse).
  */
   size_t _proposalUpdateFrequency;
  /**
  * @brief Scaling factor of ellipsoidal (only relevant for 'Ellipse' and 'Multi Ellipse' proposal).
  */
   double _ellipsoidalScaling;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] Gaussian random number generator.
  */
   korali::distribution::univariate::Normal* _normalGenerator;
  /**
  * @brief [Internal Use] Random number generator with a multivariate normal distribution.
  */
   korali::distribution::multivariate::Normal* _multivariateGenerator;
  /**
  * @brief [Internal Use] Number of accepted samples.
  */
   size_t _acceptedSamples;
  /**
  * @brief [Internal Use] Number of generated samples (after initialization).
  */
   size_t _generatedSamples;
  /**
  * @brief [Internal Use] Accumulated LogEvidence.
  */
   double _logEvidence;
  /**
  * @brief [Internal Use] Estimated accumulated variance of log evidence.
  */
   double _logEvidenceVar;
  /**
  * @brief [Internal Use] Remaining Prior Mass.
  */
   double _logVolume;
  /**
  * @brief [Internal Use] Volume within bounds.
  */
   double _boundLogVolume;
  /**
  * @brief [Internal Use] Number of generations past since a sample has been accepted.
  */
   size_t _lastAccepted;
  /**
  * @brief [Internal Use] Next time when bounds are being updated.
  */
   size_t _nextUpdate;
  /**
  * @brief [Internal Use] Accumulated information.
  */
   double _information;
  /**
  * @brief [Internal Use] Likelihood constraint for sample acceptance.
  */
   double _lStar;
  /**
  * @brief [Internal Use] Previous likelihood constraint.
  */
   double _lStarOld;
  /**
  * @brief [Internal Use] Log increment of evidence.
  */
   double _logWeight;
  /**
  * @brief [Internal Use] Expected volume shrinkage per sample.
  */
   double _expectedLogShrinkage;
  /**
  * @brief [Internal Use] Largest sum of loglikelihood and logprior in live sample set.
  */
   double _maxEvaluation;
  /**
  * @brief [Internal Use] Estimated remaining log evidence.
  */
   double _remainingLogEvidence;
  /**
  * @brief [Internal Use] Difference of current and remaining log evidence.
  */
   double _logEvidenceDifference;
  /**
  * @brief [Internal Use] Number of effective Samples estimate.
  */
   double _effectiveSampleSize;
  /**
  * @brief [Internal Use] Sum of log weights in sample database.
  */
   double _sumLogWeights;
  /**
  * @brief [Internal Use] Sum of squared log weights in sample database.
  */
   double _sumSquareLogWeights;
  /**
  * @brief [Internal Use] Lower bound of unfirom prior.
  */
   std::vector<double> _priorLowerBound;
  /**
  * @brief [Internal Use] Width of uniform prior.
  */
   std::vector<double> _priorWidth;
  /**
  * @brief [Internal Use] Sample candidates to be evaluated in current generation.
  */
   std::vector<std::vector<double>> _candidates;
  /**
  * @brief [Internal Use] Loglikelihood evaluations of candidates.
  */
   std::vector<double> _candidateLogLikelihoods;
  /**
  * @brief [Internal Use] The logpriors of the candidates.
  */
   std::vector<double> _candidateLogPriors;
  /**
  * @brief [Internal Use] The logprior weights of the candidates.
  */
   std::vector<double> _candidateLogPriorWeights;
  /**
  * @brief [Internal Use] Samples to be processed and replaced in ascending order.
  */
   std::vector<std::vector<double>> _liveSamples;
  /**
  * @brief [Internal Use] Loglikelihood evaluations of live samples.
  */
   std::vector<double> _liveLogLikelihoods;
  /**
  * @brief [Internal Use] Logprior evaluations of live samples.
  */
   std::vector<double> _liveLogPriors;
  /**
  * @brief [Internal Use] Logprior weights of live samples.
  */
   std::vector<double> _liveLogPriorWeights;
  /**
  * @brief [Internal Use] Ascending ranking of live samples (sorted based on likelihood and logprior weight).
  */
   std::vector<size_t> _liveSamplesRank;
  /**
  * @brief [Internal Use] Number of dead samples, which have been removed from the live samples.
  */
   size_t _numberDeadSamples;
  /**
  * @brief [Internal Use] Dead samples stored in database.
  */
   std::vector<std::vector<double>> _deadSamples;
  /**
  * @brief [Internal Use] Loglikelihood evaluations of dead samples.
  */
   std::vector<double> _deadLogLikelihoods;
  /**
  * @brief [Internal Use] Logprior evaluations associated with dead samples.
  */
   std::vector<double> _deadLogPriors;
  /**
  * @brief [Internal Use] Logprior weights associated with dead samples.
  */
   std::vector<double> _deadLogPriorWeights;
  /**
  * @brief [Internal Use] Log weight (Priormass x Likelihood) of dead samples.
  */
   std::vector<double> _deadLogWeights;
  /**
  * @brief [Internal Use] Sample covariance of the live samples.
  */
   std::vector<double> _covarianceMatrix;
  /**
  * @brief [Internal Use] Log of domain volumne given by uniform prior distribution.
  */
   double _logDomainSize;
  /**
  * @brief [Internal Use] Mean of the domain occupied by live samples.
  */
   std::vector<double> _domainMean;
  /**
  * @brief [Internal Use] Lower bound of box constraint (only relevant for 'Box' resampling method).
  */
   std::vector<double> _boxLowerBound;
  /**
  * @brief [Internal Use] Upper bound of box constraint (only relevant for 'Box' resampling method).
  */
   std::vector<double> _boxUpperBound;
  /**
  * @brief [Internal Use] Axes of bounding ellipse (only relevant for 'Ellipse' resampling method).
  */
   std::vector<std::vector<double>> _ellipseAxes;
  /**
  * @brief [Termination Criteria] Minimal difference between estimated remaining log evidence and current logevidence.
  */
   double _minLogEvidenceDelta;
  /**
  * @brief [Termination Criteria] Estimated maximal evidence gain smaller than accumulated evidence by given factor.
  */
   size_t _maxEffectiveSampleSize;
  /**
  * @brief [Termination Criteria] Terminates if loglikelihood of sample removed from live set exceeds given value.
  */
   size_t _maxLogLikelihood;
  
 
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
   * @brief Configures Sampler.
   */
  void setInitialConfiguration() override;

  /**
   * @brief Main solver loop.
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

} //sampler
} //solver
} //korali
;
