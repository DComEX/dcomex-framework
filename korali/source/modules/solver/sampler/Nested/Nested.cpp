#include "engine.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/solver/sampler/Nested/Nested.hpp"
#include "sample/sample.hpp"

#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>

#include <algorithm> //sort
#include <chrono>
#include <limits>
#include <math.h> //isfinite, sqrt
#include <numeric>
#include <random> // std::default_random_engine

namespace korali
{
namespace solver
{
namespace sampler
{
;

void ellipse_t::initSphere()
{
  num = 0;
  sampleIdx.clear();
  std::fill(mean.begin(), mean.end(), 0.);
  std::fill(cov.begin(), cov.end(), 0.);
  std::fill(invCov.begin(), invCov.end(), 0.);
  std::fill(axes.begin(), axes.end(), 0.);
  std::fill(evals.begin(), evals.end(), 0.);
  std::fill(paxes.begin(), paxes.end(), 0.);

  for (size_t d = 0; d < dim; ++d)
  {
    cov[d * dim + d] = 1.;
    invCov[d * dim + d] = 1.;
    axes[d * dim + d] = 1.;
    evals[d] = 1.;
    paxes[d * dim + d] = 1.;
  }
  det = 1.;
  volume = std::sqrt(std::pow(M_PI, dim)) * 2. / ((double)dim * gsl_sf_gamma(0.5 * dim));
  pointVolume = 0.;
}

void ellipse_t::scaleVolume(double factor)
{
  const double K = std::sqrt(std::pow(M_PI, dim)) * 2. / ((double)dim * gsl_sf_gamma(0.5 * dim));
  const double enlargementFactor = std::pow((volume * volume * factor * factor) / (K * K * det), 1. / ((double)dim));
  for (size_t d = 0; d < dim * dim; ++d)
  {
    cov[d] *= enlargementFactor;
    invCov[d] *= 1. / enlargementFactor;
    axes[d] *= enlargementFactor;
  }
  for (size_t d = 0; d < dim; ++d) evals[d] *= enlargementFactor;
}

void Nested::setInitialConfiguration()
{
  _shuffleSeed = _k->_randomSeed++;
  _variableCount = _k->_variables.size();

  if (_minLogEvidenceDelta < 0.) KORALI_LOG_ERROR("Min Log Evidence Delta must be larger equal 0.0 (is %lf).\n", _minLogEvidenceDelta);

  if ((_resamplingMethod != "Box") && (_resamplingMethod != "Ellipse") && (_resamplingMethod != "Multi Ellipse")) KORALI_LOG_ERROR("Only accepted Resampling Method are 'Box', 'Ellipse' and 'Multi Ellipse' (is %s).\n", _resamplingMethod.c_str());

  if (_proposalUpdateFrequency <= 0) KORALI_LOG_ERROR("Proposal Update Frequency must be larger 0");

  _priorLowerBound.resize(_variableCount);
  _priorWidth.resize(_variableCount);

  for (size_t d = 0; d < _variableCount; ++d)
  {
    if (dynamic_cast<distribution::Univariate *>(_k->_distributions[_k->_variables[d]->_distributionIndex]) == nullptr) KORALI_LOG_ERROR("Prior of variable %s is not recognized (is nullptr).\n", _k->_variables[d]->_name.c_str());

    if ((iCompare(_k->_distributions[_k->_variables[d]->_distributionIndex]->_type, "Univariate/Uniform") == false) && (std::isfinite(_k->_variables[d]->_lowerBound) == false)) KORALI_LOG_ERROR("Prior of variable %s is not 'Univariate/Uniform' (is %s) AND lower bound not set (invalid configuration).\n", _k->_variables[d]->_name.c_str(), _k->_distributions[_k->_variables[d]->_distributionIndex]->_type.c_str());

    if ((iCompare(_k->_distributions[_k->_variables[d]->_distributionIndex]->_type, "Univariate/Uniform") == false) && (std::isfinite(_k->_variables[d]->_upperBound) == false)) KORALI_LOG_ERROR("Prior of variable %s is not 'Univariate/Uniform' (is %s) AND upper bound not set (invalid configuration).\n", _k->_variables[d]->_name.c_str(), _k->_distributions[_k->_variables[d]->_distributionIndex]->_type.c_str());

    if (iCompare(_k->_distributions[_k->_variables[d]->_distributionIndex]->_type, "Univariate/Uniform"))
    {
      _priorWidth[d] = dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_maximum - dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_minimum;
      _priorLowerBound[d] = dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_minimum;
    }
    else
    {
      _priorWidth[d] = _k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound;
      _priorLowerBound[d] = _k->_variables[d]->_lowerBound;
    }
  }

  if ((_resamplingMethod == "Ellipse" || _resamplingMethod == "Multi Ellipse") && (_variableCount < 3)) KORALI_LOG_ERROR("Resampling Method 'Ellipse' and 'Multi Ellipse' only suitable for problems of dim larger 2 (use Resampling Method 'Box').");

  // Init vectors
  _candidateLogLikelihoods.resize(_batchSize);
  _candidateLogPriors.resize(_batchSize);
  _candidateLogPriorWeights.resize(_batchSize);
  _candidates.resize(_batchSize);
  for (size_t i = 0; i < _batchSize; i++) _candidates[i].resize(_variableCount);

  _liveLogLikelihoods.resize(_numberLivePoints);
  _liveLogPriors.resize(_numberLivePoints);
  _liveLogPriorWeights.resize(_numberLivePoints);
  _liveSamplesRank.resize(_numberLivePoints);
  _liveSamples.resize(_numberLivePoints);
  for (size_t i = 0; i < _numberLivePoints; i++) _liveSamples[i].resize(_variableCount);

  _numberDeadSamples = 0;
  _deadLogLikelihoods.resize(0);
  _deadLogPriors.resize(0);
  _deadLogWeights.resize(0);
  _deadLogPriorWeights.resize(0);
  _deadSamples.resize(0);

  // Init Generation
  _logEvidence = Lowest;
  _sumLogWeights = Lowest;
  _sumSquareLogWeights = Lowest;
  _logEvidenceDifference = Max;
  _expectedLogShrinkage = std::log((_numberLivePoints + 1.) / _numberLivePoints);
  _logVolume = 0;

  _logEvidenceVar = 0.;
  _information = 0.;
  _lastAccepted = 0;
  _nextUpdate = 0;
  _acceptedSamples = 0;
  _generatedSamples = 0;
  _lStarOld = Lowest;
  _lStar = Lowest;

  _domainMean.resize(_variableCount);
  if (_resamplingMethod == "Box")
  {
    _boxLowerBound.resize(_variableCount);
    _boxUpperBound.resize(_variableCount);
  }
  else if (_resamplingMethod == "Ellipse")
  {
    initEllipseVector();
  }
  else /* _resamplingMethod == "Multi Ellipse" */
  {
    initEllipseVector();
  }
}

void Nested::runGeneration()
{
  if (_k->_currentGeneration == 1)
  {
    setInitialConfiguration();
    runFirstGeneration();
    return;
  };

  // Generation > 1
  _lastAccepted = 0;
  bool accepted = false;
  std::vector<double> sample;
  std::vector<Sample> samples(_batchSize);

  // Repeat until we accept at least one sample
  while (accepted == false)
  {
    updateBounds();
    generateCandidates();

    for (size_t c = 0; c < _batchSize; c++)
    {
      samples[c]["Module"] = "Problem";
      samples[c]["Operation"] = "Evaluate";
      sample = _candidates[c];
      priorTransform(sample);
      samples[c]["Parameters"] = sample;
      samples[c]["Sample Id"] = c;
      KORALI_START(samples[c]);
      _modelEvaluationCount++;
      _generatedSamples++;
    }

    size_t finishedCandidatesCount = 0;
    // Store candidate information
    while (finishedCandidatesCount < _batchSize)
    {
      size_t finishedId = KORALI_WAITANY(samples);

      auto candidate = KORALI_GET(std::vector<double>, samples[finishedId], "Parameters");
      _candidateLogPriors[finishedId] = KORALI_GET(double, samples[finishedId], "logPrior");
      _candidateLogPriorWeights[finishedId] = logPriorWeight(candidate);
      _candidateLogLikelihoods[finishedId] = KORALI_GET(double, samples[finishedId], "logLikelihood");

      finishedCandidatesCount++;
    }

    _lastAccepted++;
    accepted = processGeneration();
  }

  return;
}

void Nested::runFirstGeneration()
{
  for (size_t i = 0; i < _numberLivePoints; i++)
    for (size_t d = 0; d < _variableCount; d++)
      _liveSamples[i][d] = _uniformGenerator->getRandomNumber();

  std::vector<double> sample;
  std::vector<Sample> samples(_numberLivePoints);

  // Evaluate all live samples
  for (size_t c = 0; c < _numberLivePoints; c++)
  {
    samples[c]["Module"] = "Problem";
    samples[c]["Operation"] = "Evaluate";
    sample = _liveSamples[c];
    priorTransform(sample);
    samples[c]["Parameters"] = sample;
    samples[c]["Sample Id"] = c;
    KORALI_START(samples[c]);
    _modelEvaluationCount++;
    _generatedSamples++;
  }

  size_t finishedCandidatesCount = 0;
  // Store live sample information
  while (finishedCandidatesCount < _numberLivePoints)
  {
    size_t finishedId = KORALI_WAITANY(samples);

    auto sample = KORALI_GET(std::vector<double>, samples[finishedId], "Parameters");
    _liveLogPriors[finishedId] = KORALI_GET(double, samples[finishedId], "logPrior");
    _liveLogPriorWeights[finishedId] = logPriorWeight(sample);
    _liveLogLikelihoods[finishedId] = KORALI_GET(double, samples[finishedId], "logLikelihood");

    finishedCandidatesCount++;
  }

  // Rank all live samples
  sortLiveSamplesAscending();

  // Find min (lStar) and max evaluation
  const size_t minRank = _liveSamplesRank[0];
  if (isfinite(_liveLogPriorWeights[minRank] + _liveLogLikelihoods[minRank])) _lStar = _liveLogPriorWeights[minRank] + _liveLogLikelihoods[minRank];
  const size_t maxRank = _liveSamplesRank[_numberLivePoints - 1];
  _maxEvaluation = _liveLogPriorWeights[maxRank] + _liveLogLikelihoods[maxRank];
}

void Nested::updateBounds()
{
  if (_generatedSamples < _nextUpdate && _ellipseVector.empty() == false) return; // no update

  // Set next update of bounding hypervolume
  _nextUpdate += _proposalUpdateFrequency;

  // Update bounding hypervolume
  if (_resamplingMethod == "Box")
  {
    updateBox();
  }
  else if (_resamplingMethod == "Ellipse")
  {
    updateEllipse(_ellipseVector.front());
  }
  else /* _resamplingMethod == "Multi Ellipse" */
  {
    updateMultiEllipse();
  }
}

void Nested::priorTransform(std::vector<double> &sample) const
{
  // Transofrmation from unit hypercube to bounded domain
  for (size_t d = 0; d < _variableCount; ++d) sample[d] = _priorLowerBound[d] + sample[d] * _priorWidth[d];
}

void Nested::generateCandidates()
{
  if (_resamplingMethod == "Box")
  {
    generateCandidatesFromBox();
  }
  else if (_resamplingMethod == "Ellipse")
  {
    generateCandidatesFromEllipse();
  }
  else /* _resamplingMethod == "Multi Ellipse" */
  {
    generateCandidatesFromMultiEllipse();
  }
}

bool Nested::processGeneration()
{
  size_t sampleIdx = _liveSamplesRank[0];
  size_t acceptedBefore = _acceptedSamples;

  // Process candidates
  for (size_t c = 0; c < _batchSize; ++c)
  {
    if (_candidateLogLikelihoods[c] < _lStar) continue; // Ignore candidate

    // Candidate accepted
    _acceptedSamples++;

    // Update evidence & domain
    double logVolumeOld = _logVolume;
    double informationOld = _information;
    double logEvidenceOld = _logEvidence;

    _logVolume -= _expectedLogShrinkage;

    double dLogVol = std::log(0.5 * std::exp(logVolumeOld) - 0.5 * std::exp(_logVolume));
    _logWeight = safeLogPlus(_lStar, _lStarOld) + dLogVol;
    _logEvidence = safeLogPlus(_logEvidence, _logWeight);

    double evidenceTerm = std::exp(_lStarOld - _logEvidence) * _lStarOld + std::exp(_lStar - _logEvidence) * _lStar;

    if (isfinite(evidenceTerm))
    {
      _information = std::exp(dLogVol) * evidenceTerm + std::exp(logEvidenceOld - _logEvidence) * (informationOld + logEvidenceOld) - _logEvidence;
      _logEvidenceVar += 2. * (_information - informationOld) * _expectedLogShrinkage;
    }

    // Add candidate to dead samples
    if (isfinite(_liveLogPriorWeights[sampleIdx] + _liveLogLikelihoods[sampleIdx]))
    {
      updateDeadSamples(sampleIdx);
    }

    // Replace worst sample from live samples by candidate
    _liveSamples[sampleIdx] = _candidates[c];
    _liveLogPriors[sampleIdx] = _candidateLogPriors[c];
    _liveLogPriorWeights[sampleIdx] = _candidateLogPriorWeights[c];
    _liveLogLikelihoods[sampleIdx] = _candidateLogLikelihoods[c];

    // Sort rank vector
    sortLiveSamplesAscending();

    // Select new worst sample
    sampleIdx = _liveSamplesRank[0];

    // Update lStar
    if (isfinite(_liveLogPriorWeights[sampleIdx] + _liveLogLikelihoods[sampleIdx]))
    {
      _lStarOld = _lStar;
      _lStar = _liveLogPriorWeights[sampleIdx] + _liveLogLikelihoods[sampleIdx];
    }
  }

  // Update statistics
  const size_t maxRank = _liveSamplesRank[_numberLivePoints - 1];
  _maxEvaluation = _liveLogPriorWeights[maxRank] + _liveLogLikelihoods[maxRank];
  _remainingLogEvidence = _maxEvaluation + _logVolume;
  _logEvidenceDifference = safeLogPlus(_logEvidence, _remainingLogEvidence) - _logEvidence;
  setBoundsVolume();

  _lastAccepted++;
  return (acceptedBefore != _acceptedSamples);
}

double Nested::logPriorWeight(std::vector<double> &sample)
{
  double logweight = 0.;
  // Calculate lof prior weight (i.e. log of prior/bounded box volume)
  for (size_t d = 0; d < _variableCount; ++d)
  {
    logweight += dynamic_cast<distribution::Univariate *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->getLogDensity(sample[d]);
    logweight += std::log(_priorWidth[d]);
  }
  return logweight;
}

void Nested::setBoundsVolume()
{
  if (_resamplingMethod == "Box")
  {
    // Calculate volume of bounding box
    _boundLogVolume = Lowest;
    for (size_t d = 0; d < _variableCount; ++d)
      _boundLogVolume = safeLogPlus(_boundLogVolume, std::log(_boxUpperBound[d] - _boxLowerBound[d]));
  }
  else if (_resamplingMethod == "Ellipse")
  {
    // Calculate volume of ellipsoid
    auto &ellipse = _ellipseVector.front();
    _boundLogVolume = std::log(ellipse.volume);
  }
  else /* _resamplingMethod == "Multi Ellipse" */
  {
    // Calculate volume of (overlapping) ellipsoids
    _boundLogVolume = Lowest;
    for (auto &ellipse : _ellipseVector)
      _boundLogVolume = safeLogPlus(_boundLogVolume, std::log(ellipse.volume));
  }
}

void Nested::generateCandidatesFromBox()
{
  // Generate sample uniformly inside bounding box
  for (size_t i = 0; i < _batchSize; i++)
  {
    for (size_t d = 0; d < _variableCount; ++d)
      _candidates[i][d] = _boxLowerBound[d] + _uniformGenerator->getRandomNumber() * (_boxUpperBound[d] - _boxLowerBound[d]);
    if (insideUnitCube(_candidates[i]) == false) i--;
  }
}

void Nested::generateSampleFromEllipse(const ellipse_t &ellipse, std::vector<double> &sample) const
{
  // Generate sample uniformly inside bounding ellipsoid
  double len = 0;
  std::vector<double> vec(_variableCount);
  for (size_t d = 0; d < _variableCount; ++d)
  {
    vec[d] = _normalGenerator->getRandomNumber();
    len += vec[d] * vec[d];
  }
  for (size_t d = 0; d < _variableCount; ++d)
    vec[d] *= std::pow(_uniformGenerator->getRandomNumber(), 1. / ((double)_variableCount)) / sqrt(len);

  for (size_t k = 0; k < _variableCount; ++k)
  {
    sample[k] = ellipse.mean[k];
    for (size_t l = 0; l < k + 1; ++l)
    {
      sample[k] += ellipse.axes[k * _variableCount + l] * vec[l];
    }
  }
}

void Nested::generateCandidatesFromEllipse()
{
  for (size_t i = 0; i < _batchSize; i++)
  {
    generateSampleFromEllipse(_ellipseVector.front(), _candidates[i]);
    if (insideUnitCube(_candidates[i]) == false) i--;
  }
}

void Nested::generateCandidatesFromMultiEllipse()
{
  double totalVol = 0.;
  for (auto &ellipse : _ellipseVector) totalVol += ellipse.volume;

  for (size_t i = 0; i < _batchSize; i++)
  {
    // randomly select ellipse
    double cumVol = 0.;
    double rnd_ellipse = _uniformGenerator->getRandomNumber() * totalVol;

    ellipse_t *ellipse_ptr = NULL;
    for (auto &ellipse : _ellipseVector)
    {
      cumVol += ellipse.volume;
      if (rnd_ellipse < cumVol)
      {
        ellipse_ptr = &ellipse;
        break;
      }
    }

    // SM - Only add a check if you can create a unit test to trigger it
    //    if (ellipse_ptr == NULL)
    //      KORALI_LOG_ERROR("Failed to assign ellipse vector.");

    bool accept = false;
    while (accept == false)
    {
      // sample from ellipse
      generateSampleFromEllipse(*ellipse_ptr, _candidates[i]);

      // check for overlaps
      size_t overlap = 0;
      for (auto &ellipse : _ellipseVector)
      {
        double dist = mahalanobisDistance(_candidates[i], ellipse);
        if (dist <= 1.) overlap++;
      }

      // accept / reject
      double rnd = _uniformGenerator->getRandomNumber();
      if (rnd < 1. / ((double)overlap)) accept = true;
      if (insideUnitCube(_candidates[i]) == false) accept = false;
    }
  }
}

void Nested::updateBox()
{
  for (size_t d = 0; d < _variableCount; d++) _boxLowerBound[d] = Max;
  for (size_t d = 0; d < _variableCount; d++) _boxUpperBound[d] = Lowest;

  for (size_t i = 0; i < _numberLivePoints; i++)
    for (size_t d = 0; d < _variableCount; d++)
    {
      _boxLowerBound[d] = std::min(_boxLowerBound[d], _liveSamples[i][d]);
      _boxUpperBound[d] = std::max(_boxUpperBound[d], _liveSamples[i][d]);
    }
}

void Nested::sortLiveSamplesAscending()
{
  // Init sample ranks
  std::iota(_liveSamplesRank.begin(), _liveSamplesRank.end(), 0);

  // clang-format off
  // Sort sample rank ascending based on likelihood and prior weight
  std::sort(_liveSamplesRank.begin(), _liveSamplesRank.end(), [this](const size_t &idx1, const size_t &idx2) -> bool
            {
              return this->_liveLogPriorWeights[idx1] + this->_liveLogLikelihoods[idx1] < this->_liveLogPriorWeights[idx2] + this->_liveLogLikelihoods[idx2];
            });
  // clang-format on
}

void Nested::updateDeadSamples(size_t sampleIdx)
{
  _numberDeadSamples++;
  _deadSamples.push_back(_liveSamples[sampleIdx]);
  priorTransform(_deadSamples.back());

  _deadLogPriors.push_back(_liveLogPriors[sampleIdx]);
  _deadLogPriorWeights.push_back(_liveLogPriorWeights[sampleIdx]);
  _deadLogLikelihoods.push_back(_liveLogLikelihoods[sampleIdx]);
  _deadLogWeights.push_back(_logWeight);

  updateEffectiveSamples();
}

void Nested::consumeLiveSamples()
{
  size_t sampleIdx;
  double dLogVol, logEvidenceOld, informationOld, evidenceTerm;

  std::vector<double> logvols(_numberLivePoints + 1, _logVolume);
  std::vector<double> logdvols(_numberLivePoints);
  std::vector<double> dlvs(_numberLivePoints);

  for (size_t i = 0; i < _numberLivePoints; ++i)
  {
    logvols[i + 1] += log(1. - (i + 1.) / (_numberLivePoints + 1.));
    logdvols[i] = safeLogMinus(logvols[i], logvols[i + 1]);
    dlvs[i] = logvols[i] - logvols[i + 1];
  }
  for (size_t i = 0; i < _numberLivePoints + 1; ++i) logdvols[i] += std::log(0.5);

  // Add reamining live samples to dead samples
  for (size_t i = 0; i < _numberLivePoints; ++i)
  {
    // Process samples in ascending order
    sampleIdx = _liveSamplesRank[i];

    logEvidenceOld = _logEvidence;
    informationOld = _information;

    // Update lStar and add sample to dead samples
    if (isfinite(_liveLogPriorWeights[sampleIdx] + _liveLogLikelihoods[sampleIdx]))
    {
      _lStarOld = _lStar;
      _lStar = _liveLogPriorWeights[sampleIdx] + _liveLogLikelihoods[sampleIdx];
      updateDeadSamples(sampleIdx);
    }

    dLogVol = logdvols[i];

    _logVolume = safeLogMinus(_logVolume, dLogVol);
    _logWeight = safeLogPlus(_lStar, _lStarOld) + dLogVol;
    _logEvidence = safeLogPlus(_logEvidence, _logWeight);

    evidenceTerm = std::exp(_lStarOld - _logEvidence) * _lStarOld + std::exp(_lStar - _logEvidence) * _lStar;

    _information = std::exp(dLogVol) * evidenceTerm + std::exp(logEvidenceOld - _logEvidence) * (informationOld + logEvidenceOld) - _logEvidence;

    _logEvidenceVar += 2. * (_information - informationOld) * dlvs[i];
  }
}

void Nested::generatePosterior()
{
  double maxLogWtDb = *max_element(std::begin(_deadLogWeights), std::end(_deadLogWeights));

  std::vector<size_t> permutation(_numberDeadSamples);
  std::iota(std::begin(permutation), std::end(permutation), 0);
  std::shuffle(permutation.begin(), permutation.end(), std::default_random_engine(_shuffleSeed));

  size_t rndIdx;
  std::vector<std::vector<double>> posteriorSamples;
  std::vector<double> posteriorSamplesLogPriorDatabase;
  std::vector<double> posteriorSamplesLogLikelihoodDatabase;

  double k = 1.;
  double sum = _uniformGenerator->getRandomNumber();
  // Resample dead samples based on their weight to produce posterior
  for (size_t i = 0; i < _numberDeadSamples; ++i)
  {
    rndIdx = permutation[i];
    sum += std::exp(_deadLogWeights[rndIdx] - maxLogWtDb);
    if (sum > k)
    {
      posteriorSamples.push_back(_deadSamples[rndIdx]);
      posteriorSamplesLogPriorDatabase.push_back(_deadLogPriors[rndIdx]);
      posteriorSamplesLogLikelihoodDatabase.push_back(_deadLogLikelihoods[rndIdx]);
      k++;
    }
  }

  (*_k)["Results"]["Posterior Sample Database"] = posteriorSamples;
  (*_k)["Results"]["Posterior Sample LogPrior Database"] = posteriorSamplesLogPriorDatabase;
  (*_k)["Results"]["Posterior Sample LogLikelihood Database"] = posteriorSamplesLogLikelihoodDatabase;
}

double Nested::l2distance(const std::vector<double> &sampleOne, const std::vector<double> &sampleTwo) const
{
  double dist = 0.;
  // Calculate L2 distance
  for (size_t d = 0; d < _variableCount; ++d) dist += (sampleOne[d] - sampleTwo[d]) * (sampleOne[d] - sampleTwo[d]);
  dist = std::sqrt(dist);
  return dist;
}

bool Nested::updateEllipse(ellipse_t &ellipse) const
{
  if (ellipse.num == 0) return false;

  updateEllipseMean(ellipse);
  bool good = updateEllipseCov(ellipse);
  if (good) good = updateEllipseVolume(ellipse);

  return good;
}

void Nested::updateMultiEllipse()
{
  // Create ellipsoid
  initEllipseVector();
  bool ok = updateEllipse(_ellipseVector.front());
  if (ok == false) KORALI_LOG_ERROR("Ellipse update failed at initialization\n");

  bool okCluster, okOne, okTwo;

  std::vector<ellipse_t> newEllipseVector;
  for (size_t idx = 0; idx < _ellipseVector.size(); ++idx)
  {
    ellipse_t &ellipse = _ellipseVector[idx];
    auto one = ellipse_t(_variableCount);
    auto two = ellipse_t(_variableCount);

    //
    okCluster = kmeansClustering(ellipse, 100, one, two);

    if (okCluster)
    {
      okOne = updateEllipse(one);
      okTwo = updateEllipse(two);
    }

    if ((okCluster == false) || (okOne == false) || (okTwo == false) || ((one.volume + two.volume >= 0.5 * ellipse.volume) && (ellipse.volume < 2. * std::exp(_logVolume))))
    {
      newEllipseVector.push_back(ellipse);
    }
    else
    {
      _ellipseVector.push_back(one);
      _ellipseVector.push_back(two);
    }
  }

  _ellipseVector = newEllipseVector;
}

void Nested::initEllipseVector()
{
  _ellipseVector.clear();
  _ellipseVector.emplace_back(ellipse_t(_variableCount));
  ellipse_t *first = _ellipseVector.data();

  first->num = _numberLivePoints;
  first->sampleIdx.resize(_numberLivePoints);
  // Assign all samples to current ellipsoid
  std::iota(first->sampleIdx.begin(), first->sampleIdx.end(), 0);
}

void Nested::updateEllipseMean(ellipse_t &ellipse) const
{
  std::fill(ellipse.mean.begin(), ellipse.mean.end(), 0.);

  for (size_t i = 0; i < ellipse.num; ++i)
    for (size_t d = 0; d < _variableCount; ++d)
    {
      size_t sidx = ellipse.sampleIdx[i];
      ellipse.mean[d] += _liveSamples[sidx][d];
    }

  if (ellipse.num > 0)
    for (size_t d = 0; d < _variableCount; ++d)
      ellipse.mean[d] /= ((double)ellipse.num);
}

bool Nested::updateEllipseCov(ellipse_t &ellipse) const
{
  double weight = 1. / (ellipse.num - 1.);

  if (ellipse.num <= ellipse.dim)
  {
    // update variance
    std::fill(ellipse.cov.begin(), ellipse.cov.end(), 0.);
    std::fill(ellipse.invCov.begin(), ellipse.invCov.end(), 0.);
    for (size_t d = 0; d < _variableCount; ++d)
    {
      double c = 0.;
      for (size_t k = 0; k < ellipse.num; ++k)
      {
        size_t sidx = ellipse.sampleIdx[k];
        c += (_liveSamples[sidx][d] - ellipse.mean[d]) * (_liveSamples[sidx][d] - ellipse.mean[d]);
        ellipse.cov[d * _variableCount + d] = (ellipse.num == 1) ? 1. : weight * c;
        ellipse.invCov[d * _variableCount + d] = (ellipse.num == 1) ? 1. : 1. / (weight * c);
      }
    }
  }
  else
  {
    // update covariance
    for (size_t i = 0; i < _variableCount; ++i)
    {
      for (size_t j = i; j < _variableCount; ++j)
      {
        double c = 0.;
        for (size_t k = 0; k < ellipse.num; ++k)
        {
          size_t sidx = ellipse.sampleIdx[k];
          c += (_liveSamples[sidx][i] - ellipse.mean[i]) * (_liveSamples[sidx][j] - ellipse.mean[j]);
        }
        ellipse.cov[j * _variableCount + i] = ellipse.cov[i * _variableCount + j] = weight * c;
      }
    }

    // update inverse covariance
    gsl_matrix_view cov = gsl_matrix_view_array(ellipse.cov.data(), _variableCount, _variableCount);

    gsl_matrix *matLU = gsl_matrix_alloc(_variableCount, _variableCount);
    gsl_matrix_memcpy(matLU, &cov.matrix);

    int signal;
    gsl_permutation *perm = gsl_permutation_alloc(_variableCount);
    gsl_linalg_LU_decomp(matLU, perm, &signal);

    std::fill(ellipse.invCov.begin(), ellipse.invCov.end(), 0.);
    gsl_matrix_view invCov = gsl_matrix_view_array(ellipse.invCov.data(), _variableCount, _variableCount);

    gsl_linalg_LU_invert(matLU, perm, &invCov.matrix);
    gsl_permutation_free(perm);
    gsl_matrix_free(matLU);

    // SM - Only add a check if you can create a unit test to trigger it
    //    if (status != 0)
    //    {
    //      _k->_logger->logWarning("Normal", "Covariance inversion failed during ellipsoid covariance update.\n");
    //      return false;
    //    }
  }

  return true;
}

bool Nested::updateEllipseVolume(ellipse_t &ellipse) const
{
  if (ellipse.num == 0) return false;

  gsl_matrix_view cov = gsl_matrix_view_array(ellipse.cov.data(), _variableCount, _variableCount);

  gsl_matrix *matLU = gsl_matrix_alloc(_variableCount, _variableCount);
  gsl_matrix_memcpy(matLU, &cov.matrix);

  // SM - Only add a check if you can create a unit test to trigger it
  //  if (status != 0)
  //  {
  //    _k->_logger->logWarning("Normal", "Memcpy failed during Ellipsoid volume update.\n");
  //    gsl_matrix_free(matLU);
  //    return false;
  //  }

  int signal;
  gsl_permutation *perm = gsl_permutation_alloc(_variableCount);
  gsl_linalg_LU_decomp(matLU, perm, &signal);
  gsl_permutation_free(perm);

  ellipse.det = gsl_linalg_LU_det(matLU, signal);
  gsl_matrix_free(matLU);

  gsl_vector_view evals = gsl_vector_view_array(ellipse.evals.data(), _variableCount);
  gsl_matrix_view paxes = gsl_matrix_view_array(ellipse.paxes.data(), _variableCount, _variableCount);

  gsl_matrix *matEigen = gsl_matrix_alloc(_variableCount, _variableCount);
  gsl_matrix_memcpy(matEigen, &cov.matrix);

  // SM - Only add a check if you can create a unit test to trigger it
  //  if (status != 0)
  //  {
  //    _k->_logger->logWarning("Normal", "Memcpy failed during Ellipsoid volume update.\n");
  //    gsl_matrix_free(matEigen);
  //    return false;
  //  }

  gsl_eigen_symmv_workspace *workEigen = gsl_eigen_symmv_alloc(_variableCount);
  gsl_eigen_symmv(matEigen, &evals.vector, &paxes.matrix, workEigen);
  gsl_matrix_free(matEigen);
  gsl_eigen_symmv_free(workEigen);

  // SM - Only add a check if you can create a unit test to trigger it
  //  if (status != 0)
  //  {
  //    _k->_logger->logWarning("Normal", "Eigenvalue Decomposition failed during Ellipsoid volume update.\n");
  //    return false;
  //  }

  gsl_eigen_symmv_sort(&evals.vector, &paxes.matrix, GSL_EIGEN_SORT_ABS_DESC);

  // SM - Only add a check if you can create a unit test to trigger it
  //  if (status != 0)
  //  {
  //    _k->_logger->logWarning("Normal", "Eigenvalue sorting failed during Ellipsoid volume update.\n");
  //    return false;
  //  }

  // Calculate axes from cholesky decomposition
  gsl_matrix_view axes = gsl_matrix_view_array(ellipse.axes.data(), _variableCount, _variableCount);

  /* On output the diagonal and lower triangular part of the
   * input matrix A contain the matrix L, while the upper triangular part
   * contains the original matrix. */

  gsl_matrix_memcpy(&axes.matrix, &cov.matrix);

  // SM - Only add a check if you can create a unit test to trigger it
  //  if (status != 0)
  //  {
  //    _k->_logger->logWarning("Normal", "Memcpy failed during Ellipsoid volume update.\n");
  //    return false;
  //  }

  gsl_linalg_cholesky_decomp1(&axes.matrix); // LL^T = A

  // SM - Only add a check if you can create a unit test to trigger it
  //  if (status != 0)
  //  {
  //    _k->_logger->logWarning("Normal", "Cholesky Decomposition failed during Ellipsoid volume update.\n");
  //    return false;
  //  }

  // Find scaling s.t. all samples are bounded by ellipse
  double res, max = Lowest;
  for (size_t i = 0; i < ellipse.num; ++i)
  {
    size_t six = ellipse.sampleIdx[i];
    res = mahalanobisDistance(_liveSamples[six], ellipse);
    if (res > max) max = res;
  }

  ellipse.pointVolume = std::exp(_logVolume) * (double)ellipse.num / ((double)_numberLivePoints);

  double K = std::sqrt(std::pow(M_PI, _variableCount)) * 2. / ((double)_variableCount * gsl_sf_gamma(0.5 * _variableCount));
  double vol = std::sqrt(std::pow(_ellipsoidalScaling * max, _variableCount) * ellipse.det) * K;

  double enlargementFactor = vol > ellipse.pointVolume ? _ellipsoidalScaling * max : std::pow((ellipse.pointVolume * ellipse.pointVolume) / (K * K * ellipse.det), 1. / ((double)_variableCount));
  ellipse.volume = std::pow(enlargementFactor, _variableCount / 2.) * sqrt(ellipse.det) * K;

  gsl_matrix_view invCov = gsl_matrix_view_array(ellipse.invCov.data(), _variableCount, _variableCount);

  // resize volume
  gsl_matrix_scale(&cov.matrix, enlargementFactor);
  gsl_matrix_scale(&invCov.matrix, 1. / enlargementFactor);
  gsl_vector_scale(&evals.vector, enlargementFactor);
  gsl_matrix_scale(&axes.matrix, sqrt(enlargementFactor));

  return true; // all good
}

double Nested::mahalanobisDistance(const std::vector<double> &sample, const ellipse_t &ellipse) const
{
  std::vector<double> dif(_variableCount);
  for (size_t d = 0; d < _variableCount; ++d) dif[d] = sample[d] - ellipse.mean[d];

  double tmp;
  double dist = 0.;
  // Calculate Mahalanobis distance between sample and ellipsoid
  for (size_t i = 0; i < _variableCount; ++i)
  {
    tmp = 0.;
    for (size_t j = 0; j < _variableCount; ++j)
      tmp += dif[j] * ellipse.invCov[i + _variableCount * j];
    tmp *= dif[i];
    dist += tmp;
  }
  return dist;
}

bool Nested::kmeansClustering(const ellipse_t &parent, size_t maxIter, ellipse_t &childOne, ellipse_t &childTwo) const
{
  childOne.initSphere();
  childTwo.initSphere();

  size_t ax = 0;
  // Initialize center of ellipsoidals one and two at the ends of the longest axe
  for (size_t d = 0; d < _variableCount; ++d)
  {
    childOne.mean[d] = parent.mean[d] + parent.paxes[d * _variableCount + ax] * parent.evals[ax];
    childTwo.mean[d] = parent.mean[d] - parent.paxes[d * _variableCount + ax] * parent.evals[ax];
  }

  size_t nOne, nTwo, idxOne, idxTwo;
  std::vector<int8_t> clusterFlag(parent.num, 0);

  bool ok = true;
  size_t iter = 0;
  size_t diffs = 1;

  // Iterate until no sample updates cluster assignment
  while ((diffs > 0) && (iter++ < maxIter))
  {
    diffs = 0;
    nOne = 0;
    nTwo = 0;

    // assign samples to means
    for (size_t i = 0; i < parent.num; ++i)
    {
      size_t six = parent.sampleIdx[i];

      // measure distances to cluster means
      double d1 = l2distance(_liveSamples[six], childOne.mean);
      double d2 = l2distance(_liveSamples[six], childTwo.mean);

      int8_t flag = (d1 < d2) ? 1 : 2;

      // count updates
      if (clusterFlag[i] != flag) diffs++;
      // assign cluster
      clusterFlag[i] = flag;

      if (flag == 1)
        nOne++;
      else
        nTwo++;
    }

    childOne.num = nOne;
    childOne.sampleIdx.resize(nOne);

    childTwo.num = nTwo;
    childTwo.sampleIdx.resize(nTwo);

    idxOne = 0;
    idxTwo = 0;

    for (size_t i = 0; i < parent.num; ++i)
    {
      if (clusterFlag[i] == 1)
        childOne.sampleIdx[idxOne++] = parent.sampleIdx[i];
      else /* clusterFlag[i] == 2 */
        childTwo.sampleIdx[idxTwo++] = parent.sampleIdx[i];
    }

    // update means of ellipsoids
    updateEllipseMean(childOne);
    updateEllipseMean(childTwo);

    ok = ok & updateEllipseCov(childOne);
    ok = ok & updateEllipseCov(childTwo);

    if (ok == false) break;
  }

  // SM - Only add a check if you can create a unit test to trigger it
  //  if (iter >= maxIter) _k->_logger->logWarning("Normal", "K-Means Clustering did not terminate in %zu steps.\n", maxIter);

  return ok;
}

void Nested::updateEffectiveSamples()
{
  double w = _deadLogWeights.back();
  _sumLogWeights = safeLogPlus(_sumLogWeights, w);
  _sumSquareLogWeights = safeLogPlus(_sumSquareLogWeights, 2. * w);
  _effectiveSampleSize = std::exp(2. * _sumLogWeights - _sumSquareLogWeights);
}

bool Nested::insideUnitCube(const std::vector<double> &sample) const
{
  // Check if sample exceeds bounds of cube in any dimension
  for (auto &s : sample)
  {
    if (s < 0.) return false;
    if (s > 1.) return false;
  }
  return true;
}

void Nested::printGenerationBefore() { return; }

void Nested::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Log Evidence: %.4f (+- %.4f)\n", _logEvidence, sqrt(_logEvidenceVar));
  _k->_logger->logInfo("Minimal", "Sampling Efficiency: %.2f%%\n", 100.0 * _acceptedSamples / ((double)(_generatedSamples - _numberLivePoints)));
  _k->_logger->logInfo("Detailed", "Last Accepted: %zu\n", _lastAccepted);
  _k->_logger->logInfo("Detailed", "Effective Sample Size: %.2f\n", _effectiveSampleSize);
  _k->_logger->logInfo("Detailed", "Log Volume (shrinkage): %.2f/%.2f (%.2f%%)\n", _logVolume, _boundLogVolume, 100. * (1. - std::exp(_logVolume)));
  _k->_logger->logInfo("Normal", "lStar: %.2f (max llk evaluation %.2f)\n", _lStar, _maxEvaluation);
  _k->_logger->logInfo("Minimal", "Remaining Log Evidence: %.2f (dlogz: %.3f)\n", _remainingLogEvidence, _logEvidenceDifference);
  if (_resamplingMethod == "Multi Ellipse")
  {
    _k->_logger->logInfo("Detailed", "Num Ellipsoids: %zu\n", _ellipseVector.size());
  }
}

void Nested::finalize()
{
  if (_k->_currentGeneration <= 1) return;
  if (_addLivePoints == true) consumeLiveSamples();

  (*_k)["Results"]["LogEvidence"] = _logEvidence;
  generatePosterior();

  _k->_logger->logInfo("Minimal", "Final Log Evidence: %.4f (+- %.4F)\n", _logEvidence, std::sqrt(_logEvidenceVar));
  _k->_logger->logInfo("Minimal", "Max evaluation: %.2f\n", _maxEvaluation);
  _k->_logger->logInfo("Minimal", "Sampling Efficiency: %.2f%%\n", 100.0 * _acceptedSamples / ((double)(_generatedSamples - _numberLivePoints)));
}

void Nested::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Normal Generator"))
 {
 _normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(js["Normal Generator"], _k));
 _normalGenerator->applyVariableDefaults();
 _normalGenerator->applyModuleDefaults(js["Normal Generator"]);
 _normalGenerator->setConfiguration(js["Normal Generator"]);
   eraseValue(js, "Normal Generator");
 }

 if (isDefined(js, "Multivariate Generator"))
 {
 _multivariateGenerator = dynamic_cast<korali::distribution::multivariate::Normal*>(korali::Module::getModule(js["Multivariate Generator"], _k));
 _multivariateGenerator->applyVariableDefaults();
 _multivariateGenerator->applyModuleDefaults(js["Multivariate Generator"]);
 _multivariateGenerator->setConfiguration(js["Multivariate Generator"]);
   eraseValue(js, "Multivariate Generator");
 }

 if (isDefined(js, "Accepted Samples"))
 {
 try { _acceptedSamples = js["Accepted Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Accepted Samples']\n%s", e.what()); } 
   eraseValue(js, "Accepted Samples");
 }

 if (isDefined(js, "Generated Samples"))
 {
 try { _generatedSamples = js["Generated Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Generated Samples']\n%s", e.what()); } 
   eraseValue(js, "Generated Samples");
 }

 if (isDefined(js, "LogEvidence"))
 {
 try { _logEvidence = js["LogEvidence"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LogEvidence']\n%s", e.what()); } 
   eraseValue(js, "LogEvidence");
 }

 if (isDefined(js, "LogEvidence Var"))
 {
 try { _logEvidenceVar = js["LogEvidence Var"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LogEvidence Var']\n%s", e.what()); } 
   eraseValue(js, "LogEvidence Var");
 }

 if (isDefined(js, "LogVolume"))
 {
 try { _logVolume = js["LogVolume"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LogVolume']\n%s", e.what()); } 
   eraseValue(js, "LogVolume");
 }

 if (isDefined(js, "Bound LogVolume"))
 {
 try { _boundLogVolume = js["Bound LogVolume"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Bound LogVolume']\n%s", e.what()); } 
   eraseValue(js, "Bound LogVolume");
 }

 if (isDefined(js, "Last Accepted"))
 {
 try { _lastAccepted = js["Last Accepted"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Last Accepted']\n%s", e.what()); } 
   eraseValue(js, "Last Accepted");
 }

 if (isDefined(js, "Next Update"))
 {
 try { _nextUpdate = js["Next Update"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Next Update']\n%s", e.what()); } 
   eraseValue(js, "Next Update");
 }

 if (isDefined(js, "Information"))
 {
 try { _information = js["Information"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Information']\n%s", e.what()); } 
   eraseValue(js, "Information");
 }

 if (isDefined(js, "LStar"))
 {
 try { _lStar = js["LStar"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LStar']\n%s", e.what()); } 
   eraseValue(js, "LStar");
 }

 if (isDefined(js, "LStarOld"))
 {
 try { _lStarOld = js["LStarOld"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LStarOld']\n%s", e.what()); } 
   eraseValue(js, "LStarOld");
 }

 if (isDefined(js, "LogWeight"))
 {
 try { _logWeight = js["LogWeight"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['LogWeight']\n%s", e.what()); } 
   eraseValue(js, "LogWeight");
 }

 if (isDefined(js, "Expected LogShrinkage"))
 {
 try { _expectedLogShrinkage = js["Expected LogShrinkage"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Expected LogShrinkage']\n%s", e.what()); } 
   eraseValue(js, "Expected LogShrinkage");
 }

 if (isDefined(js, "Max Evaluation"))
 {
 try { _maxEvaluation = js["Max Evaluation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Max Evaluation']\n%s", e.what()); } 
   eraseValue(js, "Max Evaluation");
 }

 if (isDefined(js, "Remaining Log Evidence"))
 {
 try { _remainingLogEvidence = js["Remaining Log Evidence"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Remaining Log Evidence']\n%s", e.what()); } 
   eraseValue(js, "Remaining Log Evidence");
 }

 if (isDefined(js, "Log Evidence Difference"))
 {
 try { _logEvidenceDifference = js["Log Evidence Difference"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Log Evidence Difference']\n%s", e.what()); } 
   eraseValue(js, "Log Evidence Difference");
 }

 if (isDefined(js, "Effective Sample Size"))
 {
 try { _effectiveSampleSize = js["Effective Sample Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Effective Sample Size']\n%s", e.what()); } 
   eraseValue(js, "Effective Sample Size");
 }

 if (isDefined(js, "Sum Log Weights"))
 {
 try { _sumLogWeights = js["Sum Log Weights"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Sum Log Weights']\n%s", e.what()); } 
   eraseValue(js, "Sum Log Weights");
 }

 if (isDefined(js, "Sum Square Log Weights"))
 {
 try { _sumSquareLogWeights = js["Sum Square Log Weights"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Sum Square Log Weights']\n%s", e.what()); } 
   eraseValue(js, "Sum Square Log Weights");
 }

 if (isDefined(js, "Prior Lower Bound"))
 {
 try { _priorLowerBound = js["Prior Lower Bound"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Prior Lower Bound']\n%s", e.what()); } 
   eraseValue(js, "Prior Lower Bound");
 }

 if (isDefined(js, "Prior Width"))
 {
 try { _priorWidth = js["Prior Width"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Prior Width']\n%s", e.what()); } 
   eraseValue(js, "Prior Width");
 }

 if (isDefined(js, "Candidates"))
 {
 try { _candidates = js["Candidates"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Candidates']\n%s", e.what()); } 
   eraseValue(js, "Candidates");
 }

 if (isDefined(js, "Candidate LogLikelihoods"))
 {
 try { _candidateLogLikelihoods = js["Candidate LogLikelihoods"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Candidate LogLikelihoods']\n%s", e.what()); } 
   eraseValue(js, "Candidate LogLikelihoods");
 }

 if (isDefined(js, "Candidate LogPriors"))
 {
 try { _candidateLogPriors = js["Candidate LogPriors"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Candidate LogPriors']\n%s", e.what()); } 
   eraseValue(js, "Candidate LogPriors");
 }

 if (isDefined(js, "Candidate LogPrior Weights"))
 {
 try { _candidateLogPriorWeights = js["Candidate LogPrior Weights"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Candidate LogPrior Weights']\n%s", e.what()); } 
   eraseValue(js, "Candidate LogPrior Weights");
 }

 if (isDefined(js, "Live Samples"))
 {
 try { _liveSamples = js["Live Samples"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live Samples']\n%s", e.what()); } 
   eraseValue(js, "Live Samples");
 }

 if (isDefined(js, "Live LogLikelihoods"))
 {
 try { _liveLogLikelihoods = js["Live LogLikelihoods"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live LogLikelihoods']\n%s", e.what()); } 
   eraseValue(js, "Live LogLikelihoods");
 }

 if (isDefined(js, "Live LogPriors"))
 {
 try { _liveLogPriors = js["Live LogPriors"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live LogPriors']\n%s", e.what()); } 
   eraseValue(js, "Live LogPriors");
 }

 if (isDefined(js, "Live LogPrior Weights"))
 {
 try { _liveLogPriorWeights = js["Live LogPrior Weights"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live LogPrior Weights']\n%s", e.what()); } 
   eraseValue(js, "Live LogPrior Weights");
 }

 if (isDefined(js, "Live Samples Rank"))
 {
 try { _liveSamplesRank = js["Live Samples Rank"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Live Samples Rank']\n%s", e.what()); } 
   eraseValue(js, "Live Samples Rank");
 }

 if (isDefined(js, "Number Dead Samples"))
 {
 try { _numberDeadSamples = js["Number Dead Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Number Dead Samples']\n%s", e.what()); } 
   eraseValue(js, "Number Dead Samples");
 }

 if (isDefined(js, "Dead Samples"))
 {
 try { _deadSamples = js["Dead Samples"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Dead Samples']\n%s", e.what()); } 
   eraseValue(js, "Dead Samples");
 }

 if (isDefined(js, "Dead LogLikelihoods"))
 {
 try { _deadLogLikelihoods = js["Dead LogLikelihoods"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Dead LogLikelihoods']\n%s", e.what()); } 
   eraseValue(js, "Dead LogLikelihoods");
 }

 if (isDefined(js, "Dead LogPriors"))
 {
 try { _deadLogPriors = js["Dead LogPriors"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Dead LogPriors']\n%s", e.what()); } 
   eraseValue(js, "Dead LogPriors");
 }

 if (isDefined(js, "Dead LogPrior Weights"))
 {
 try { _deadLogPriorWeights = js["Dead LogPrior Weights"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Dead LogPrior Weights']\n%s", e.what()); } 
   eraseValue(js, "Dead LogPrior Weights");
 }

 if (isDefined(js, "Dead LogWeights"))
 {
 try { _deadLogWeights = js["Dead LogWeights"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Dead LogWeights']\n%s", e.what()); } 
   eraseValue(js, "Dead LogWeights");
 }

 if (isDefined(js, "Covariance Matrix"))
 {
 try { _covarianceMatrix = js["Covariance Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Covariance Matrix");
 }

 if (isDefined(js, "Log Domain Size"))
 {
 try { _logDomainSize = js["Log Domain Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Log Domain Size']\n%s", e.what()); } 
   eraseValue(js, "Log Domain Size");
 }

 if (isDefined(js, "Domain Mean"))
 {
 try { _domainMean = js["Domain Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Domain Mean']\n%s", e.what()); } 
   eraseValue(js, "Domain Mean");
 }

 if (isDefined(js, "Box Lower Bound"))
 {
 try { _boxLowerBound = js["Box Lower Bound"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Box Lower Bound']\n%s", e.what()); } 
   eraseValue(js, "Box Lower Bound");
 }

 if (isDefined(js, "Box Upper Bound"))
 {
 try { _boxUpperBound = js["Box Upper Bound"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Box Upper Bound']\n%s", e.what()); } 
   eraseValue(js, "Box Upper Bound");
 }

 if (isDefined(js, "Ellipse Axes"))
 {
 try { _ellipseAxes = js["Ellipse Axes"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Ellipse Axes']\n%s", e.what()); } 
   eraseValue(js, "Ellipse Axes");
 }

 if (isDefined(js, "Number Live Points"))
 {
 try { _numberLivePoints = js["Number Live Points"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Number Live Points']\n%s", e.what()); } 
   eraseValue(js, "Number Live Points");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Number Live Points'] required by Nested.\n"); 

 if (isDefined(js, "Batch Size"))
 {
 try { _batchSize = js["Batch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Batch Size']\n%s", e.what()); } 
   eraseValue(js, "Batch Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Batch Size'] required by Nested.\n"); 

 if (isDefined(js, "Add Live Points"))
 {
 try { _addLivePoints = js["Add Live Points"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Add Live Points']\n%s", e.what()); } 
   eraseValue(js, "Add Live Points");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Add Live Points'] required by Nested.\n"); 

 if (isDefined(js, "Resampling Method"))
 {
 try { _resamplingMethod = js["Resampling Method"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Resampling Method']\n%s", e.what()); } 
   eraseValue(js, "Resampling Method");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Resampling Method'] required by Nested.\n"); 

 if (isDefined(js, "Proposal Update Frequency"))
 {
 try { _proposalUpdateFrequency = js["Proposal Update Frequency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Proposal Update Frequency']\n%s", e.what()); } 
   eraseValue(js, "Proposal Update Frequency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Proposal Update Frequency'] required by Nested.\n"); 

 if (isDefined(js, "Ellipsoidal Scaling"))
 {
 try { _ellipsoidalScaling = js["Ellipsoidal Scaling"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Ellipsoidal Scaling']\n%s", e.what()); } 
   eraseValue(js, "Ellipsoidal Scaling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Ellipsoidal Scaling'] required by Nested.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Log Evidence Delta"))
 {
 try { _minLogEvidenceDelta = js["Termination Criteria"]["Min Log Evidence Delta"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Termination Criteria']['Min Log Evidence Delta']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Log Evidence Delta");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Log Evidence Delta'] required by Nested.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Effective Sample Size"))
 {
 try { _maxEffectiveSampleSize = js["Termination Criteria"]["Max Effective Sample Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Termination Criteria']['Max Effective Sample Size']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Effective Sample Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Effective Sample Size'] required by Nested.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Log Likelihood"))
 {
 try { _maxLogLikelihood = js["Termination Criteria"]["Max Log Likelihood"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ Nested ] \n + Key:    ['Termination Criteria']['Max Log Likelihood']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Log Likelihood");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Log Likelihood'] required by Nested.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Sampler::setConfiguration(js);
 _type = "sampler/Nested";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: Nested: \n%s\n", js.dump(2).c_str());
} 

void Nested::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Number Live Points"] = _numberLivePoints;
   js["Batch Size"] = _batchSize;
   js["Add Live Points"] = _addLivePoints;
   js["Resampling Method"] = _resamplingMethod;
   js["Proposal Update Frequency"] = _proposalUpdateFrequency;
   js["Ellipsoidal Scaling"] = _ellipsoidalScaling;
   js["Termination Criteria"]["Min Log Evidence Delta"] = _minLogEvidenceDelta;
   js["Termination Criteria"]["Max Effective Sample Size"] = _maxEffectiveSampleSize;
   js["Termination Criteria"]["Max Log Likelihood"] = _maxLogLikelihood;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_multivariateGenerator != NULL) _multivariateGenerator->getConfiguration(js["Multivariate Generator"]);
   js["Accepted Samples"] = _acceptedSamples;
   js["Generated Samples"] = _generatedSamples;
   js["LogEvidence"] = _logEvidence;
   js["LogEvidence Var"] = _logEvidenceVar;
   js["LogVolume"] = _logVolume;
   js["Bound LogVolume"] = _boundLogVolume;
   js["Last Accepted"] = _lastAccepted;
   js["Next Update"] = _nextUpdate;
   js["Information"] = _information;
   js["LStar"] = _lStar;
   js["LStarOld"] = _lStarOld;
   js["LogWeight"] = _logWeight;
   js["Expected LogShrinkage"] = _expectedLogShrinkage;
   js["Max Evaluation"] = _maxEvaluation;
   js["Remaining Log Evidence"] = _remainingLogEvidence;
   js["Log Evidence Difference"] = _logEvidenceDifference;
   js["Effective Sample Size"] = _effectiveSampleSize;
   js["Sum Log Weights"] = _sumLogWeights;
   js["Sum Square Log Weights"] = _sumSquareLogWeights;
   js["Prior Lower Bound"] = _priorLowerBound;
   js["Prior Width"] = _priorWidth;
   js["Candidates"] = _candidates;
   js["Candidate LogLikelihoods"] = _candidateLogLikelihoods;
   js["Candidate LogPriors"] = _candidateLogPriors;
   js["Candidate LogPrior Weights"] = _candidateLogPriorWeights;
   js["Live Samples"] = _liveSamples;
   js["Live LogLikelihoods"] = _liveLogLikelihoods;
   js["Live LogPriors"] = _liveLogPriors;
   js["Live LogPrior Weights"] = _liveLogPriorWeights;
   js["Live Samples Rank"] = _liveSamplesRank;
   js["Number Dead Samples"] = _numberDeadSamples;
   js["Dead Samples"] = _deadSamples;
   js["Dead LogLikelihoods"] = _deadLogLikelihoods;
   js["Dead LogPriors"] = _deadLogPriors;
   js["Dead LogPrior Weights"] = _deadLogPriorWeights;
   js["Dead LogWeights"] = _deadLogWeights;
   js["Covariance Matrix"] = _covarianceMatrix;
   js["Log Domain Size"] = _logDomainSize;
   js["Domain Mean"] = _domainMean;
   js["Box Lower Bound"] = _boxLowerBound;
   js["Box Upper Bound"] = _boxUpperBound;
   js["Ellipse Axes"] = _ellipseAxes;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Sampler::getConfiguration(js);
} 

void Nested::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Number Live Points\": 1500, \"Batch Size\": 1, \"Add Live Points\": true, \"Resampling Method\": \"Ellipse\", \"Proposal Update Frequency\": 1500, \"Ellipsoidal Scaling\": 1.0, \"Termination Criteria\": {\"Min Log Evidence Delta\": 0.01, \"Max Effective Sample Size\": 10000000.0, \"Max Log Likelihood\": 10000000.0}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Multivariate Generator\": {\"Type\": \"Multivariate/Normal\"}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Sampler::applyModuleDefaults(js);
} 

void Nested::applyVariableDefaults() 
{

 Sampler::applyVariableDefaults();
} 

bool Nested::checkTermination()
{
 bool hasFinished = false;

 if ((_k->_currentGeneration > 1) && (_logEvidenceDifference <= _minLogEvidenceDelta))
 {
  _terminationCriteria.push_back("Nested['Min Log Evidence Delta'] = " + std::to_string(_minLogEvidenceDelta) + ".");
  hasFinished = true;
 }

 if (_maxEffectiveSampleSize <= _effectiveSampleSize)
 {
  _terminationCriteria.push_back("Nested['Max Effective Sample Size'] = " + std::to_string(_maxEffectiveSampleSize) + ".");
  hasFinished = true;
 }

 if (_maxLogLikelihood <= _lStar)
 {
  _terminationCriteria.push_back("Nested['Max Log Likelihood'] = " + std::to_string(_maxLogLikelihood) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Sampler::checkTermination();
 return hasFinished;
}

;

} //sampler
} //solver
} //korali
;
