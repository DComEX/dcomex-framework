#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/solver/sampler/TMCMC/TMCMC.hpp"
#include "sample/sample.hpp"
#include <chrono>
#include <limits>
#include <numeric>

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics.h>
#include <math.h>

namespace korali
{
namespace solver
{
namespace sampler
{
;

void TMCMC::setInitialConfiguration()
{
  knlohmann::json problemConfig = (*_k)["Problem"];
  _variableCount = _k->_variables.size();

  if (_maxChainLength == 0) KORALI_LOG_ERROR("Max Chain Length must be greater 0.");
  if (_covarianceScaling <= 0.0) KORALI_LOG_ERROR("Covariance Scaling must be larger 0.0 (is %lf).\n", _covarianceScaling);

  // Allocating TMCMC memory
  _chainLeadersLogPriors.resize(_populationSize);
  _chainLeadersLogLikelihoods.resize(_populationSize);
  _chainLeaders.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _chainLeaders[i].resize(_variableCount);

  _meanTheta.resize(_variableCount);
  _covarianceMatrix.resize(_variableCount * _variableCount);

  _chainCandidatesLogPriors.resize(_populationSize);
  _chainCandidatesLogLikelihoods.resize(_populationSize);
  _chainCandidates.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _chainCandidates[i].resize(_variableCount);

  _chainLengths.resize(_populationSize);
  _currentChainStep.resize(_populationSize);
  _chainPendingEvaluation.resize(_populationSize);
  _chainPendingGradient.resize(_populationSize);

  if (_version == "mTMCMC")
  {
    if (iCompare((*_k)["Problem"]["Type"], "Bayesian/Reference") == false)
      KORALI_LOG_ERROR("mTMCMC works only for problems of type 'Bayesian/Reference'\n");

    if (_maxChainLength != 1) KORALI_LOG_ERROR("Current version of 'mTMCMC' supports only 'Max Chain Length' of 1 (BASIS).");
    if (_stepSize < 0.0) KORALI_LOG_ERROR("Step Size lower than 0.0 (is %lf)\n", _stepSize);
    if (_domainExtensionFactor < 0.0) KORALI_LOG_ERROR("Domain Extension Factor lower than 0.0 (is %lf)\n", _domainExtensionFactor);

    _chainCandidatesErrors.resize(_populationSize, -1);
    _chainLeadersErrors.resize(_populationSize, -1);
    _sampleErrorDatabase.resize(_populationSize, -1);
    _chainCandidatesGradients.resize(_populationSize);
    for (size_t i = 0; i < _populationSize; i++) _chainCandidatesGradients[i].resize(_variableCount);
    _chainLeadersGradients.resize(_populationSize);
    for (size_t i = 0; i < _populationSize; i++) _chainLeadersGradients[i].resize(_variableCount);
    _sampleGradientDatabase.resize(_populationSize);
    for (size_t i = 0; i < _populationSize; i++) _sampleGradientDatabase[i].resize(_variableCount);
    _chainCandidatesCovariance.resize(_populationSize);
    for (size_t i = 0; i < _populationSize; i++) _chainCandidatesCovariance[i].resize(_variableCount * _variableCount);
    _chainLeadersCovariance.resize(_populationSize);
    for (size_t i = 0; i < _populationSize; i++) _chainLeadersCovariance[i].resize(_variableCount * _variableCount);
    _sampleCovarianceDatabase.resize(_populationSize);
    for (size_t i = 0; i < _populationSize; i++) _sampleCovarianceDatabase[i].resize(_variableCount);

    _upperExtendedBoundaries.resize(_variableCount);
    _lowerExtendedBoundaries.resize(_variableCount);

    for (size_t d = 0; d < _variableCount; ++d)
    {
      if (iCompare(_k->_distributions[_k->_variables[d]->_distributionIndex]->_type, "Univariate/Uniform") == false) KORALI_LOG_ERROR("Only 'Univariate/Uniform' priors allowed (is %s).\n", _k->_distributions[_k->_variables[d]->_distributionIndex]->_type.c_str());
      double width = dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_maximum - dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_minimum;
      _upperExtendedBoundaries[d] = dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_maximum + width * _domainExtensionFactor;
      _lowerExtendedBoundaries[d] = dynamic_cast<distribution::univariate::Uniform *>(_k->_distributions[_k->_variables[d]->_distributionIndex])->_minimum - width * _domainExtensionFactor;
    }
  }

  // Init
  _annealingExponent = 0.0;
  _currentAccumulatedLogEvidence = 0.0;
  _coefficientOfVariation = 0.0;
  _maxLoglikelihood = -Inf;
  _chainCount = _populationSize;

  _numCovarianceCorrections = 0;

  _numFinitePriorEvaluations = 0;
  _numFiniteLikelihoodEvaluations = 0;

  _numNegativeDefiniteProposals = 0;
  _numLUDecompositionFailuresProposal = 0;
  _numEigenDecompositionFailuresProposal = 0;
  _numInversionFailuresProposal = 0;
  _numCholeskyDecompositionFailuresProposal = 0;

  // Initializing Chain Length first Generation
  std::fill(std::begin(_chainLengths), std::end(_chainLengths), 1);
}

void TMCMC::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  prepareGeneration();
  std::vector<Sample> samples(_chainCount);

  while (_finishedChainsCount < _chainCount)
  {
    for (size_t c = 0; c < _chainCount; c++)
    {
      if (_currentChainStep[c] < _chainLengths[c] + _currentBurnIn)
        if (_chainPendingEvaluation[c] == false)
        {
          _chainPendingEvaluation[c] = true;
          samples[c]["Module"] = "Problem";
          samples[c]["Operation"] = "Evaluate";
          samples[c]["Parameters"] = _chainCandidates[c];
          samples[c]["Sample Id"] = c;
          _currentChainStep[c]++;
          _modelEvaluationCount++;
          KORALI_START(samples[c]);
        }
    }

    size_t finishedId = KORALI_WAITANY(samples);
    _chainPendingEvaluation[finishedId] = false;

    _chainCandidatesLogLikelihoods[finishedId] = KORALI_GET(double, samples[finishedId], "logLikelihood");
    _chainCandidatesLogPriors[finishedId] = KORALI_GET(double, samples[finishedId], "logPrior");

    if (isfinite(_chainCandidatesLogPriors[finishedId])) _numFinitePriorEvaluations++;
    if (isfinite(_chainCandidatesLogLikelihoods[finishedId])) _numFiniteLikelihoodEvaluations++;

    if (_version == "TMCMC") processCandidate(finishedId);
    if (_currentChainStep[finishedId] == _chainLengths[finishedId] + _currentBurnIn) _finishedChainsCount++;
  }

  if (_version == "mTMCMC")
  {
    if (_k->_currentGeneration > 1)
    {
      calculateGradients(samples);
      calculateProposals(samples);
    }
    for (size_t sampleId = 0; sampleId < _populationSize; ++sampleId) processCandidate(sampleId);
  }

  processGeneration();
}

void TMCMC::prepareGeneration()
{
  setBurnIn();

  _acceptedSamplesCount = 0;
  _finishedChainsCount = 0;
  _maxLoglikelihood = -std::numeric_limits<double>::infinity();

  _sampleLogLikelihoodDatabase.clear();
  _sampleLogPriorDatabase.clear();
  _sampleDatabase.clear();

  _numFinitePriorEvaluations = 0;
  _numFiniteLikelihoodEvaluations = 0;

  if (_version == "mTMCMC")
  {
    _numCovarianceCorrections = 0;
    _numNegativeDefiniteProposals = 0;
    _numLUDecompositionFailuresProposal = 0;
    _numEigenDecompositionFailuresProposal = 0;
    _numInversionFailuresProposal = 0;
    _numCholeskyDecompositionFailuresProposal = 0;

    _sampleErrorDatabase.clear();
    _sampleGradientDatabase.clear();
    _sampleCovarianceDatabase.clear();
    if (_k->_currentGeneration > 1)
      std::fill(_chainCandidatesErrors.begin(), _chainCandidatesErrors.end(), 0);
    else
      std::fill(_chainCandidatesErrors.begin(), _chainCandidatesErrors.end(), -1);

    // annealing
    for (size_t i = 0; i < _populationSize; ++i)
    {
      gsl_matrix_view covLeader = gsl_matrix_view_array(&_chainLeadersCovariance[i][0], _variableCount, _variableCount);
      gsl_matrix_scale(&covLeader.matrix, _previousAnnealingExponent / _annealingExponent);

      gsl_vector_view gradLeader = gsl_vector_view_array(&_chainLeadersGradients[i][0], _variableCount);
      gsl_vector_scale(&gradLeader.vector, _annealingExponent / _previousAnnealingExponent);
    }
  }

  std::fill(_currentChainStep.begin(), _currentChainStep.end(), 0);
  std::fill(_chainPendingEvaluation.begin(), _chainPendingEvaluation.end(), false);

  std::vector<double> zeroMean(_variableCount, 0.0);
  _multivariateGenerator->_meanVector = zeroMean;
  _multivariateGenerator->_sigma = _covarianceMatrix;

  /* Cholesky Decomp */
  gsl_matrix_view sigma = gsl_matrix_view_array(&_multivariateGenerator->_sigma[0], _variableCount, _variableCount);
  gsl_linalg_cholesky_decomp(&sigma.matrix);

  _multivariateGenerator->updateDistribution();

  for (size_t i = 0; i < _populationSize; i++)
  {
    if (_k->_currentGeneration == 1)
    {
      for (size_t d = 0; d < _variableCount; d++)
        _chainCandidates[i][d] = _k->_distributions[_k->_variables[d]->_distributionIndex]->getRandomNumber();
    }
    else
    {
      generateCandidate(i);
    }
  }
}

void TMCMC::processCandidate(const size_t sampleId)
{
  double P = calculateAcceptanceProbability(sampleId);
  double U = _uniformGenerator->getRandomNumber();

  if (P > U || _k->_currentGeneration == 1)
  {
    _chainLeaders[sampleId] = _chainCandidates[sampleId];
    _chainLeadersLogPriors[sampleId] = _chainCandidatesLogPriors[sampleId];
    _chainLeadersLogLikelihoods[sampleId] = _chainCandidatesLogLikelihoods[sampleId];
    if (_version == "mTMCMC")
    {
      _chainLeadersErrors[sampleId] = _chainCandidatesErrors[sampleId];
      _chainLeadersGradients[sampleId] = _chainCandidatesGradients[sampleId];
      _chainLeadersCovariance[sampleId] = _chainCandidatesCovariance[sampleId];
    }

    if (_currentChainStep[sampleId] > _currentBurnIn) _acceptedSamplesCount++;
  }

  if (_currentChainStep[sampleId] < _currentBurnIn + _chainLengths[sampleId]) generateCandidate(sampleId);

  if (_currentChainStep[sampleId] > _currentBurnIn) updateDatabase(sampleId);
}

void TMCMC::processGeneration()
{
  // Compute annealing exponent for next generation
  double fmin = 0, xmin = 0;
  minSearch(_sampleLogLikelihoodDatabase.data(), _populationSize, _annealingExponent, _targetCoefficientOfVariation, xmin, fmin);

  _previousAnnealingExponent = _annealingExponent;

  if (xmin > _previousAnnealingExponent + _maxAnnealingExponentUpdate)
  {
    _k->_logger->logWarning("Normal", "Annealing Step larger than Max Rho Update, updating Annealing Exponent by %f (Max Rho Update). \n", _maxAnnealingExponentUpdate);
    _annealingExponent = _previousAnnealingExponent + _maxAnnealingExponentUpdate;
    _coefficientOfVariation = sqrt(calculateSquaredCVDifference(_annealingExponent, _sampleLogLikelihoodDatabase.data(), _populationSize, _previousAnnealingExponent, _targetCoefficientOfVariation)) + _targetCoefficientOfVariation;
  }
  else if (xmin < 1.0 && xmin < _previousAnnealingExponent + _minAnnealingExponentUpdate)
  {
    _k->_logger->logWarning("Normal", "Annealing Step smaller than Min Rho Update, updating Annealing Exponent by %f (Min Rho Update). \n", _minAnnealingExponentUpdate);
    _annealingExponent = _previousAnnealingExponent + _minAnnealingExponentUpdate;
    _coefficientOfVariation = sqrt(calculateSquaredCVDifference(_annealingExponent, &_sampleLogLikelihoodDatabase[0], _populationSize, _previousAnnealingExponent, _targetCoefficientOfVariation)) + _targetCoefficientOfVariation;
  }
  else
  {
    _annealingExponent = xmin;
    _coefficientOfVariation = sqrt(fmin) + _targetCoefficientOfVariation;
  }

  /* Compute weights and normalize*/
  std::vector<double> log_weight(_populationSize);
  std::vector<double> weight(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) log_weight[i] = _sampleLogLikelihoodDatabase[i] * (_annealingExponent - _previousAnnealingExponent);

  const double loglikemax = gsl_stats_max(log_weight.data(), 1, _populationSize);
  for (size_t i = 0; i < _populationSize; i++) weight[i] = exp(log_weight[i] - loglikemax);

  double sum_weight = std::accumulate(weight.begin(), weight.end(), 0.0);
  for (size_t i = 0; i < _populationSize; i++) weight[i] = weight[i] / sum_weight;

  _currentAccumulatedLogEvidence += log(sum_weight) + loglikemax - log(_populationSize);

  /* Sample candidate selections based on database entries */
  std::vector<unsigned int> numselections(_populationSize);
  _multinomialGenerator->getSelections(weight, numselections, _populationSize);

  /* scale weights with number repeated samples */
  for (size_t i = 0; i < _populationSize; i++) weight[i] = weight[i] * numselections[i];

  sum_weight = std::accumulate(weight.begin(), weight.end(), 0.0);

  for (size_t i = 0; i < _populationSize; i++) weight[i] = weight[i] / sum_weight;

  double sum_weight2 = 0.0;
  for (size_t i = 0; i < _populationSize; i++) sum_weight2 += weight[i] * weight[i];

  /* Update mean and covariance */
  for (size_t i = 0; i < _variableCount; i++)
  {
    _meanTheta[i] = 0;
    for (size_t j = 0; j < _populationSize; j++) _meanTheta[i] += _sampleDatabase[j][i] * weight[j];
  }

  for (size_t i = 0; i < _variableCount; i++)
  {
    for (size_t j = i; j < _variableCount; ++j)
    {
      double s = 0.0;
      for (size_t k = 0; k < _populationSize; ++k)
        s += weight[k] * (_sampleDatabase[k][i] - _meanTheta[i]) * (_sampleDatabase[k][j] - _meanTheta[j]);
      _covarianceMatrix[i * _variableCount + j] = _covarianceMatrix[j * _variableCount + i] = _covarianceScaling * s / (1.0 - sum_weight2);
    }
  }

  /* Resampling - Init new chains */
  std::fill(std::begin(_chainLengths), std::end(_chainLengths), 0);

  size_t leaderChainLen;
  size_t zeroCount = 0;
  size_t leaderId = 0;
  for (size_t i = 0; i < _populationSize; i++)
  {
    if (numselections[i] == 0) zeroCount++;
    while (numselections[i] > 0)
    {
      _chainLeaders[leaderId] = _sampleDatabase[i];
      _chainLeadersLogPriors[leaderId] = _sampleLogPriorDatabase[i];
      _chainLeadersLogLikelihoods[leaderId] = _sampleLogLikelihoodDatabase[i];
      if (_version == "mTMCMC")
      {
        _chainLeadersErrors[leaderId] = _sampleErrorDatabase[i];
        _chainLeadersGradients[leaderId] = _sampleGradientDatabase[i];
        _chainLeadersCovariance[leaderId] = _sampleCovarianceDatabase[i];
      }

      if (numselections[i] > _maxChainLength)
      {
        /* uniform splitting of chains */
        size_t rest = (numselections[i] % _maxChainLength != 0);
        leaderChainLen = _maxChainLength - rest;
      }
      else
      {
        leaderChainLen = numselections[i];
      }
      _chainLengths[leaderId] = leaderChainLen;
      numselections[i] -= leaderChainLen;
      leaderId++;
    }
  }

  /* Anneal gradients and proposal */
  if (_version == "mTMCMC" && _previousAnnealingExponent > 0.0)
    for (size_t i = 0; i < _populationSize; ++i)
    {
      if (_chainLeadersErrors[i] != 0) continue;
      gsl_vector_view gradLeader = gsl_vector_view_array(&_chainLeadersGradients[i][0], _variableCount);
      gsl_vector_scale(&gradLeader.vector, _annealingExponent / _previousAnnealingExponent);

      gsl_matrix_view covLeader = gsl_matrix_view_array(&_chainLeadersCovariance[i][0], _variableCount, _variableCount);
      gsl_matrix_scale(&covLeader.matrix, _annealingExponent / _previousAnnealingExponent);
    }

  /* Update acceptance statistics */
  size_t uniqueSelections = _populationSize - zeroCount;
  _proposalsAcceptanceRate = (1.0 * _acceptedSamplesCount) / _populationSize;
  _selectionAcceptanceRate = (1.0 * uniqueSelections) / _populationSize;

  _maxLoglikelihood = *std::max_element(_sampleLogLikelihoodDatabase.begin(), _sampleLogLikelihoodDatabase.end());
  _chainCount = leaderId;
}

void TMCMC::calculateGradients(std::vector<Sample> &samples)
{
  size_t numGradientCalculations = 0.0;
  for (size_t c = 0; c < _chainCount; ++c)
  {
    if (!(std::isfinite(_chainCandidatesLogPriors[c]) && std::isfinite(_chainCandidatesLogLikelihoods[c]))) continue;
    samples[c]["Module"] = "Problem";
    samples[c]["Sample Id"] = c;
    samples[c]["Operation"] = "Evaluate Gradient";
    KORALI_START(samples[c]);
    numGradientCalculations++;
  }

  for (size_t c = 0; c < numGradientCalculations; c++)
  {
    size_t finishedId = KORALI_WAITANY(samples);

    _chainCandidatesGradients[finishedId] = KORALI_GET(std::vector<double>, samples[finishedId], "logLikelihood Gradient");
    for (size_t d = 0; d < _variableCount; ++d) _chainCandidatesGradients[finishedId][d] *= _annealingExponent;
  }
}

void TMCMC::calculateProposals(std::vector<Sample> &samples)
{
  size_t numFIMCalculations = 0.0;
  for (size_t c = 0; c < _chainCount; ++c)
  {
    if (!(std::isfinite(_chainCandidatesLogPriors[c]) && std::isfinite(_chainCandidatesLogLikelihoods[c]))) continue;
    samples[c]["Sample Id"] = c;
    samples[c]["Module"] = "Problem";
    samples[c]["Operation"] = "Evaluate Fisher Information";
    KORALI_START(samples[c]);
    numFIMCalculations++;
  }

  size_t Nth = _variableCount;
  gsl_vector *ccpy0 = gsl_vector_alloc(Nth);
  gsl_vector *ccpy1 = gsl_vector_alloc(Nth);

  for (size_t c = 0; c < numFIMCalculations; c++)
  {
    size_t finishedId = KORALI_WAITANY(samples);
    // printf("%s\n", samples[finishedId]._js.getJson().dump(2).c_str());

    // reset
    std::fill(_chainCandidatesCovariance[finishedId].begin(), _chainCandidatesCovariance[finishedId].end(), 0.0);

    std::vector<double> FIM = KORALI_GET(std::vector<double>, samples[finishedId], "Fisher Information");
    gsl_matrix_view FIMview = gsl_matrix_view_array(&FIM[0], Nth, Nth);

    // scale FIM
    gsl_matrix_scale(&FIMview.matrix, _annealingExponent);

    // invert FIM
    gsl_permutation *perm = gsl_permutation_alloc(Nth);

    int s;
    gsl_linalg_LU_decomp(&FIMview.matrix, perm, &s);

    // SM - Only add a check if you can create a unit test to trigger it
    //    if (status != GSL_SUCCESS)
    //    {
    //      _chainCandidatesErrors[finishedId] = 1;
    //      gsl_permutation_free(perm);
    //      _numLUDecompositionFailuresProposal++;
    //      continue;
    //    }

    gsl_matrix *FIMinv = gsl_matrix_alloc(Nth, Nth);
    gsl_linalg_LU_invert(&FIMview.matrix, perm, FIMinv);
    gsl_permutation_free(perm);

    // SM - Only add a check if you can create a unit test to trigger it
    //    if (status != GSL_SUCCESS)
    //    {
    //      _chainCandidatesErrors[finishedId] = 2;
    //      gsl_matrix_free(FIMinv);
    //      _numInversionFailuresProposal++;
    //      continue;
    //    }

    // eigenvalue decomposition
    gsl_vector *Evals = gsl_vector_alloc(Nth);
    gsl_matrix *Evecs = gsl_matrix_alloc(Nth, Nth);
    gsl_eigen_symmv_workspace *work = gsl_eigen_symmv_alloc(Nth);
    gsl_eigen_symmv(FIMinv, Evals, Evecs, work);
    gsl_eigen_symmv_free(work);

    // SM - Only add a check if you can create a unit test to trigger it
    //    if (status != GSL_SUCCESS)
    //    {
    //      _chainCandidatesErrors[finishedId] = 3;
    //      gsl_matrix_free(FIMinv);
    //      gsl_vector_free(Evals);
    //      gsl_matrix_free(Evecs);
    //      _numEigenDecompositionFailuresProposal++;
    //      continue;
    //    }

    gsl_vector_min(Evals);

    // SM - Only add a check if you can create a unit test to trigger it
    //    if (minEval <= 0.0)
    //    {
    //      //printf("minEval %lf\n", minEval);
    //      _chainCandidatesErrors[finishedId] = 4;
    //      gsl_matrix_free(FIMinv);
    //      gsl_vector_free(Evals);
    //      gsl_matrix_free(Evecs);
    //      _numNegativeDefiniteProposals++;
    //      continue;
    //    }

    // correction
    double correction = false;
    double distToUpper, distToLower;
    double len, chi2inv = gsl_cdf_chisq_Pinv(0.68, Nth);

    gsl_vector_const_view candidate = gsl_vector_const_view_array(&_chainCandidates[finishedId][0], Nth);

    for (size_t d = 0; d < Nth; ++d)
    {
      gsl_vector_memcpy(ccpy0, &candidate.vector);
      gsl_vector_memcpy(ccpy1, &candidate.vector);

      double scale = sqrt(gsl_vector_get(Evals, d) * chi2inv);
      double scaleBefore = scale;

      gsl_vector_const_view evec = gsl_matrix_const_column(Evecs, d);
      gsl_blas_daxpy(+1.0 * scale, &evec.vector, ccpy0);
      gsl_blas_daxpy(-1.0 * scale, &evec.vector, ccpy1);

      // measure overshoot & undershoot in all dims
      for (size_t e = 0; e < Nth; ++e)
      {
        distToUpper = _upperExtendedBoundaries[e] - _chainCandidates[finishedId][e];
        distToLower = _chainCandidates[finishedId][e] - _lowerExtendedBoundaries[e];

        len = gsl_vector_get(ccpy0, e) - _upperExtendedBoundaries[e];
        if (len > 0.) scale = std::min(scale, std::abs(1.0 / gsl_vector_get(&evec.vector, e) * distToUpper));

        len = _lowerExtendedBoundaries[e] - gsl_vector_get(ccpy0, e);
        if (len > 0.) scale = std::min(scale, std::abs(1.0 / gsl_vector_get(&evec.vector, e) * distToLower));

        len = gsl_vector_get(ccpy1, e) - _upperExtendedBoundaries[e];
        if (len > 0.) scale = std::min(scale, std::abs(1.0 / gsl_vector_get(&evec.vector, e) * distToUpper));

        len = _lowerExtendedBoundaries[e] - gsl_vector_get(ccpy1, e);
        if (len > 0.) scale = std::min(scale, std::abs(1.0 / gsl_vector_get(&evec.vector, e) * distToLower));
      }

      // scale evals
      gsl_vector_set(Evals, d, scale * scale / chi2inv);
      if (scaleBefore != scale) correction = true;
    }

    if (correction) _numCovarianceCorrections++;

    // construct & store
    for (size_t d = 0; d < Nth; ++d)
    {
      gsl_vector_view evec = gsl_matrix_column(Evecs, d);
      gsl_vector_scale(&evec.vector, sqrt(gsl_vector_get(Evals, d)));
    }

    gsl_matrix_view candidatecov = gsl_matrix_view_array(&_chainCandidatesCovariance[finishedId][0], Nth, Nth);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Evecs, Evecs, 0.0, &candidatecov.matrix);

    gsl_matrix_free(Evecs);
    gsl_vector_free(Evals);
    gsl_matrix_free(FIMinv);
  }

  gsl_vector_free(ccpy0);
  gsl_vector_free(ccpy1);
}

void TMCMC::generateCandidate(const size_t sampleId)
{
  if (_version == "TMCMC")
  {
    _multivariateGenerator->getRandomVector(&_chainCandidates[sampleId][0], _variableCount);
    for (size_t d = 0; d < _variableCount; d++) _chainCandidates[sampleId][d] += _chainLeaders[sampleId][d];
  }
  else /* "mTMCMC" */
  {
    // TODO: refine error treatment granularity
    if (_chainLeadersErrors[sampleId] == 0)
    {
      _multivariateGenerator->_meanVector = _chainLeaders[sampleId];
      for (size_t i = 0; i < _variableCount * _variableCount; ++i) _multivariateGenerator->_sigma[i] = _chainLeadersCovariance[sampleId][i] * _stepSize;

      /* Cholesky Decomp */
      gsl_matrix_view sigma = gsl_matrix_view_array(&_multivariateGenerator->_sigma[0], _variableCount, _variableCount);
      int status = gsl_linalg_cholesky_decomp(&sigma.matrix);
      if (status == GSL_SUCCESS)
      {
        _multivariateGenerator->updateDistribution();
        _multivariateGenerator->getRandomVector(&_chainCandidates[sampleId][0], _variableCount);

        gsl_vector_view candidate = gsl_vector_view_array(&_chainCandidates[sampleId][0], _variableCount);
        gsl_vector_const_view leaderGrad = gsl_vector_const_view_array(&_chainLeadersGradients[sampleId][0], _variableCount);
        gsl_matrix_const_view leaderCov = gsl_matrix_const_view_array(&_chainLeadersCovariance[sampleId][0], _variableCount, _variableCount);
        gsl_blas_dgemv(CblasNoTrans, 0.5 * _stepSize, &leaderCov.matrix, &leaderGrad.vector, 1.0, &candidate.vector);
      }

      // SM - Only add a check if you can create a unit test to trigger it
      //      else
      //      {
      //        _numCholeskyDecompositionFailuresProposal++;
      //        _chainLeadersErrors[sampleId] = 5;
      //      }
    }
    if (_chainLeadersErrors[sampleId] != 0) /* error */
    {
      _multivariateGenerator->_meanVector = _chainLeaders[sampleId];
      _multivariateGenerator->_sigma = _covarianceMatrix;

      /* Cholesky Decomp */
      gsl_matrix_view sigma = gsl_matrix_view_array(&_multivariateGenerator->_sigma[0], _variableCount, _variableCount);
      gsl_linalg_cholesky_decomp(&sigma.matrix);

      _multivariateGenerator->updateDistribution();
      _multivariateGenerator->getRandomVector(&_chainCandidates[sampleId][0], _variableCount);
    }
  }
}

void TMCMC::updateDatabase(const size_t sampleId)
{
  _sampleDatabase.push_back(_chainLeaders[sampleId]);
  _sampleLogPriorDatabase.push_back(_chainLeadersLogPriors[sampleId]);
  _sampleLogLikelihoodDatabase.push_back(_chainLeadersLogLikelihoods[sampleId]);

  if (_version == "mTMCMC")
  {
    _sampleErrorDatabase.push_back(_chainLeadersErrors[sampleId]);
    _sampleGradientDatabase.push_back(_chainLeadersGradients[sampleId]);
    _sampleCovarianceDatabase.push_back(_chainLeadersCovariance[sampleId]);
  }
}

double TMCMC::calculateAcceptanceProbability(const size_t sampleId)
{
  double P = 0.0;
  if (std::isfinite(_chainCandidatesLogPriors[sampleId]) && std::isfinite(_chainCandidatesLogLikelihoods[sampleId]))
  {
    if (_version == "TMCMC")
    {
      P = exp((_chainCandidatesLogLikelihoods[sampleId] - _chainLeadersLogLikelihoods[sampleId]) * _annealingExponent + (_chainCandidatesLogPriors[sampleId] - _chainLeadersLogPriors[sampleId]));
    }
    else /* mTMCMC */
    {
      // TODO: refine error treatment granularity
      if ((_chainLeadersErrors[sampleId] == 0) && (_chainCandidatesErrors[sampleId] == 0))
      {
        gsl_vector_const_view leader = gsl_vector_const_view_array(&_chainLeaders[sampleId][0], _variableCount);
        gsl_vector_const_view candidate = gsl_vector_const_view_array(&_chainCandidates[sampleId][0], _variableCount);

        gsl_vector *meanLeader = gsl_vector_alloc(_variableCount);
        gsl_vector_memcpy(meanLeader, &leader.vector);

        gsl_vector *meanCandidate = gsl_vector_alloc(_variableCount);
        gsl_vector_memcpy(meanCandidate, &candidate.vector);

        gsl_vector_const_view gradLeader = gsl_vector_const_view_array(&_chainLeadersGradients[sampleId][0], _variableCount);
        gsl_vector_const_view gradCandidate = gsl_vector_const_view_array(&_chainCandidatesGradients[sampleId][0], _variableCount);

        gsl_matrix_const_view covLeader = gsl_matrix_const_view_array(&_chainLeadersCovariance[sampleId][0], _variableCount, _variableCount);

        gsl_blas_dgemv(CblasNoTrans, 0.5 * _stepSize, &covLeader.matrix, &gradLeader.vector, 1.0, meanLeader);
        gsl_blas_dgemv(CblasNoTrans, 0.5 * _stepSize, &covLeader.matrix, &gradCandidate.vector, 1.0, meanCandidate);

        gsl_matrix *cholCovLeader = gsl_matrix_alloc(_variableCount, _variableCount);
        gsl_matrix_memcpy(cholCovLeader, &covLeader.matrix);
        gsl_matrix_scale(cholCovLeader, _stepSize);
        gsl_linalg_cholesky_decomp1(cholCovLeader);

        gsl_vector *work = gsl_vector_alloc(_variableCount);

        double logpCandidate, logpLeader;
        gsl_ran_multivariate_gaussian_log_pdf(&candidate.vector, meanLeader, cholCovLeader, &logpCandidate, work);
        gsl_ran_multivariate_gaussian_log_pdf(&leader.vector, meanCandidate, cholCovLeader, &logpLeader, work);

        gsl_vector_free(work);
        gsl_matrix_free(cholCovLeader);
        gsl_vector_free(meanLeader);
        gsl_vector_free(meanCandidate);

        P = exp((_chainCandidatesLogLikelihoods[sampleId] - _chainLeadersLogLikelihoods[sampleId]) * _annealingExponent + (logpLeader - logpCandidate) + (_chainCandidatesLogPriors[sampleId] - _chainLeadersLogPriors[sampleId]));
      }
      else /* error */
      {
        P = exp((_chainCandidatesLogLikelihoods[sampleId] - _chainLeadersLogLikelihoods[sampleId]) * _annealingExponent + (_chainCandidatesLogPriors[sampleId] - _chainLeadersLogPriors[sampleId]));
      }
    }
  }
  return P;
}

double TMCMC::calculateSquaredCVDifference(double x, const double *loglike, size_t Ns, double exponent, double targetCOV)
{
  std::vector<double> weight(Ns);
  const double loglike_max = gsl_stats_max(loglike, 1, Ns);

  for (size_t i = 0; i < Ns; i++) weight[i] = exp((loglike[i] - loglike_max) * (x - exponent));

  double sum_weight = std::accumulate(weight.begin(), weight.end(), 0.0);

  for (size_t i = 0; i < Ns; i++) weight[i] = weight[i] / sum_weight;

  double mean = gsl_stats_mean(weight.data(), 1, Ns);
  double std = gsl_stats_sd_m(weight.data(), 1, Ns, mean);
  double cov2 = (std / mean) - targetCOV;
  cov2 *= cov2;

  if (isfinite(cov2) == false)
    return Lowest;
  else
    return cov2;
}

double TMCMC::calculateSquaredCVDifferenceOptimizationWrapper(const gsl_vector *v, void *param)
{
  double x = gsl_vector_get(v, 0);
  fparam_t *fp = (fparam_t *)param;
  return TMCMC::calculateSquaredCVDifference(x, fp->loglike, fp->Ns, fp->exponent, fp->cov);
}

void TMCMC::minSearch(double const *loglike, size_t Ns, double exponent, double objCov, double &xmin, double &fmin)
{
  // Minimizer Options
  const size_t MaxIter = 1000; /* Max number of search iterations */
  const double Tol = 1e-12;    /* Tolerance for root finding */
  const double Step = 1e-8;    /* Search stepsize */

  const gsl_multimin_fminimizer_type *T;
  gsl_multimin_fminimizer *s = NULL;
  gsl_vector *ss, *x;
  gsl_multimin_function minex_func;

  size_t iter = 0;
  int status;
  double size;

  fparam_t fp;
  fp.loglike = loglike;
  fp.Ns = Ns;
  fp.exponent = exponent;
  fp.cov = objCov;

  x = gsl_vector_alloc(1);
  gsl_vector_set(x, 0, exponent);

  ss = gsl_vector_alloc(1);
  gsl_vector_set_all(ss, Step);

  minex_func.n = 1;
  minex_func.f = calculateSquaredCVDifferenceOptimizationWrapper;
  minex_func.params = &fp;

  T = gsl_multimin_fminimizer_nmsimplex;
  s = gsl_multimin_fminimizer_alloc(T, 1);
  gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

  fmin = 0;
  xmin = 0.0;

  do
  {
    iter++;
    status = gsl_multimin_fminimizer_iterate(s);
    size = gsl_multimin_fminimizer_size(s);
    status = gsl_multimin_test_size(size, Tol);
  } while (status == GSL_CONTINUE && iter < MaxIter);

  if (status == GSL_SUCCESS && s->fval > Tol) _k->_logger->logWarning("Normal", "Min Search converged but did not find minimum. \n");
  if (status != GSL_SUCCESS && s->fval <= Tol) _k->_logger->logWarning("Normal", "Min Search did not converge but minimum found\n");
  if (status != GSL_SUCCESS && s->fval > Tol) _k->_logger->logWarning("Normal", "Min Search did not converge and did not find minimum\n");
  if (iter >= MaxIter) _k->_logger->logWarning("Normal", "Min Search MaxIter (%zu) reached\n", MaxIter);

  if (s->fval <= Tol)
  {
    fmin = s->fval;
    xmin = gsl_vector_get(s->x, 0);
  }

  if (xmin >= 1.0)
  {
    fmin = calculateSquaredCVDifference(1.0, loglike, Ns, exponent, objCov);
    xmin = 1.0;
  }

  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free(s);
}

void TMCMC::setBurnIn()
{
  if (_k->_currentGeneration <= 1)
    _currentBurnIn = 0;
  else if (_k->_currentGeneration - 2 < _perGenerationBurnIn.size())
    _currentBurnIn = _perGenerationBurnIn[_k->_currentGeneration - 2];
  else
    _currentBurnIn = _burnIn;
}

void TMCMC::finalize()
{
  // Setting results
  (*_k)["Results"]["Posterior Sample Database"] = _sampleDatabase;
  (*_k)["Results"]["Posterior Sample LogPrior Database"] = _sampleLogPriorDatabase;
  (*_k)["Results"]["Posterior Sample LogLikelihood Database"] = _sampleLogLikelihoodDatabase;
  (*_k)["Results"]["Log Evidence"] = _currentAccumulatedLogEvidence;
}

void TMCMC::printGenerationBefore()
{
  _k->_logger->logInfo("Minimal", "Annealing Exponent:          %.3e.\n", _annealingExponent);
}

void TMCMC::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Acceptance Rate (proposals / selections): (%.2f%% / %.2f%%)\n", 100 * _proposalsAcceptanceRate, 100 * _selectionAcceptanceRate);
  _k->_logger->logInfo("Normal", "Coefficient of Variation: %.2f%%\n", 100.0 * _coefficientOfVariation);
  _k->_logger->logInfo("Normal", "log of accumulated evidence: %.3f\n", _currentAccumulatedLogEvidence);
  _k->_logger->logInfo("Detailed", "max logLikelihood: %.3f\n", _maxLoglikelihood);
  _k->_logger->logInfo("Detailed", "Number of finite Evaluations (prior / likelihood): (%zu / %zu)\n", _numFinitePriorEvaluations, _numFiniteLikelihoodEvaluations);

  if (_version == "mTMCMC")
  {
    _k->_logger->logInfo("Normal", "Number Of Covariance Corrections: %zu\n", _numCovarianceCorrections);
    _k->_logger->logInfo("Detailed", "Number Of LU Decomposition Errors: %zu\n", _numLUDecompositionFailuresProposal);
    _k->_logger->logInfo("Detailed", "Number Of Eigenvalue Errors: %zu\n", _numEigenDecompositionFailuresProposal);
    _k->_logger->logInfo("Detailed", "Number Of FIM Inversion Errors: %zu\n", _numInversionFailuresProposal);
    _k->_logger->logInfo("Detailed", "Number Of Negative Definite Proposals: %zu\n", _numNegativeDefiniteProposals);
    _k->_logger->logInfo("Detailed", "Number Of Cholesky Decomposition Errors: %zu\n", _numCholeskyDecompositionFailuresProposal);
  }

  _k->_logger->logInfo("Detailed", "Sample Mean:\n");
  for (size_t i = 0; i < _variableCount; i++) _k->_logger->logData("Detailed", " %s = %+6.3e\n", _k->_variables[i]->_name.c_str(), _meanTheta[i]);
  _k->_logger->logInfo("Detailed", "Sample Covariance:\n");

  for (size_t i = 0; i < _variableCount; i++)
  {
    _k->_logger->logData("Detailed", "   | ");
    for (size_t j = 0; j < _variableCount; j++)
      if (j <= i)
        _k->_logger->logData("Detailed", "%+6.3e  ", _covarianceMatrix[i * _variableCount + j]);
      else
        _k->_logger->logData("Detailed", "     -      ");
    _k->_logger->logData("Detailed", " |\n");
  }
}

void TMCMC::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Multinomial Generator"))
 {
 _multinomialGenerator = dynamic_cast<korali::distribution::specific::Multinomial*>(korali::Module::getModule(js["Multinomial Generator"], _k));
 _multinomialGenerator->applyVariableDefaults();
 _multinomialGenerator->applyModuleDefaults(js["Multinomial Generator"]);
 _multinomialGenerator->setConfiguration(js["Multinomial Generator"]);
   eraseValue(js, "Multinomial Generator");
 }

 if (isDefined(js, "Multivariate Generator"))
 {
 _multivariateGenerator = dynamic_cast<korali::distribution::multivariate::Normal*>(korali::Module::getModule(js["Multivariate Generator"], _k));
 _multivariateGenerator->applyVariableDefaults();
 _multivariateGenerator->applyModuleDefaults(js["Multivariate Generator"]);
 _multivariateGenerator->setConfiguration(js["Multivariate Generator"]);
   eraseValue(js, "Multivariate Generator");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Current Burn In"))
 {
 try { _currentBurnIn = js["Current Burn In"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Current Burn In']\n%s", e.what()); } 
   eraseValue(js, "Current Burn In");
 }

 if (isDefined(js, "Chain Pending Evaluation"))
 {
 try { _chainPendingEvaluation = js["Chain Pending Evaluation"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Pending Evaluation']\n%s", e.what()); } 
   eraseValue(js, "Chain Pending Evaluation");
 }

 if (isDefined(js, "Chain Pending Gradient"))
 {
 try { _chainPendingGradient = js["Chain Pending Gradient"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Pending Gradient']\n%s", e.what()); } 
   eraseValue(js, "Chain Pending Gradient");
 }

 if (isDefined(js, "Chain Candidates"))
 {
 try { _chainCandidates = js["Chain Candidates"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Candidates']\n%s", e.what()); } 
   eraseValue(js, "Chain Candidates");
 }

 if (isDefined(js, "Chain Candidates LogLikelihoods"))
 {
 try { _chainCandidatesLogLikelihoods = js["Chain Candidates LogLikelihoods"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Candidates LogLikelihoods']\n%s", e.what()); } 
   eraseValue(js, "Chain Candidates LogLikelihoods");
 }

 if (isDefined(js, "Chain Candidates LogPriors"))
 {
 try { _chainCandidatesLogPriors = js["Chain Candidates LogPriors"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Candidates LogPriors']\n%s", e.what()); } 
   eraseValue(js, "Chain Candidates LogPriors");
 }

 if (isDefined(js, "Chain Candidates Gradients"))
 {
 try { _chainCandidatesGradients = js["Chain Candidates Gradients"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Candidates Gradients']\n%s", e.what()); } 
   eraseValue(js, "Chain Candidates Gradients");
 }

 if (isDefined(js, "Chain Candidates Errors"))
 {
 try { _chainCandidatesErrors = js["Chain Candidates Errors"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Candidates Errors']\n%s", e.what()); } 
   eraseValue(js, "Chain Candidates Errors");
 }

 if (isDefined(js, "Chain Candidates Covariance"))
 {
 try { _chainCandidatesCovariance = js["Chain Candidates Covariance"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Candidates Covariance']\n%s", e.what()); } 
   eraseValue(js, "Chain Candidates Covariance");
 }

 if (isDefined(js, "Chain Leaders"))
 {
 try { _chainLeaders = js["Chain Leaders"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Leaders']\n%s", e.what()); } 
   eraseValue(js, "Chain Leaders");
 }

 if (isDefined(js, "Chain Leaders LogLikelihoods"))
 {
 try { _chainLeadersLogLikelihoods = js["Chain Leaders LogLikelihoods"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Leaders LogLikelihoods']\n%s", e.what()); } 
   eraseValue(js, "Chain Leaders LogLikelihoods");
 }

 if (isDefined(js, "Chain Leaders LogPriors"))
 {
 try { _chainLeadersLogPriors = js["Chain Leaders LogPriors"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Leaders LogPriors']\n%s", e.what()); } 
   eraseValue(js, "Chain Leaders LogPriors");
 }

 if (isDefined(js, "Chain Leaders Gradients"))
 {
 try { _chainLeadersGradients = js["Chain Leaders Gradients"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Leaders Gradients']\n%s", e.what()); } 
   eraseValue(js, "Chain Leaders Gradients");
 }

 if (isDefined(js, "Chain Leaders Errors"))
 {
 try { _chainLeadersErrors = js["Chain Leaders Errors"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Leaders Errors']\n%s", e.what()); } 
   eraseValue(js, "Chain Leaders Errors");
 }

 if (isDefined(js, "Chain Leaders Covariance"))
 {
 try { _chainLeadersCovariance = js["Chain Leaders Covariance"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Leaders Covariance']\n%s", e.what()); } 
   eraseValue(js, "Chain Leaders Covariance");
 }

 if (isDefined(js, "Finished Chains Count"))
 {
 try { _finishedChainsCount = js["Finished Chains Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Finished Chains Count']\n%s", e.what()); } 
   eraseValue(js, "Finished Chains Count");
 }

 if (isDefined(js, "Current Chain Step"))
 {
 try { _currentChainStep = js["Current Chain Step"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Current Chain Step']\n%s", e.what()); } 
   eraseValue(js, "Current Chain Step");
 }

 if (isDefined(js, "Chain Lengths"))
 {
 try { _chainLengths = js["Chain Lengths"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Lengths']\n%s", e.what()); } 
   eraseValue(js, "Chain Lengths");
 }

 if (isDefined(js, "Coefficient Of Variation"))
 {
 try { _coefficientOfVariation = js["Coefficient Of Variation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Coefficient Of Variation']\n%s", e.what()); } 
   eraseValue(js, "Coefficient Of Variation");
 }

 if (isDefined(js, "Chain Count"))
 {
 try { _chainCount = js["Chain Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Chain Count']\n%s", e.what()); } 
   eraseValue(js, "Chain Count");
 }

 if (isDefined(js, "Annealing Exponent"))
 {
 try { _annealingExponent = js["Annealing Exponent"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Annealing Exponent']\n%s", e.what()); } 
   eraseValue(js, "Annealing Exponent");
 }

 if (isDefined(js, "Previous Annealing Exponent"))
 {
 try { _previousAnnealingExponent = js["Previous Annealing Exponent"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Previous Annealing Exponent']\n%s", e.what()); } 
   eraseValue(js, "Previous Annealing Exponent");
 }

 if (isDefined(js, "Num Finite Prior Evaluations"))
 {
 try { _numFinitePriorEvaluations = js["Num Finite Prior Evaluations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Num Finite Prior Evaluations']\n%s", e.what()); } 
   eraseValue(js, "Num Finite Prior Evaluations");
 }

 if (isDefined(js, "Num Finite Likelihood Evaluations"))
 {
 try { _numFiniteLikelihoodEvaluations = js["Num Finite Likelihood Evaluations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Num Finite Likelihood Evaluations']\n%s", e.what()); } 
   eraseValue(js, "Num Finite Likelihood Evaluations");
 }

 if (isDefined(js, "Accepted Samples Count"))
 {
 try { _acceptedSamplesCount = js["Accepted Samples Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Accepted Samples Count']\n%s", e.what()); } 
   eraseValue(js, "Accepted Samples Count");
 }

 if (isDefined(js, "Current Accumulated LogEvidence"))
 {
 try { _currentAccumulatedLogEvidence = js["Current Accumulated LogEvidence"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Current Accumulated LogEvidence']\n%s", e.what()); } 
   eraseValue(js, "Current Accumulated LogEvidence");
 }

 if (isDefined(js, "Proposals Acceptance Rate"))
 {
 try { _proposalsAcceptanceRate = js["Proposals Acceptance Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Proposals Acceptance Rate']\n%s", e.what()); } 
   eraseValue(js, "Proposals Acceptance Rate");
 }

 if (isDefined(js, "Selection Acceptance Rate"))
 {
 try { _selectionAcceptanceRate = js["Selection Acceptance Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Selection Acceptance Rate']\n%s", e.what()); } 
   eraseValue(js, "Selection Acceptance Rate");
 }

 if (isDefined(js, "Covariance Matrix"))
 {
 try { _covarianceMatrix = js["Covariance Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Covariance Matrix");
 }

 if (isDefined(js, "Max Loglikelihood"))
 {
 try { _maxLoglikelihood = js["Max Loglikelihood"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Max Loglikelihood']\n%s", e.what()); } 
   eraseValue(js, "Max Loglikelihood");
 }

 if (isDefined(js, "Mean Theta"))
 {
 try { _meanTheta = js["Mean Theta"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Mean Theta']\n%s", e.what()); } 
   eraseValue(js, "Mean Theta");
 }

 if (isDefined(js, "Sample Database"))
 {
 try { _sampleDatabase = js["Sample Database"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Sample Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Database");
 }

 if (isDefined(js, "Sample LogLikelihood Database"))
 {
 try { _sampleLogLikelihoodDatabase = js["Sample LogLikelihood Database"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Sample LogLikelihood Database']\n%s", e.what()); } 
   eraseValue(js, "Sample LogLikelihood Database");
 }

 if (isDefined(js, "Sample LogPrior Database"))
 {
 try { _sampleLogPriorDatabase = js["Sample LogPrior Database"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Sample LogPrior Database']\n%s", e.what()); } 
   eraseValue(js, "Sample LogPrior Database");
 }

 if (isDefined(js, "Sample Gradient Database"))
 {
 try { _sampleGradientDatabase = js["Sample Gradient Database"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Sample Gradient Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Gradient Database");
 }

 if (isDefined(js, "Sample Error Database"))
 {
 try { _sampleErrorDatabase = js["Sample Error Database"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Sample Error Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Error Database");
 }

 if (isDefined(js, "Sample Covariance Database"))
 {
 try { _sampleCovarianceDatabase = js["Sample Covariance Database"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Sample Covariance Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Covariance Database");
 }

 if (isDefined(js, "Upper Extended Boundaries"))
 {
 try { _upperExtendedBoundaries = js["Upper Extended Boundaries"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Upper Extended Boundaries']\n%s", e.what()); } 
   eraseValue(js, "Upper Extended Boundaries");
 }

 if (isDefined(js, "Lower Extended Boundaries"))
 {
 try { _lowerExtendedBoundaries = js["Lower Extended Boundaries"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Lower Extended Boundaries']\n%s", e.what()); } 
   eraseValue(js, "Lower Extended Boundaries");
 }

 if (isDefined(js, "Num LU Decomposition Failures Proposal"))
 {
 try { _numLUDecompositionFailuresProposal = js["Num LU Decomposition Failures Proposal"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Num LU Decomposition Failures Proposal']\n%s", e.what()); } 
   eraseValue(js, "Num LU Decomposition Failures Proposal");
 }

 if (isDefined(js, "Num Eigen Decomposition Failures Proposal"))
 {
 try { _numEigenDecompositionFailuresProposal = js["Num Eigen Decomposition Failures Proposal"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Num Eigen Decomposition Failures Proposal']\n%s", e.what()); } 
   eraseValue(js, "Num Eigen Decomposition Failures Proposal");
 }

 if (isDefined(js, "Num Inversion Failures Proposal"))
 {
 try { _numInversionFailuresProposal = js["Num Inversion Failures Proposal"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Num Inversion Failures Proposal']\n%s", e.what()); } 
   eraseValue(js, "Num Inversion Failures Proposal");
 }

 if (isDefined(js, "Num Negative Definite Proposals"))
 {
 try { _numNegativeDefiniteProposals = js["Num Negative Definite Proposals"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Num Negative Definite Proposals']\n%s", e.what()); } 
   eraseValue(js, "Num Negative Definite Proposals");
 }

 if (isDefined(js, "Num Cholesky Decomposition Failures Proposal"))
 {
 try { _numCholeskyDecompositionFailuresProposal = js["Num Cholesky Decomposition Failures Proposal"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Num Cholesky Decomposition Failures Proposal']\n%s", e.what()); } 
   eraseValue(js, "Num Cholesky Decomposition Failures Proposal");
 }

 if (isDefined(js, "Num Covariance Corrections"))
 {
 try { _numCovarianceCorrections = js["Num Covariance Corrections"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Num Covariance Corrections']\n%s", e.what()); } 
   eraseValue(js, "Num Covariance Corrections");
 }

 if (isDefined(js, "Version"))
 {
 try { _version = js["Version"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Version']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_version == "TMCMC") validOption = true; 
 if (_version == "mTMCMC") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Version'] required by TMCMC.\n", _version.c_str()); 
}
   eraseValue(js, "Version");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Version'] required by TMCMC.\n"); 

 if (isDefined(js, "Population Size"))
 {
 try { _populationSize = js["Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Population Size']\n%s", e.what()); } 
   eraseValue(js, "Population Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Population Size'] required by TMCMC.\n"); 

 if (isDefined(js, "Max Chain Length"))
 {
 try { _maxChainLength = js["Max Chain Length"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Max Chain Length']\n%s", e.what()); } 
   eraseValue(js, "Max Chain Length");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Max Chain Length'] required by TMCMC.\n"); 

 if (isDefined(js, "Burn In"))
 {
 try { _burnIn = js["Burn In"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Burn In']\n%s", e.what()); } 
   eraseValue(js, "Burn In");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Burn In'] required by TMCMC.\n"); 

 if (isDefined(js, "Per Generation Burn In"))
 {
 try { _perGenerationBurnIn = js["Per Generation Burn In"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Per Generation Burn In']\n%s", e.what()); } 
   eraseValue(js, "Per Generation Burn In");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Per Generation Burn In'] required by TMCMC.\n"); 

 if (isDefined(js, "Target Coefficient Of Variation"))
 {
 try { _targetCoefficientOfVariation = js["Target Coefficient Of Variation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Target Coefficient Of Variation']\n%s", e.what()); } 
   eraseValue(js, "Target Coefficient Of Variation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Target Coefficient Of Variation'] required by TMCMC.\n"); 

 if (isDefined(js, "Covariance Scaling"))
 {
 try { _covarianceScaling = js["Covariance Scaling"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Covariance Scaling']\n%s", e.what()); } 
   eraseValue(js, "Covariance Scaling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Covariance Scaling'] required by TMCMC.\n"); 

 if (isDefined(js, "Min Annealing Exponent Update"))
 {
 try { _minAnnealingExponentUpdate = js["Min Annealing Exponent Update"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Min Annealing Exponent Update']\n%s", e.what()); } 
   eraseValue(js, "Min Annealing Exponent Update");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Min Annealing Exponent Update'] required by TMCMC.\n"); 

 if (isDefined(js, "Max Annealing Exponent Update"))
 {
 try { _maxAnnealingExponentUpdate = js["Max Annealing Exponent Update"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Max Annealing Exponent Update']\n%s", e.what()); } 
   eraseValue(js, "Max Annealing Exponent Update");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Max Annealing Exponent Update'] required by TMCMC.\n"); 

 if (isDefined(js, "Step Size"))
 {
 try { _stepSize = js["Step Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Step Size']\n%s", e.what()); } 
   eraseValue(js, "Step Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Step Size'] required by TMCMC.\n"); 

 if (isDefined(js, "Domain Extension Factor"))
 {
 try { _domainExtensionFactor = js["Domain Extension Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Domain Extension Factor']\n%s", e.what()); } 
   eraseValue(js, "Domain Extension Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Domain Extension Factor'] required by TMCMC.\n"); 

 if (isDefined(js, "Termination Criteria", "Target Annealing Exponent"))
 {
 try { _targetAnnealingExponent = js["Termination Criteria"]["Target Annealing Exponent"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TMCMC ] \n + Key:    ['Termination Criteria']['Target Annealing Exponent']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Target Annealing Exponent");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Target Annealing Exponent'] required by TMCMC.\n"); 

 Sampler::setConfiguration(js);
 _type = "sampler/TMCMC";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: TMCMC: \n%s\n", js.dump(2).c_str());
} 

void TMCMC::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Version"] = _version;
   js["Population Size"] = _populationSize;
   js["Max Chain Length"] = _maxChainLength;
   js["Burn In"] = _burnIn;
   js["Per Generation Burn In"] = _perGenerationBurnIn;
   js["Target Coefficient Of Variation"] = _targetCoefficientOfVariation;
   js["Covariance Scaling"] = _covarianceScaling;
   js["Min Annealing Exponent Update"] = _minAnnealingExponentUpdate;
   js["Max Annealing Exponent Update"] = _maxAnnealingExponentUpdate;
   js["Step Size"] = _stepSize;
   js["Domain Extension Factor"] = _domainExtensionFactor;
   js["Termination Criteria"]["Target Annealing Exponent"] = _targetAnnealingExponent;
 if(_multinomialGenerator != NULL) _multinomialGenerator->getConfiguration(js["Multinomial Generator"]);
 if(_multivariateGenerator != NULL) _multivariateGenerator->getConfiguration(js["Multivariate Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Current Burn In"] = _currentBurnIn;
   js["Chain Pending Evaluation"] = _chainPendingEvaluation;
   js["Chain Pending Gradient"] = _chainPendingGradient;
   js["Chain Candidates"] = _chainCandidates;
   js["Chain Candidates LogLikelihoods"] = _chainCandidatesLogLikelihoods;
   js["Chain Candidates LogPriors"] = _chainCandidatesLogPriors;
   js["Chain Candidates Gradients"] = _chainCandidatesGradients;
   js["Chain Candidates Errors"] = _chainCandidatesErrors;
   js["Chain Candidates Covariance"] = _chainCandidatesCovariance;
   js["Chain Leaders"] = _chainLeaders;
   js["Chain Leaders LogLikelihoods"] = _chainLeadersLogLikelihoods;
   js["Chain Leaders LogPriors"] = _chainLeadersLogPriors;
   js["Chain Leaders Gradients"] = _chainLeadersGradients;
   js["Chain Leaders Errors"] = _chainLeadersErrors;
   js["Chain Leaders Covariance"] = _chainLeadersCovariance;
   js["Finished Chains Count"] = _finishedChainsCount;
   js["Current Chain Step"] = _currentChainStep;
   js["Chain Lengths"] = _chainLengths;
   js["Coefficient Of Variation"] = _coefficientOfVariation;
   js["Chain Count"] = _chainCount;
   js["Annealing Exponent"] = _annealingExponent;
   js["Previous Annealing Exponent"] = _previousAnnealingExponent;
   js["Num Finite Prior Evaluations"] = _numFinitePriorEvaluations;
   js["Num Finite Likelihood Evaluations"] = _numFiniteLikelihoodEvaluations;
   js["Accepted Samples Count"] = _acceptedSamplesCount;
   js["Current Accumulated LogEvidence"] = _currentAccumulatedLogEvidence;
   js["Proposals Acceptance Rate"] = _proposalsAcceptanceRate;
   js["Selection Acceptance Rate"] = _selectionAcceptanceRate;
   js["Covariance Matrix"] = _covarianceMatrix;
   js["Max Loglikelihood"] = _maxLoglikelihood;
   js["Mean Theta"] = _meanTheta;
   js["Sample Database"] = _sampleDatabase;
   js["Sample LogLikelihood Database"] = _sampleLogLikelihoodDatabase;
   js["Sample LogPrior Database"] = _sampleLogPriorDatabase;
   js["Sample Gradient Database"] = _sampleGradientDatabase;
   js["Sample Error Database"] = _sampleErrorDatabase;
   js["Sample Covariance Database"] = _sampleCovarianceDatabase;
   js["Upper Extended Boundaries"] = _upperExtendedBoundaries;
   js["Lower Extended Boundaries"] = _lowerExtendedBoundaries;
   js["Num LU Decomposition Failures Proposal"] = _numLUDecompositionFailuresProposal;
   js["Num Eigen Decomposition Failures Proposal"] = _numEigenDecompositionFailuresProposal;
   js["Num Inversion Failures Proposal"] = _numInversionFailuresProposal;
   js["Num Negative Definite Proposals"] = _numNegativeDefiniteProposals;
   js["Num Cholesky Decomposition Failures Proposal"] = _numCholeskyDecompositionFailuresProposal;
   js["Num Covariance Corrections"] = _numCovarianceCorrections;
 Sampler::getConfiguration(js);
} 

void TMCMC::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Multinomial Generator\": {\"Type\": \"Specific/Multinomial\"}, \"Multivariate Generator\": {\"Type\": \"Multivariate/Normal\"}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Version\": \"TMCMC\", \"Max Chain Length\": 1, \"Burn In\": 0, \"Per Generation Burn In\": [], \"Target Coefficient Of Variation\": 1.0, \"Covariance Scaling\": 0.04, \"Min Annealing Exponent Update\": 1e-05, \"Max Annealing Exponent Update\": 1.0, \"Domain Extension Factor\": 0.2, \"Step Size\": 0.1, \"Termination Criteria\": {\"Target Annealing Exponent\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Sampler::applyModuleDefaults(js);
} 

void TMCMC::applyVariableDefaults() 
{

 Sampler::applyVariableDefaults();
} 

bool TMCMC::checkTermination()
{
 bool hasFinished = false;

 if (_previousAnnealingExponent >= _targetAnnealingExponent)
 {
  _terminationCriteria.push_back("TMCMC['Target Annealing Exponent'] = " + std::to_string(_targetAnnealingExponent) + ".");
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
