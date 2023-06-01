#include "auxiliar/math.hpp"
#include "modules/conduit/conduit.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/sampler/HMC/HMC.hpp"

#include <chrono>
#include <limits>
#include <numeric>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics.h>

namespace korali
{
namespace solver
{
namespace sampler
{
;

void HMC::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  // Set version of HMC
  if (_version == "Static")
    _metricType = Metric::Static;
  else if (_version == "Euclidean")
    _metricType = Metric::Euclidean;
  else if (_version == "Riemannian")
    _metricType = Metric::Riemannian;
  else if (_version == "Riemannian Const")
    _metricType = Metric::Riemannian_Const;
  else
    KORALI_LOG_ERROR("Version must be either 'Static', 'Euclidean', 'Riemannian' or 'Riemannian Const' (is %s)\n", _version.c_str());

  // Validate configuration
  if (_burnIn < 0) KORALI_LOG_ERROR("Burn In must be larger equal 0 (is %zu).\n", _burnIn);
  if (_stepSize <= 0.) KORALI_LOG_ERROR("Step Size must be larger 0 (is %lf).\n", _stepSize);
  if (_maxDepth < 0) KORALI_LOG_ERROR("Max Depth must be non-negative (is %zu).\n", _maxDepth);
  if (_inverseRegularizationParameter <= 0.) KORALI_LOG_ERROR("Inverse Regularization Parameter must be positive (is %lf).\n", _inverseRegularizationParameter);
  if (_targetAcceptanceRate <= 0. || _targetAcceptanceRate > 1.0) KORALI_LOG_ERROR("Target Acceptance Rate must be in interval (0.0, 1.0] (is %lf).\n", _targetAcceptanceRate);
  if (_acceptanceRateLearningRate <= 0. || _acceptanceRateLearningRate > 1.) KORALI_LOG_ERROR("Acceptance Rate Learning Rate must be in interval (0.0, 1.0] (is %lf).\n", _acceptanceRateLearningRate);
  if (_targetIntegrationTime <= 0.) KORALI_LOG_ERROR("Target Integration Time must be greater 0 (is %lf).\n", _targetIntegrationTime);
  if (_numIntegrationSteps <= 0) KORALI_LOG_ERROR("Num Integration Steps must be larger equal 1 (is %zu).\n", _numIntegrationSteps);
  if (_maxIntegrationSteps < _numIntegrationSteps) KORALI_LOG_ERROR("Max Integration Steps must be larger equal 'Num Integrations Steps' (is %zu).\n", _maxIntegrationSteps);
  if (_maxFixedPointIterations <= 0) KORALI_LOG_ERROR("Max Num Fixed Point Iterations must be larger equal 1 (is %zu).\n", _maxFixedPointIterations);
  if (_adaptiveStepSizeSpeedConstant <= 0.) KORALI_LOG_ERROR("Adaptive Step Size Speed Constant must be positive (is %lf).\n", _adaptiveStepSizeSpeedConstant);
  if (_adaptiveStepSizeStabilizationConstant <= 0.) KORALI_LOG_ERROR("Adaptive Step Size Stabilization Constant must be non-negative (is %lf).\n", _adaptiveStepSizeStabilizationConstant);
  if (_adaptiveStepSizeScheduleConstant <= 0. || _adaptiveStepSizeScheduleConstant > 1.) KORALI_LOG_ERROR("Adaptive Step Size Schedule Constant must be in the interval (0.0, 1.0] (is %lf).\n", _adaptiveStepSizeScheduleConstant);
  if (_stepSizeJitter < 0. || _stepSizeJitter > 1.0) KORALI_LOG_ERROR("Step Size Jitter has to be in [0.0, 1.0] (is %lf).\n", _stepSizeJitter);

  if (_initialFastAdaptionInterval < 0) KORALI_LOG_ERROR("Initial Fast Adaption Interval must be greater equal 0 (is %zu).\n", _initialFastAdaptionInterval);
  if (_initialSlowAdaptionInterval < 0) KORALI_LOG_ERROR("Initial Slow Adaption Interval must be greater equal 0 (is %zu).\n", _initialSlowAdaptionInterval);
  if (_finalFastAdaptionInterval < 0) KORALI_LOG_ERROR("Final Fast Adaption Interval must be greater equal 0 (is %zu).\n", _finalFastAdaptionInterval);

  // Check adaption intervals
  if (_metricType == Metric::Euclidean && _useAdaptiveStepSize)
  {
    if (_burnIn < _initialFastAdaptionInterval + _initialSlowAdaptionInterval + _finalFastAdaptionInterval)
    {
      _k->_logger->logInfo("Minimal", "Burn In (%zu) smaller than adaption length: %zu\n", _initialFastAdaptionInterval + _initialSlowAdaptionInterval + _finalFastAdaptionInterval);
      _k->_logger->logInfo("Minimal", "Reducing adaption window lengths to (15%/75%/10%) of Burn In period\n");
      _initialFastAdaptionInterval = _burnIn * 0.15;
      _finalFastAdaptionInterval = _burnIn * 0.10;
      _initialSlowAdaptionInterval = _burnIn - _initialFastAdaptionInterval - _finalFastAdaptionInterval;
    }

    const size_t minBurnInLength = 100;
    if (_burnIn < minBurnInLength)
      KORALI_LOG_ERROR("Burn In too short for adaptive step size, must be larger than %zu (is %zu).", minBurnInLength, _burnIn);
  }

  // Resizing vectors of internal settings to correct dimensions
  _positionLeader.resize(_variableCount);
  _positionCandidate.resize(_variableCount);
  _momentumLeader.resize(_variableCount);
  _momentumCandidate.resize(_variableCount);

  // Prepare DBs
  _sampleDatabase.resize(0);
  _euclideanWarmupSampleDatabase.resize(0);
  _sampleEvaluationDatabase.resize(0);

  // Initializing variable defaults
  for (size_t i = 0; i < _variableCount; i++)
    _positionLeader[i] = _k->_variables[i]->_initialMean;

  // Initializing metric and inverseMetric for diagonal and dense
  if (_useDiagonalMetric == true || _metricType == Metric::Static)
  {
    _metric.resize(_variableCount);
    _inverseMetric.resize(_variableCount);
    for (size_t i = 0; i < _variableCount; ++i)
    {
      _inverseMetric[i] = _k->_variables[i]->_initialStandardDeviation * _k->_variables[i]->_initialStandardDeviation;
      _metric[i] = 1. / _inverseMetric[i];
    }
  }
  else
  {
    _metric.resize(_variableCount * _variableCount, 0.);
    _inverseMetric.resize(_variableCount * _variableCount, 0.);

    for (size_t d = 0; d < _variableCount; ++d)
    {
      _inverseMetric[d * _variableCount + d] = _k->_variables[d]->_initialStandardDeviation * _k->_variables[d]->_initialStandardDeviation;
      _metric[d * _variableCount + d] = 1. / _inverseMetric[d * _variableCount + d];
    }
  }

  // Initializing Hamiltonian
  if (_metricType == Metric::Static)
    _hamiltonian = std::make_shared<HamiltonianEuclideanDiag>(_variableCount, _normalGenerator, _k);
  if (_metricType == Metric::Euclidean && _useDiagonalMetric == true)
    _hamiltonian = std::make_shared<HamiltonianEuclideanDiag>(_variableCount, _normalGenerator, _k);
  if (_metricType == Metric::Euclidean && _useDiagonalMetric == false)
    _hamiltonian = std::make_shared<HamiltonianEuclideanDense>(_variableCount, _multivariateGenerator, _metric, _k);
  if (_metricType == Metric::Riemannian_Const && _useDiagonalMetric == true)
    _hamiltonian = std::make_shared<HamiltonianRiemannianConstDiag>(_variableCount, _normalGenerator, _inverseRegularizationParameter, _k);
  if (_metricType == Metric::Riemannian_Const && _useDiagonalMetric == false)
    _hamiltonian = std::make_shared<HamiltonianRiemannianConstDense>(_variableCount, _multivariateGenerator, _metric, _inverseRegularizationParameter, _k);
  if (_metricType == Metric::Riemannian && _useDiagonalMetric == true)
    _hamiltonian = std::make_shared<HamiltonianRiemannianDiag>(_variableCount, _normalGenerator, _inverseRegularizationParameter, _k);
  if (_metricType == Metric::Riemannian && _useDiagonalMetric == false)
    KORALI_LOG_ERROR("Riemannian Metric only supported with Diagonal Metric");

  // Initializing Integrator
  if (_metricType == Metric::Riemannian)
    _integrator = std::make_unique<LeapfrogExplicit>(_hamiltonian);
  else
    _integrator = std::make_unique<LeapfrogExplicit>(_hamiltonian);

  // Initialize common variables
  _acceptanceCount = 0;
  _proposedSampleCount = 0;
  _chainLength = 0;
  _acceptanceRate = 1.;
  _acceptanceRateError = 0.;
  _runningAcceptanceRate = 1.;
  _currentDepth = 0;

  // Initialize step size
  _logDualStepSize = std::log(_stepSize);
  if (_useAdaptiveStepSize == true)
  {
    _mu = std::log(10. * _stepSize);
    _hBar = 0.;
  }
}

void HMC::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();
  _hamiltonian->updateHamiltonian(_positionLeader, _metric, _inverseMetric);

  // Samples Momentum Candidate from N(0, metric)
  _momentumCandidate = _hamiltonian->sampleMomentum(_metric);
  _proposedSampleCount++;

  _positionCandidate = _positionLeader;
  _momentumLeader = _momentumCandidate;

  if (_metricType == Metric::Riemannian_Const)
  {
    int err = _hamiltonian->updateMetricMatricesRiemannian(_metric, _inverseMetric);
    if (err > 0) _k->_logger->logWarning("Normal", "Error during metric update.");
  }

  // New uniform sample for accept / reject
  const double logUniSample = std::log(_uniformGenerator->getRandomNumber());

  // Execute version specific generation
  if (_useNUTS)
  {
    if (_metricType == Metric::Riemannian)
    {
      runGenerationNUTSRiemannian(logUniSample);
    }
    else
    {
      runGenerationNUTS(logUniSample);
    }
  }
  else
  {
    runGenerationHMC(logUniSample);
  }

  saveSample();

  updateState();
}

void HMC::runGenerationHMC(const double logUniSample)
{
  // Track energies from leader
  const double oldK = _hamiltonian->K(_momentumLeader, _inverseMetric);
  const double oldU = _hamiltonian->U();

  for (size_t i = 0; i < _numIntegrationSteps; ++i)
    _integrator->step(_positionCandidate, _momentumCandidate, _metric, _inverseMetric, _stepSize);

  // Negate proposed momentum to make proposal symmetric
  std::transform(std::cbegin(_momentumCandidate), std::cend(_momentumCandidate), std::begin(_momentumCandidate), std::negate<double>());

  // Save energies from candidate
  const double newK = _hamiltonian->K(_momentumCandidate, _inverseMetric);
  const double newU = _hamiltonian->U();
  const double logAlpha = std::min(0., -(newK - oldK + newU - oldU));

  // Accept or reject sample
  bool isNanPositionCandidate = isanynan(_positionCandidate);
  _acceptanceProbability = isNanPositionCandidate ? 0. : std::exp(logAlpha);
  _runningAcceptanceRate = _acceptanceRateLearningRate * _runningAcceptanceRate + (1. - _acceptanceRateLearningRate) * _acceptanceProbability;
  if (logUniSample <= logAlpha && !isNanPositionCandidate)
  {
    _acceptanceCount++;
    _positionLeader = _positionCandidate;
    _leaderEvaluation = -newU;
  }
}

void HMC::runGenerationNUTS(const double logUniSample)
{
  const double oldK = _hamiltonian->K(_momentumLeader, _inverseMetric);
  const double oldU = _hamiltonian->U();

  auto qLeft = _positionLeader;
  auto pLeft = _momentumLeader;
  auto qRight = _positionLeader;
  auto pRight = _momentumLeader;

  _currentDepth = 0;
  size_t numLeavesSubtree;
  bool buildCriterion = true;
  bool buildCriterionSubtree;
  double numValidLeaves = 1.;
  double numValidLeavesSubtree;
  const double oldH = oldU + oldK;

  auto helper = std::make_shared<TreeHelperEuclidean>();
  helper->logUniSampleIn = logUniSample;
  helper->rootHIn = oldH;
  helper->buildCriterionOut = true;

  do
  {
    if (_uniformGenerator->getRandomNumber() < 0.5)
    {
      helper->qIn = qLeft;
      helper->pIn = pLeft;
      helper->directionIn = -1;

      buildTree(helper, _currentDepth);

      // qRight = --
      // pRight = --
      qLeft = helper->qLeftOut;
      pLeft = helper->pLeftOut;

      // Setting for termination criterium
      helper->qRightOut = qRight;
      helper->pRightOut = pRight;
    }
    else
    {
      helper->qIn = qRight;
      helper->pIn = pRight;
      helper->directionIn = +1;

      buildTree(helper, _currentDepth);

      // qLeft = --
      // pLeft = --
      qRight = helper->qRightOut;
      pRight = helper->pRightOut;

      // setting for termination criterium
      helper->qLeftOut = qLeft;
      helper->pLeftOut = pLeft;
    }
    _positionCandidate = helper->qProposedOut;
    numValidLeavesSubtree = helper->numValidLeavesOut;
    buildCriterionSubtree = helper->buildCriterionOut;
    _acceptanceProbability = helper->alphaOut;
    numLeavesSubtree = helper->numLeavesOut;

    if (buildCriterionSubtree == true)
      if (_uniformGenerator->getRandomNumber() < numValidLeavesSubtree / numValidLeaves && isanynan(_positionCandidate) == false)
      {
        _positionLeader = _positionCandidate;
        _leaderEvaluation = -_hamiltonian->U();
      }

    numValidLeaves += numValidLeavesSubtree;

    buildCriterion = buildCriterionSubtree && helper->computeCriterion(*_hamiltonian);

    _currentDepth++;
  } while (buildCriterion == true && _currentDepth <= _maxDepth);

  _currentDepth--;
  _acceptanceProbability /= (double)numLeavesSubtree;
  _acceptanceCountNUTS += _acceptanceProbability;
  _runningAcceptanceRate = _acceptanceRateLearningRate * _runningAcceptanceRate + (1. - _acceptanceRateLearningRate) * _acceptanceProbability;
}

void HMC::runGenerationNUTSRiemannian(const double logUniSample)
{
  const double oldK = _hamiltonian->K(_momentumLeader, _inverseMetric);
  const double oldU = _hamiltonian->U();

  std::vector<double> qLeft = _positionLeader;
  std::vector<double> pLeft = _momentumLeader;
  std::vector<double> qRight = _positionLeader;
  std::vector<double> pRight = _momentumLeader;

  size_t depth = 0;
  size_t numLeavesSubtree;
  bool buildCriterion = true;
  bool buildCriterionSubtree;
  double numValidLeaves = 1.;
  double numValidLeavesSubtree;
  const double oldH = oldU + oldK;

  auto helper = std::make_shared<TreeHelperRiemannian>();
  helper->logUniSampleIn = logUniSample;
  helper->rootHIn = oldH;
  helper->buildCriterionOut = true;

  std::vector<double> rhoInit = _momentumLeader;
  std::vector<double> rhoLeft(_variableCount, 0.);
  std::vector<double> rhoRight(_variableCount, 0.);
  while (buildCriterion == true && depth <= _maxDepth)
  {
    if (_uniformGenerator->getRandomNumber() < 0.5)
    {
      helper->qIn = qLeft;
      helper->pIn = pLeft;
      helper->directionIn = -1;

      buildTreeIntegration(helper, rhoLeft, depth);

      // qRight = --
      // pRight = --
      qLeft = helper->qLeftOut;
      pLeft = helper->pLeftOut;

      // Setting for termination criterium
      helper->qRightOut = qRight;
      helper->pRightOut = pRight;
    }
    else
    {
      helper->qIn = qRight;
      helper->pIn = pRight;
      helper->directionIn = +1;

      buildTreeIntegration(helper, rhoRight, depth);

      // qLeft = --
      // pLeft = --
      qRight = helper->qRightOut;
      pRight = helper->pRightOut;

      // Setting for termination criterium
      helper->qLeftOut = qLeft;
      helper->pLeftOut = pLeft;
    }

    _positionCandidate = helper->qProposedOut;
    _acceptanceProbability = helper->alphaOut;
    numValidLeavesSubtree = helper->numValidLeavesOut;
    buildCriterionSubtree = helper->buildCriterionOut;
    numLeavesSubtree = helper->numLeavesOut;

    if (buildCriterionSubtree == true)
      if (_uniformGenerator->getRandomNumber() < numValidLeavesSubtree / numValidLeaves && isanynan(_positionCandidate) == false)
      {
        _positionLeader = _positionCandidate;
        _leaderEvaluation = -_hamiltonian->U();
      }

    numValidLeaves += numValidLeavesSubtree;

    std::vector<double> deltaRho(_variableCount, 0.);

    std::transform(std::cbegin(rhoLeft), std::cend(rhoLeft), std::cbegin(deltaRho), std::begin(deltaRho), std::plus<double>());
    std::transform(cbegin(rhoInit), std::cend(rhoInit), std::cbegin(deltaRho), std::begin(deltaRho), std::plus<double>());
    std::transform(std::cbegin(rhoRight), std::cend(rhoRight), std::cbegin(deltaRho), std::begin(deltaRho), std::plus<double>());

    buildCriterion = buildCriterionSubtree && helper->computeCriterion(*_hamiltonian, pLeft, pRight, _inverseMetric, deltaRho);

    depth++;
  }

  _acceptanceProbability /= (double)numLeavesSubtree;
  _acceptanceCountNUTS += _acceptanceProbability;
  _runningAcceptanceRate = _acceptanceRateLearningRate * _runningAcceptanceRate + (1. - _acceptanceRateLearningRate) * _acceptanceProbability;
}

void HMC::saveSample()
{
  // Store samples after burn in period
  if (_burnIn <= _chainLength)
  {
    _sampleDatabase.push_back(_positionLeader);
    _sampleEvaluationDatabase.push_back(_leaderEvaluation);
  }
  // Keep samples during slow adaption interval for metric calculation
  else if (_metricType == Metric::Euclidean && (_initialFastAdaptionInterval <= _chainLength) && (_chainLength < _initialFastAdaptionInterval + _initialSlowAdaptionInterval))
  {
    _euclideanWarmupSampleDatabase.push_back(_positionLeader);
  }
}

void HMC::updateStepSize()
{
  // Adapt step size during burn in period
  if (_chainLength <= _burnIn)
  {
    const size_t terminalFastPeriodStart = _initialFastAdaptionInterval + _initialSlowAdaptionInterval;
    const size_t terminalFastPeriodEnd = terminalFastPeriodStart + _finalFastAdaptionInterval;

    bool isInitalFastPeriod = _chainLength < _initialFastAdaptionInterval;
    bool isTerminalFastPeriod = (_chainLength >= terminalFastPeriodStart) && (_chainLength < terminalFastPeriodEnd);

    if (_useAdaptiveStepSize == true)
    {
      // Reset statistics before starting terminal fast period
      if (_chainLength == terminalFastPeriodStart)
      {
        _acceptanceRateError = 0.;
        _hBar = 0.;
        _mu = std::log(10. * _stepSize);
      }

      // Update step size and dual step size during fast adaption periods
      if (isInitalFastPeriod || isTerminalFastPeriod)
      {
        const double counter = isInitalFastPeriod ? _chainLength + 1 : _chainLength + 1 - terminalFastPeriodStart;
        _acceptanceProbability = (_acceptanceProbability > 1.) ? 1. : _acceptanceProbability;
        _runningAcceptanceRate = (_runningAcceptanceRate > 1.) ? 1. : _runningAcceptanceRate;
        _acceptanceRateError = _targetAcceptanceRate - _runningAcceptanceRate;
        const double eta = 1. / (counter + 10.);
        _hBar = (1. - eta) * _hBar + eta * _acceptanceRateError;

        const double logStep = _mu - _hBar * std::sqrt(counter) / _adaptiveStepSizeSpeedConstant;
        const double dualEta = std::pow(counter, -_adaptiveStepSizeScheduleConstant);
        _logDualStepSize = (1. - dualEta) * _logDualStepSize + dualEta * logStep;
        _stepSize = std::exp(logStep);
      }
      // Fix step size after terminal fast epriod
      else if (_chainLength == terminalFastPeriodEnd)
      {
        _stepSize = std::exp(_logDualStepSize);
      }
    }
  }

  // Apply Step Size Jitter after burn in period
  else if (_stepSizeJitter > 0.)
    _stepSize = std::exp(_logDualStepSize) * (1. + _stepSizeJitter * (2. * _uniformGenerator->getRandomNumber() - 1.));

  if (_useNUTS == false && _useAdaptiveStepSize == true)
  {
    _numIntegrationSteps = std::max((size_t)1, (size_t)std::round(_targetIntegrationTime / _stepSize));
    if (_numIntegrationSteps > _maxIntegrationSteps)
    {
      _k->_logger->logWarning("Minimal", "Num Integration Steps (%zu) exceeding hard limit 'Max Integration Steps' (%zu).\n", _numIntegrationSteps, _maxIntegrationSteps);
      _numIntegrationSteps = _maxIntegrationSteps;
    }
  }
}

void HMC::updateState()
{
  _modelEvaluationCount = _hamiltonian->_modelEvaluationCount;

  // Update Acceptance Rate
  if (_useNUTS)
    _acceptanceRate = (double)_acceptanceCountNUTS / ((double)_chainLength + 1);
  else
    _acceptanceRate = (double)_acceptanceCount / ((double)(_chainLength + 1));

  // Update Step Size, Dual Step Size, H Bar and apply step size jitter
  updateStepSize();

  // Update metric at the end of slow adaption period
  if (_metricType == Metric::Euclidean && _chainLength == _initialFastAdaptionInterval + _initialSlowAdaptionInterval)
  {
    int err = _hamiltonian->updateMetricMatricesEuclidean(_euclideanWarmupSampleDatabase, _metric, _inverseMetric);
    if (err == GSL_EDOM) KORALI_LOG_ERROR("Inverse Metric negative definite (not updating Metric). Increase 'Initial Slow Adaption Interval' (is %zu).", _initialSlowAdaptionInterval);
    _euclideanWarmupSampleDatabase.clear();
  }

  _chainLength++;
}

void HMC::buildTree(std::shared_ptr<TreeHelperEuclidean> helper, const size_t depth)
{
  helper->qLeftOut = helper->qIn;
  helper->pLeftOut = helper->pIn;

  if (depth == 0)
  {
    const double deltaMax = 100;
    _integrator->step(helper->qLeftOut, helper->pLeftOut, _metric, _inverseMetric, helper->directionIn * _stepSize);

    const double leafK = _hamiltonian->K(helper->pLeftOut, _metric);
    const double leafU = _hamiltonian->U();
    const double leafH = leafU + leafK;

    helper->numValidLeavesOut = (helper->logUniSampleIn <= helper->rootHIn - leafH) ? 1. : 0.;

    helper->buildCriterionOut = (helper->rootHIn - leafH + deltaMax > helper->logUniSampleIn);

    helper->alphaOut = isanynan(helper->qLeftOut) ? 0.0 : std::min(1., std::exp(helper->rootHIn - leafH));
    helper->numLeavesOut = 1;
    helper->qRightOut = helper->qLeftOut;
    helper->pRightOut = helper->pLeftOut;
    helper->qProposedOut = helper->qLeftOut;
  }
  else
  {
    size_t numLeavesSubtree;
    bool buildCriterionSubtree;
    double numValidLeavesSubtree, alphaSubtree;

    buildTree(helper, depth - 1);

    if (helper->buildCriterionOut == true)
    {
      auto helperSubtree = std::make_shared<TreeHelperEuclidean>();
      helperSubtree->logUniSampleIn = helper->logUniSampleIn;
      helperSubtree->directionIn = helper->directionIn;
      helperSubtree->rootHIn = helper->rootHIn;

      if (helper->directionIn == -1)
      {
        helperSubtree->qIn = helper->qLeftOut;
        helperSubtree->pIn = helper->pLeftOut;

        buildTree(helperSubtree, depth - 1);

        helper->qLeftOut = helperSubtree->qLeftOut;
        helper->pLeftOut = helperSubtree->pLeftOut;
        // helperSubtree->qRightOut = --
        // helperSubtree->pRightOut = --
      }
      else
      {
        helperSubtree->qIn = helper->qRightOut;
        helperSubtree->pIn = helper->pRightOut;

        buildTree(helperSubtree, depth - 1);

        // helperSubtree->qLeftOut = --
        // helperSubtree->pLeftOut = --
        helper->qRightOut = helperSubtree->qRightOut;
        helper->pRightOut = helperSubtree->pRightOut;
      }
      numValidLeavesSubtree = helperSubtree->numValidLeavesOut;
      buildCriterionSubtree = helperSubtree->buildCriterionOut;
      alphaSubtree = helperSubtree->alphaOut;
      numLeavesSubtree = helperSubtree->numLeavesOut;

      // First check to avoid divising by zero
      if (numValidLeavesSubtree != 0 && _uniformGenerator->getRandomNumber() < numValidLeavesSubtree / (helper->numValidLeavesOut + numValidLeavesSubtree))
      {
        helper->qProposedOut = helperSubtree->qProposedOut;
      }

      helper->numValidLeavesOut += numValidLeavesSubtree;
      helper->alphaOut += alphaSubtree;
      helper->numLeavesOut += numLeavesSubtree;

      helper->buildCriterionOut = buildCriterionSubtree && helper->computeCriterion(*_hamiltonian);
    }
  }
}

void HMC::buildTreeIntegration(std::shared_ptr<TreeHelperRiemannian> helper, std::vector<double> &rho, const size_t depth)
{
  helper->qLeftOut = helper->qIn;
  helper->pLeftOut = helper->pIn;

  if (depth == 0)
  {
    const double deltaMax = 100;
    _integrator->step(helper->qLeftOut, helper->pLeftOut, _metric, _inverseMetric, helper->directionIn * _stepSize);

    const double leafK = _hamiltonian->K(helper->pLeftOut, _metric);
    const double leafU = _hamiltonian->U();
    const double leafH = leafU + leafK;

    helper->numValidLeavesOut = (helper->logUniSampleIn <= helper->rootHIn - leafH) ? 1. : 0.;

    helper->buildCriterionOut = (helper->rootHIn - leafH + deltaMax > helper->logUniSampleIn);

    helper->alphaOut = isanynan(helper->qLeftOut) ? 0.0 : std::min(1., std::exp(helper->rootHIn - leafH));
    helper->numLeavesOut = 1;
    helper->qRightOut = helper->qLeftOut;
    helper->pRightOut = helper->pLeftOut;
    helper->qProposedOut = helper->qLeftOut;

    std::transform(std::cbegin(rho), std::cend(rho), std::cbegin(helper->pRightOut), std::begin(rho), std::plus<double>());
  }
  else
  {
    size_t numLeavesSubtree;
    bool buildCriterionSubtree;
    double numValidLeavesSubtree, alphaSubtree;
    auto pStart = helper->pIn;
    std::vector<double> rhoFirstSubtree(_variableCount, 0.);

    buildTreeIntegration(helper, rhoFirstSubtree, depth - 1);

    if (helper->buildCriterionOut == true)
    {
      auto helperSubtree = std::make_shared<TreeHelperRiemannian>();
      helperSubtree->logUniSampleIn = helper->logUniSampleIn;
      helperSubtree->directionIn = helper->directionIn;
      helperSubtree->rootHIn = helper->rootHIn;

      std::vector<double> rhoSecondSubtree(_variableCount, 0.);
      if (helper->directionIn == -1)
      {
        helperSubtree->qIn = helper->qLeftOut;
        helperSubtree->pIn = helper->pLeftOut;
        buildTreeIntegration(helperSubtree, rhoSecondSubtree, depth - 1);

        helper->qLeftOut = helperSubtree->qLeftOut;
        helper->pLeftOut = helperSubtree->pLeftOut;
        // helperSubtree->qRightOut = --
        // helperSubtree->pRightOut = --
      }
      else
      {
        helperSubtree->qIn = helper->qRightOut;
        helperSubtree->pIn = helper->pRightOut;
        buildTreeIntegration(helperSubtree, rhoSecondSubtree, depth - 1);

        // helperSubtree->qLeftOut = --
        // helperSubtree->pLeftOut = --
        helper->qRightOut = helperSubtree->qRightOut;
        helper->pRightOut = helperSubtree->pRightOut;
      }

      numValidLeavesSubtree = helperSubtree->numValidLeavesOut;
      buildCriterionSubtree = helperSubtree->buildCriterionOut;
      alphaSubtree = helperSubtree->alphaOut;
      numLeavesSubtree = helperSubtree->numLeavesOut;

      std::vector<double> rhoSubtree = rhoFirstSubtree;
      std::transform(std::cbegin(rhoSubtree), std::cend(rhoSubtree), std::cbegin(rhoSecondSubtree), std::begin(rhoSubtree), std::plus<double>());

      std::transform(std::cbegin(rho), std::cend(rho), std::cbegin(rhoSubtree), std::begin(rho), std::plus<double>());

      // First check to avoid divising by zero
      if (numValidLeavesSubtree != 0 && _uniformGenerator->getRandomNumber() < numValidLeavesSubtree / (helper->numValidLeavesOut + numValidLeavesSubtree))
      {
        helper->qProposedOut = helperSubtree->qProposedOut;
      }

      helper->numValidLeavesOut += numValidLeavesSubtree;
      helper->alphaOut += alphaSubtree;
      helper->numLeavesOut += numLeavesSubtree;

      std::vector<double> pEnd = helper->directionIn == -1 ? helperSubtree->pLeftOut : helperSubtree->pRightOut;
      helper->buildCriterionOut = buildCriterionSubtree && helper->computeCriterion(*_hamiltonian, pStart, pEnd, _inverseMetric, rhoSubtree);
    }
  }
}

void HMC::printGenerationBefore()
{
  return;
}

void HMC::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Database Entries %ld\n", _sampleDatabase.size());
  _k->_logger->logInfo("Normal", "Acceptance Rate Proposals: %.2f%%\n", 100. * _acceptanceRate);
  _k->_logger->logInfo("Normal", "Running Acceptance Rate: %.2f%%\n", 100. * _runningAcceptanceRate);
  _k->_logger->logInfo("Detailed", "Num Model Evaluations: %zu\n", _hamiltonian->_modelEvaluationCount);

  _k->_logger->logInfo("Detailed", "Current Leader:\n");
  for (size_t d = 0; d < _variableCount; ++d) _k->_logger->logData("Detailed", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _positionLeader[d]);
  _k->_logger->logInfo("Detailed", "Current Candidate:\n");
  for (size_t d = 0; d < _variableCount; ++d) _k->_logger->logData("Detailed", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _positionCandidate[d]);

  _k->_logger->logInfo("Detailed", "Gradient:\n");
  auto grad = _hamiltonian->dU();
  for (auto &g : grad) _k->_logger->logData("Detailed", "         %+6.3e  \n", g);

  _k->_logger->logInfo("Detailed", "Current Metric:\n");
  if (_useDiagonalMetric == true || _metricType == Metric::Static)
  {
    for (size_t e = 0; e < _variableCount; ++e)
      _k->_logger->logData("Detailed", "         %+6.3e  \n", _metric[e]);
  }
  else
  {
    for (size_t d = 0; d < _variableCount; ++d)
    {
      for (size_t e = 0; e < _variableCount; ++e)
        _k->_logger->logData("Detailed", "         %+6.3e  ", _metric[d * _variableCount + e]);
      _k->_logger->logInfo("Detailed", "\n");
    }
  }

  _k->_logger->logInfo("Detailed", "Current Inverse Metric:\n");

  if (_useDiagonalMetric == true || _metricType == Metric::Static)
  {
    for (size_t e = 0; e < _variableCount; ++e)
      _k->_logger->logData("Detailed", "         %+6.3e  \n", _inverseMetric[e]);
  }
  else
  {
    for (size_t d = 0; d < _variableCount; ++d)
    {
      for (size_t e = 0; e < _variableCount; ++e)
        _k->_logger->logData("Detailed", "         %+6.3e  ", _inverseMetric[d * _variableCount + e]);
      _k->_logger->logInfo("Detailed", "\n");
    }
  }

  // Chain Length + 1 = m in Algorithm
  _k->_logger->logInfo("Detailed", "Chain Length: %ld\n", _chainLength);

  if (_useAdaptiveStepSize == true)
  {
    _k->_logger->logInfo("Detailed", "Step Size: %lf\n", _stepSize);
    _k->_logger->logInfo("Detailed", "Dual Step Size: %lf\n", std::exp(_logDualStepSize));
  }

  if (_useNUTS == false)
  {
    _k->_logger->logInfo("Detailed", "Num Integration Steps: %ld\n", _numIntegrationSteps);
    _k->_logger->logInfo("Normal", "Accepted Samples: %zu\n", _acceptanceCount);
  }
  else
  {
    _k->_logger->logInfo("Detailed", "Accepted Samples Indicator (NUTS): %lf\n", _acceptanceCountNUTS);
    _k->_logger->logInfo("Detailed", "Current Depth (NUTS): %zu\n", _currentDepth);
  }
}

void HMC::finalize()
{
  _k->_logger->logInfo("Normal", "Number of Generated Samples: %zu\n", _proposedSampleCount);
  _k->_logger->logInfo("Normal", "Acceptance Rate: %.2f%%\n", 100 * _acceptanceRate);
  _k->_logger->logInfo("Normal", "Num Model Evaluations: %zu\n", _modelEvaluationCount);
  (*_k)["Results"]["Sample Database"] = _sampleDatabase;
}

void HMC::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Metric Type"))
 {
 try { _metricType = js["Metric Type"].get<Metric>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Metric Type']\n%s", e.what()); } 
   eraseValue(js, "Metric Type");
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

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Acceptance Rate"))
 {
 try { _acceptanceRate = js["Acceptance Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Acceptance Rate']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Rate");
 }

 if (isDefined(js, "Running Acceptance Rate"))
 {
 try { _runningAcceptanceRate = js["Running Acceptance Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Running Acceptance Rate']\n%s", e.what()); } 
   eraseValue(js, "Running Acceptance Rate");
 }

 if (isDefined(js, "Acceptance Count"))
 {
 try { _acceptanceCount = js["Acceptance Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Acceptance Count']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Count");
 }

 if (isDefined(js, "Proposed Sample Count"))
 {
 try { _proposedSampleCount = js["Proposed Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Proposed Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Proposed Sample Count");
 }

 if (isDefined(js, "Sample Database"))
 {
 try { _sampleDatabase = js["Sample Database"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Sample Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Database");
 }

 if (isDefined(js, "Euclidean Warmup Sample Database"))
 {
 try { _euclideanWarmupSampleDatabase = js["Euclidean Warmup Sample Database"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Euclidean Warmup Sample Database']\n%s", e.what()); } 
   eraseValue(js, "Euclidean Warmup Sample Database");
 }

 if (isDefined(js, "Sample Evaluation Database"))
 {
 try { _sampleEvaluationDatabase = js["Sample Evaluation Database"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Sample Evaluation Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Evaluation Database");
 }

 if (isDefined(js, "Chain Length"))
 {
 try { _chainLength = js["Chain Length"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Chain Length']\n%s", e.what()); } 
   eraseValue(js, "Chain Length");
 }

 if (isDefined(js, "Leader Evaluation"))
 {
 try { _leaderEvaluation = js["Leader Evaluation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Leader Evaluation']\n%s", e.what()); } 
   eraseValue(js, "Leader Evaluation");
 }

 if (isDefined(js, "Candidate Evaluation"))
 {
 try { _candidateEvaluation = js["Candidate Evaluation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Candidate Evaluation']\n%s", e.what()); } 
   eraseValue(js, "Candidate Evaluation");
 }

 if (isDefined(js, "Position Leader"))
 {
 try { _positionLeader = js["Position Leader"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Position Leader']\n%s", e.what()); } 
   eraseValue(js, "Position Leader");
 }

 if (isDefined(js, "Position Candidate"))
 {
 try { _positionCandidate = js["Position Candidate"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Position Candidate']\n%s", e.what()); } 
   eraseValue(js, "Position Candidate");
 }

 if (isDefined(js, "Momentum Leader"))
 {
 try { _momentumLeader = js["Momentum Leader"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Momentum Leader']\n%s", e.what()); } 
   eraseValue(js, "Momentum Leader");
 }

 if (isDefined(js, "Momentum Candidate"))
 {
 try { _momentumCandidate = js["Momentum Candidate"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Momentum Candidate']\n%s", e.what()); } 
   eraseValue(js, "Momentum Candidate");
 }

 if (isDefined(js, "Log Dual Step Size"))
 {
 try { _logDualStepSize = js["Log Dual Step Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Log Dual Step Size']\n%s", e.what()); } 
   eraseValue(js, "Log Dual Step Size");
 }

 if (isDefined(js, "Mu"))
 {
 try { _mu = js["Mu"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Mu']\n%s", e.what()); } 
   eraseValue(js, "Mu");
 }

 if (isDefined(js, "H Bar"))
 {
 try { _hBar = js["H Bar"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['H Bar']\n%s", e.what()); } 
   eraseValue(js, "H Bar");
 }

 if (isDefined(js, "Acceptance Count NUTS"))
 {
 try { _acceptanceCountNUTS = js["Acceptance Count NUTS"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Acceptance Count NUTS']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Count NUTS");
 }

 if (isDefined(js, "Current Depth"))
 {
 try { _currentDepth = js["Current Depth"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Current Depth']\n%s", e.what()); } 
   eraseValue(js, "Current Depth");
 }

 if (isDefined(js, "Acceptance Probability"))
 {
 try { _acceptanceProbability = js["Acceptance Probability"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Acceptance Probability']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Probability");
 }

 if (isDefined(js, "Acceptance Rate Error"))
 {
 try { _acceptanceRateError = js["Acceptance Rate Error"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Acceptance Rate Error']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Rate Error");
 }

 if (isDefined(js, "Metric"))
 {
 try { _metric = js["Metric"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Metric']\n%s", e.what()); } 
   eraseValue(js, "Metric");
 }

 if (isDefined(js, "Inverse Metric"))
 {
 try { _inverseMetric = js["Inverse Metric"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Inverse Metric']\n%s", e.what()); } 
   eraseValue(js, "Inverse Metric");
 }

 if (isDefined(js, "Burn In"))
 {
 try { _burnIn = js["Burn In"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Burn In']\n%s", e.what()); } 
   eraseValue(js, "Burn In");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Burn In'] required by HMC.\n"); 

 if (isDefined(js, "Use Diagonal Metric"))
 {
 try { _useDiagonalMetric = js["Use Diagonal Metric"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Use Diagonal Metric']\n%s", e.what()); } 
   eraseValue(js, "Use Diagonal Metric");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Use Diagonal Metric'] required by HMC.\n"); 

 if (isDefined(js, "Num Integration Steps"))
 {
 try { _numIntegrationSteps = js["Num Integration Steps"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Num Integration Steps']\n%s", e.what()); } 
   eraseValue(js, "Num Integration Steps");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Num Integration Steps'] required by HMC.\n"); 

 if (isDefined(js, "Max Integration Steps"))
 {
 try { _maxIntegrationSteps = js["Max Integration Steps"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Max Integration Steps']\n%s", e.what()); } 
   eraseValue(js, "Max Integration Steps");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Max Integration Steps'] required by HMC.\n"); 

 if (isDefined(js, "Use NUTS"))
 {
 try { _useNUTS = js["Use NUTS"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Use NUTS']\n%s", e.what()); } 
   eraseValue(js, "Use NUTS");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Use NUTS'] required by HMC.\n"); 

 if (isDefined(js, "Step Size"))
 {
 try { _stepSize = js["Step Size"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Step Size']\n%s", e.what()); } 
   eraseValue(js, "Step Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Step Size'] required by HMC.\n"); 

 if (isDefined(js, "Use Adaptive Step Size"))
 {
 try { _useAdaptiveStepSize = js["Use Adaptive Step Size"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Use Adaptive Step Size']\n%s", e.what()); } 
   eraseValue(js, "Use Adaptive Step Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Use Adaptive Step Size'] required by HMC.\n"); 

 if (isDefined(js, "Target Acceptance Rate"))
 {
 try { _targetAcceptanceRate = js["Target Acceptance Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Target Acceptance Rate']\n%s", e.what()); } 
   eraseValue(js, "Target Acceptance Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Target Acceptance Rate'] required by HMC.\n"); 

 if (isDefined(js, "Acceptance Rate Learning Rate"))
 {
 try { _acceptanceRateLearningRate = js["Acceptance Rate Learning Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Acceptance Rate Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Rate Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Acceptance Rate Learning Rate'] required by HMC.\n"); 

 if (isDefined(js, "Target Integration Time"))
 {
 try { _targetIntegrationTime = js["Target Integration Time"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Target Integration Time']\n%s", e.what()); } 
   eraseValue(js, "Target Integration Time");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Target Integration Time'] required by HMC.\n"); 

 if (isDefined(js, "Adaptive Step Size Speed Constant"))
 {
 try { _adaptiveStepSizeSpeedConstant = js["Adaptive Step Size Speed Constant"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Adaptive Step Size Speed Constant']\n%s", e.what()); } 
   eraseValue(js, "Adaptive Step Size Speed Constant");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Adaptive Step Size Speed Constant'] required by HMC.\n"); 

 if (isDefined(js, "Adaptive Step Size Stabilization Constant"))
 {
 try { _adaptiveStepSizeStabilizationConstant = js["Adaptive Step Size Stabilization Constant"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Adaptive Step Size Stabilization Constant']\n%s", e.what()); } 
   eraseValue(js, "Adaptive Step Size Stabilization Constant");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Adaptive Step Size Stabilization Constant'] required by HMC.\n"); 

 if (isDefined(js, "Adaptive Step Size Schedule Constant"))
 {
 try { _adaptiveStepSizeScheduleConstant = js["Adaptive Step Size Schedule Constant"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Adaptive Step Size Schedule Constant']\n%s", e.what()); } 
   eraseValue(js, "Adaptive Step Size Schedule Constant");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Adaptive Step Size Schedule Constant'] required by HMC.\n"); 

 if (isDefined(js, "Max Depth"))
 {
 try { _maxDepth = js["Max Depth"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Max Depth']\n%s", e.what()); } 
   eraseValue(js, "Max Depth");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Max Depth'] required by HMC.\n"); 

 if (isDefined(js, "Version"))
 {
 try { _version = js["Version"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Version']\n%s", e.what()); } 
   eraseValue(js, "Version");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Version'] required by HMC.\n"); 

 if (isDefined(js, "Inverse Regularization Parameter"))
 {
 try { _inverseRegularizationParameter = js["Inverse Regularization Parameter"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Inverse Regularization Parameter']\n%s", e.what()); } 
   eraseValue(js, "Inverse Regularization Parameter");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Inverse Regularization Parameter'] required by HMC.\n"); 

 if (isDefined(js, "Max Fixed Point Iterations"))
 {
 try { _maxFixedPointIterations = js["Max Fixed Point Iterations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Max Fixed Point Iterations']\n%s", e.what()); } 
   eraseValue(js, "Max Fixed Point Iterations");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Max Fixed Point Iterations'] required by HMC.\n"); 

 if (isDefined(js, "Step Size Jitter"))
 {
 try { _stepSizeJitter = js["Step Size Jitter"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Step Size Jitter']\n%s", e.what()); } 
   eraseValue(js, "Step Size Jitter");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Step Size Jitter'] required by HMC.\n"); 

 if (isDefined(js, "Initial Fast Adaption Interval"))
 {
 try { _initialFastAdaptionInterval = js["Initial Fast Adaption Interval"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Initial Fast Adaption Interval']\n%s", e.what()); } 
   eraseValue(js, "Initial Fast Adaption Interval");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Fast Adaption Interval'] required by HMC.\n"); 

 if (isDefined(js, "Final Fast Adaption Interval"))
 {
 try { _finalFastAdaptionInterval = js["Final Fast Adaption Interval"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Final Fast Adaption Interval']\n%s", e.what()); } 
   eraseValue(js, "Final Fast Adaption Interval");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Final Fast Adaption Interval'] required by HMC.\n"); 

 if (isDefined(js, "Initial Slow Adaption Interval"))
 {
 try { _initialSlowAdaptionInterval = js["Initial Slow Adaption Interval"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Initial Slow Adaption Interval']\n%s", e.what()); } 
   eraseValue(js, "Initial Slow Adaption Interval");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Slow Adaption Interval'] required by HMC.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Samples"))
 {
 try { _maxSamples = js["Termination Criteria"]["Max Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Termination Criteria']['Max Samples']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Samples");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Samples'] required by HMC.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Mean"))
 {
 try { _k->_variables[i]->_initialMean = _k->_js["Variables"][i]["Initial Mean"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Initial Mean']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Mean");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Mean'] required by HMC.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Initial Standard Deviation"))
 {
 try { _k->_variables[i]->_initialStandardDeviation = _k->_js["Variables"][i]["Initial Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ HMC ] \n + Key:    ['Initial Standard Deviation']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Standard Deviation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Standard Deviation'] required by HMC.\n"); 

 } 
 Sampler::setConfiguration(js);
 _type = "sampler/HMC";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: HMC: \n%s\n", js.dump(2).c_str());
} 

void HMC::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Burn In"] = _burnIn;
   js["Use Diagonal Metric"] = _useDiagonalMetric;
   js["Num Integration Steps"] = _numIntegrationSteps;
   js["Max Integration Steps"] = _maxIntegrationSteps;
   js["Use NUTS"] = _useNUTS;
   js["Step Size"] = _stepSize;
   js["Use Adaptive Step Size"] = _useAdaptiveStepSize;
   js["Target Acceptance Rate"] = _targetAcceptanceRate;
   js["Acceptance Rate Learning Rate"] = _acceptanceRateLearningRate;
   js["Target Integration Time"] = _targetIntegrationTime;
   js["Adaptive Step Size Speed Constant"] = _adaptiveStepSizeSpeedConstant;
   js["Adaptive Step Size Stabilization Constant"] = _adaptiveStepSizeStabilizationConstant;
   js["Adaptive Step Size Schedule Constant"] = _adaptiveStepSizeScheduleConstant;
   js["Max Depth"] = _maxDepth;
   js["Version"] = _version;
   js["Inverse Regularization Parameter"] = _inverseRegularizationParameter;
   js["Max Fixed Point Iterations"] = _maxFixedPointIterations;
   js["Step Size Jitter"] = _stepSizeJitter;
   js["Initial Fast Adaption Interval"] = _initialFastAdaptionInterval;
   js["Final Fast Adaption Interval"] = _finalFastAdaptionInterval;
   js["Initial Slow Adaption Interval"] = _initialSlowAdaptionInterval;
   js["Termination Criteria"]["Max Samples"] = _maxSamples;
   js["Metric Type"] = _metricType;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_multivariateGenerator != NULL) _multivariateGenerator->getConfiguration(js["Multivariate Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Acceptance Rate"] = _acceptanceRate;
   js["Running Acceptance Rate"] = _runningAcceptanceRate;
   js["Acceptance Count"] = _acceptanceCount;
   js["Proposed Sample Count"] = _proposedSampleCount;
   js["Sample Database"] = _sampleDatabase;
   js["Euclidean Warmup Sample Database"] = _euclideanWarmupSampleDatabase;
   js["Sample Evaluation Database"] = _sampleEvaluationDatabase;
   js["Chain Length"] = _chainLength;
   js["Leader Evaluation"] = _leaderEvaluation;
   js["Candidate Evaluation"] = _candidateEvaluation;
   js["Position Leader"] = _positionLeader;
   js["Position Candidate"] = _positionCandidate;
   js["Momentum Leader"] = _momentumLeader;
   js["Momentum Candidate"] = _momentumCandidate;
   js["Log Dual Step Size"] = _logDualStepSize;
   js["Mu"] = _mu;
   js["H Bar"] = _hBar;
   js["Acceptance Count NUTS"] = _acceptanceCountNUTS;
   js["Current Depth"] = _currentDepth;
   js["Acceptance Probability"] = _acceptanceProbability;
   js["Acceptance Rate Error"] = _acceptanceRateError;
   js["Metric"] = _metric;
   js["Inverse Metric"] = _inverseMetric;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Mean"] = _k->_variables[i]->_initialMean;
   _k->_js["Variables"][i]["Initial Standard Deviation"] = _k->_variables[i]->_initialStandardDeviation;
 } 
 Sampler::getConfiguration(js);
} 

void HMC::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Burn In\": 300, \"Use Diagonal Metric\": true, \"Step Size\": 0.1, \"Num Integration Steps\": 4, \"Max Integration Steps\": 100, \"Use Adaptive Step Size\": true, \"Target Acceptance Rate\": 0.65, \"Acceptance Rate Learning Rate\": 0.85, \"Target Integration Time\": 1.0, \"Use NUTS\": true, \"Acceptance Count NUTS\": 0.0, \"Adaptive Step Size Speed Constant\": 0.05, \"Adaptive Step Size Stabilization Constant\": 10.0, \"Adaptive Step Size Schedule Constant\": 0.75, \"Max Depth\": 5, \"Version\": \"Euclidean\", \"Inverse Regularization Parameter\": 1.0, \"Max Fixed Point Iterations\": 8, \"Step Size Jitter\": 0.0, \"Initial Fast Adaption Interval\": 75, \"Final Fast Adaption Interval\": 50, \"Initial Slow Adaption Interval\": 25, \"Termination Criteria\": {\"Max Samples\": 500}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Multivariate Generator\": {\"Type\": \"Multivariate/Normal\"}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Sampler::applyModuleDefaults(js);
} 

void HMC::applyVariableDefaults() 
{

 Sampler::applyVariableDefaults();
} 

bool HMC::checkTermination()
{
 bool hasFinished = false;

 if (_sampleDatabase.size() >= _maxSamples)
 {
  _terminationCriteria.push_back("HMC['Max Samples'] = " + std::to_string(_maxSamples) + ".");
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
