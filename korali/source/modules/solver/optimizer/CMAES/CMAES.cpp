#include "engine.hpp"
#include "modules/solver/optimizer/CMAES/CMAES.hpp"
#include "sample/sample.hpp"

#include <algorithm> // std::sort
#include <chrono>
#include <gsl/gsl_eigen.h>
#include <numeric> // std::iota
#include <stdio.h>
#include <unistd.h>

namespace korali
{
namespace solver
{
namespace optimizer
{
;

void CMAES::setInitialConfiguration()
{
  knlohmann::json problemConfig = (*_k)["Problem"];
  _variableCount = _k->_variables.size();

  // Establishing optimization goal
  _bestEverValue = -std::numeric_limits<double>::infinity();

  _previousBestEverValue = _bestEverValue;
  _previousBestValue = _bestEverValue;
  _currentBestValue = _bestEverValue;

  if (_populationSize == 1) KORALI_LOG_ERROR("'Population Size' must be larger 1.");
  if (_muValue == 0) _muValue = _populationSize / 2;
  if (_viabilityMuValue == 0) _viabilityMuValue = _viabilityPopulationSize / 2;

  const size_t s_max = std::max(_populationSize, _viabilityPopulationSize);
  const size_t mu_max = std::max(_muValue, _viabilityMuValue);

  _chiSquareNumber = sqrt((double)_variableCount) * (1. - 1. / (4. * _variableCount) + 1. / (21. * _variableCount * _variableCount));
  _chiSquareNumberDiscreteMutations = sqrt((double)_variableCount) * (1. - 1. / (4. * _variableCount) + 1. / (21. * _variableCount * _variableCount));

  _hasConstraints = false;
  if (isDefined(problemConfig, "Constraints"))
  {
    std::vector<size_t> problemConstraints = problemConfig["Constraints"];
    if (problemConstraints.size() > 0) _hasConstraints = true;
    _constraintEvaluations.resize(problemConstraints.size());
  }

  _hasDiscreteVariables = false;
  /* check _granularity for discrete variables */
  for (size_t i = 0; i < _k->_variables.size(); i++)
  {
    if (_k->_variables[i]->_granularity < 0.0) KORALI_LOG_ERROR("Negative granularity for variable \'%s\'.\n", _k->_variables[i]->_name.c_str());
    if (_k->_variables[i]->_granularity > 0.0) _hasDiscreteVariables = true;
  }

  _isViabilityRegime = _hasConstraints;
  if (_isViabilityRegime)
  {
    _currentPopulationSize = _viabilityPopulationSize;
    _currentMuValue = _viabilityMuValue;
  }
  else
  {
    _currentPopulationSize = _populationSize;
    _currentMuValue = _muValue;
  }

  // Allocating Memory
  _samplePopulation.resize(s_max);
  for (size_t i = 0; i < s_max; i++) _samplePopulation[i].resize(_variableCount);

  _evolutionPath.resize(_variableCount);
  _conjugateEvolutionPath.resize(_variableCount);
  _auxiliarBDZMatrix.resize(_variableCount);
  _meanUpdate.resize(_variableCount);
  _currentMean.resize(_variableCount);
  _previousMean.resize(_variableCount);
  _bestEverVariables.resize(_variableCount);
  _axisLengths.resize(_variableCount);
  _auxiliarAxisLengths.resize(_variableCount);
  _currentBestVariables.resize(_variableCount);

  _sortingIndex.resize(s_max);
  _valueVector.resize(s_max);

  if (_useGradientInformation)
  {
    _gradients.resize(s_max);
    for (size_t i = 0; i < s_max; ++i) _gradients[i].resize(_variableCount);
    if (_gradientStepSize <= 0.) KORALI_LOG_ERROR("Gradient Step Size must be larger than 0.0 (is %f)", _gradientStepSize);
  }

  if (_mirroredSampling)
  {
    if (_populationSize % 2 == 1) KORALI_LOG_ERROR("Mirrored Sampling can only be applied with an even Sample Population (is %zu)", _populationSize);
    if (_hasConstraints) KORALI_LOG_ERROR("Mirrored Sampling not applicable to problems with constraints");
  }

  _covarianceMatrix.resize(_variableCount * _variableCount);
  _auxiliarCovarianceMatrix.resize(_variableCount * _variableCount);
  _covarianceEigenvectorMatrix.resize(_variableCount * _variableCount);
  _auxiliarCovarianceEigenvectorMatrix.resize(_variableCount * _variableCount);
  _bDZMatrix.resize(s_max * _variableCount);

  _maskingMatrix.resize(_variableCount);
  _maskingMatrixSigma.resize(_variableCount);
  _discreteMutations.resize(_variableCount * _populationSize);
  std::fill(std::begin(_discreteMutations), std::end(_discreteMutations), 0.0);

  _numberMaskingMatrixEntries = 0;
  _numberOfDiscreteMutations = 0;
  _muWeights.resize(mu_max);

  // Initializing variable defaults
  for (size_t i = 0; i < _variableCount; ++i)
  {
    if (std::isfinite(_k->_variables[i]->_initialValue) == false)
    {
      if (std::isfinite(_k->_variables[i]->_lowerBound) == false) KORALI_LOG_ERROR("'Initial Value' of variable \'%s\' not defined, and cannot be inferred because variable lower bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (std::isfinite(_k->_variables[i]->_upperBound) == false) KORALI_LOG_ERROR("'Initial Value' of variable \'%s\' not defined, and cannot be inferred because variable upper bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (_k->_variables[i]->_lowerBound >= _k->_variables[i]->_upperBound) KORALI_LOG_ERROR("Lower bound of variable \'%s\' is not strictly smaller than upper bound.\n", _k->_variables[i]->_name.c_str());
      _k->_variables[i]->_initialValue = (_k->_variables[i]->_upperBound + _k->_variables[i]->_lowerBound) * 0.5;
    }

    if (std::isfinite(_k->_variables[i]->_initialStandardDeviation) == false)
    {
      if (std::isfinite(_k->_variables[i]->_lowerBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable lower bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (std::isfinite(_k->_variables[i]->_upperBound) == false) KORALI_LOG_ERROR("Initial Standard Deviation \'%s\' not defined, and cannot be inferred because variable upper bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (_k->_variables[i]->_lowerBound >= _k->_variables[i]->_upperBound) KORALI_LOG_ERROR("Lower bound of variable \'%s\' is not strictly smaller than upper bound.\n", _k->_variables[i]->_name.c_str());
      _k->_variables[i]->_initialStandardDeviation = (_k->_variables[i]->_upperBound - _k->_variables[i]->_lowerBound) * 0.3;
    }
  }

  // CMAES variables
  if (_hasConstraints)
  {
    if ((_globalSuccessLearningRate <= 0.0) || (_globalSuccessLearningRate > 1.0))
      KORALI_LOG_ERROR("Invalid Global Success Learning Rate (%f), must be greater than 0.0 and less than 1.0\n", _globalSuccessLearningRate);
    if ((_targetSuccessRate <= 0.0) || (_targetSuccessRate > 1.0))
      KORALI_LOG_ERROR("Invalid Target Success Rate (%f), must be greater than 0.0 and less than 1.0\n", _targetSuccessRate);
    if (_covarianceMatrixAdaptionStrength <= 0.0)
      KORALI_LOG_ERROR("Invalid Adaption Size (%f), must be greater than 0.0\n", _covarianceMatrixAdaptionStrength);

    _globalSuccessRate = 0.5;
    _bestValidSample = -1;
    _sampleConstraintViolationCounts.resize(s_max);
    _viabilityBoundaries.resize(_constraintEvaluations.size());

    _viabilityImprovement.resize(s_max);
    _viabilityIndicator.resize(_constraintEvaluations.size());

    for (size_t c = 0; c < _constraintEvaluations.size(); c++) _viabilityIndicator[c].resize(s_max);
    for (size_t c = 0; c < _constraintEvaluations.size(); c++) _constraintEvaluations[c].resize(s_max);

    _normalConstraintApproximation.resize(_constraintEvaluations.size());
    for (size_t i = 0; i < _constraintEvaluations.size(); i++) _normalConstraintApproximation[i].resize(_variableCount);

    _bestConstraintEvaluations.resize(_constraintEvaluations.size());

    _normalVectorLearningRate = 1.0 / (2.0 + _variableCount);
    _covarianceMatrixAdaptionFactor = _covarianceMatrixAdaptionStrength / (_variableCount + 2.);
  }
  else
  {
    _globalSuccessRate = -1.0;
    _covarianceMatrixAdaptionFactor = -1.0;
    _bestValidSample = 0;
  }

  _covarianceMatrixAdaptationCount = 0;
  _maxConstraintViolationCount = 0;

  // Setting algorithm internal variables
  if (_hasConstraints)
    initMuWeights(_viabilityMuValue);
  else
    initMuWeights(_muValue);

  initCovariance();

  _infeasibleSampleCount = 0;
  _resampledParameterCount = 0;

  _conjugateEvolutionPathL2Norm = 0.0;

  for (size_t i = 0; i < _variableCount; i++) _currentMean[i] = _previousMean[i] = _k->_variables[i]->_initialValue;

  _currentMinStandardDeviation = +std::numeric_limits<double>::infinity();
  _currentMaxStandardDeviation = -std::numeric_limits<double>::infinity();
}

void CMAES::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  if (_hasConstraints) checkMeanAndSetRegime();
  prepareGeneration();
  if (_hasConstraints)
  {
    updateConstraints();
    handleConstraints();
  }

  std::string operation;
  if (_useGradientInformation)
    operation = "Evaluate With Gradients";
  else
    operation = "Evaluate";

  // Initializing Sample Evaluation
  std::vector<Sample> samples(_currentPopulationSize);
  for (size_t i = 0; i < _currentPopulationSize; i++)
  {
    if (_hasDiscreteVariables) discretize(_samplePopulation[i]);

    samples[i]["Module"] = "Problem";
    samples[i]["Operation"] = operation;
    samples[i]["Parameters"] = _samplePopulation[i];
    samples[i]["Sample Id"] = i;
    _modelEvaluationCount++;

    KORALI_START(samples[i]);
  }

  // Waiting for samples to finish
  KORALI_WAITALL(samples);

  // Gathering evaluations
  for (size_t i = 0; i < _currentPopulationSize; i++)
    _valueVector[i] = KORALI_GET(double, samples[i], "F(x)");

  if (_useGradientInformation)
    for (size_t i = 0; i < _currentPopulationSize; i++)
      _gradients[i] = KORALI_GET(std::vector<double>, samples[i], "Gradient");

  updateDistribution();
}

void CMAES::initMuWeights(size_t numsamplesmu)
{
  // Initializing Mu Weights
  if (_muType == "Linear")
    for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = numsamplesmu - i;
  else if (_muType == "Equal")
    for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1.;
  else if (_muType == "Logarithmic")
    for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = log(std::max((double)numsamplesmu, 0.5 * _currentPopulationSize) + 0.5) - log(i + 1.);
  else if (_muType == "Proportional")
    for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1.;
  else
    KORALI_LOG_ERROR("Invalid setting of Mu Type (%s) (Linear, Equal, Logarithmic, or Proportional accepted).", _muType.c_str());

  // Normalize weights vector and set mueff
  double s1 = 0.0;
  double s2 = 0.0;

  for (size_t i = 0; i < numsamplesmu; i++)
  {
    s1 += _muWeights[i];
    s2 += _muWeights[i] * _muWeights[i];
  }
  _effectiveMu = s1 * s1 / s2;

  for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] /= s1;

  // Setting Cumulative Covariancea
  if ((_initialCumulativeCovariance <= 0) || (_initialCumulativeCovariance > 1))
    _cumulativeCovariance = (4.0 + _effectiveMu / (1.0 * _variableCount)) / (_variableCount + 4.0 + 2.0 * _effectiveMu / (1.0 * _variableCount));
  else
    _cumulativeCovariance = _initialCumulativeCovariance;

  // Setting Sigma Cumulation Factor
  _sigmaCumulationFactor = _initialSigmaCumulationFactor;
  if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1)
  {
    if (_hasConstraints)
    {
      _sigmaCumulationFactor = sqrt(_effectiveMu) / (sqrt(_effectiveMu) + sqrt(_variableCount));
    }
    else
    {
      _sigmaCumulationFactor = (_effectiveMu + 2.0) / (_variableCount + _effectiveMu + 3.0);
    }
  }

  // Setting Damping Factor
  _dampFactor = _initialDampFactor;
  if (_dampFactor <= 0.0)
    _dampFactor = (1.0 + 2 * std::max(0.0, sqrt((_effectiveMu - 1.0) / (_variableCount + 1.0)) - 1)) + _sigmaCumulationFactor;
}

void CMAES::initCovariance()
{
  // Setting Sigma
  _trace = 0.0;
  for (size_t i = 0; i < _variableCount; ++i) _trace += _k->_variables[i]->_initialStandardDeviation * _k->_variables[i]->_initialStandardDeviation;
  _sigma = sqrt(_trace / _variableCount);

  // Setting B, C and _axisD
  for (size_t i = 0; i < _variableCount; ++i)
  {
    _covarianceEigenvectorMatrix[i * _variableCount + i] = 1.0;
    _covarianceMatrix[i * _variableCount + i] = _axisLengths[i] = _k->_variables[i]->_initialStandardDeviation * sqrt(_variableCount / _trace);
    _covarianceMatrix[i * _variableCount + i] *= _covarianceMatrix[i * _variableCount + i];
  }

  _minimumCovarianceEigenvalue = *std::min_element(std::begin(_axisLengths), std::end(_axisLengths));
  _maximumCovarianceEigenvalue = *std::max_element(std::begin(_axisLengths), std::end(_axisLengths));

  _minimumCovarianceEigenvalue = _minimumCovarianceEigenvalue * _minimumCovarianceEigenvalue;
  _maximumCovarianceEigenvalue = _maximumCovarianceEigenvalue * _maximumCovarianceEigenvalue;

  _maximumDiagonalCovarianceMatrixElement = _covarianceMatrix[0];
  for (size_t i = 1; i < _variableCount; ++i)
    if (_maximumDiagonalCovarianceMatrixElement < _covarianceMatrix[i * _variableCount + i]) _maximumDiagonalCovarianceMatrixElement = _covarianceMatrix[i * _variableCount + i];
  _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[0];
  for (size_t i = 1; i < _variableCount; ++i)
    if (_minimumDiagonalCovarianceMatrixElement > _covarianceMatrix[i * _variableCount + i]) _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[i * _variableCount + i];
}

void CMAES::checkMeanAndSetRegime()
{
  if (_isViabilityRegime == false) return; /* mean already inside valid domain, no udpates */

  if (_hasDiscreteVariables) discretize(_currentMean);

  Sample sample;
  sample["Sample Id"] = 0;
  sample["Parameters"] = _currentMean;
  sample["Module"] = "Problem";
  sample["Operation"] = "Evaluate Constraints";

  KORALI_START(sample);
  KORALI_WAIT(sample);
  _constraintEvaluationCount++;

  const auto cEvals = KORALI_GET(std::vector<double>, sample, "Constraint Evaluations");

  for (size_t c = 0; c < _constraintEvaluations.size(); c++)
    if (cEvals[c] > 0.0) return; /* mean violates constraint, do nothing */

  /* mean inside domain, switch regime and update internal variables */
  _isViabilityRegime = false;

  for (size_t c = 0; c < _constraintEvaluations.size(); c++) { _viabilityBoundaries[c] = 0; }
  _currentPopulationSize = _populationSize;
  _currentMuValue = _muValue;

  initMuWeights(_currentMuValue);
  initCovariance();
}

void CMAES::updateConstraints()
{
  for (size_t i = 0; i < _currentPopulationSize; i++)
  {
    _sampleConstraintViolationCounts[i] = 0;

    if (_hasDiscreteVariables) discretize(_samplePopulation[i]);

    Sample sample;
    sample["Sample Id"] = 0;
    sample["Parameters"] = _samplePopulation[i];
    sample["Module"] = "Problem";
    sample["Operation"] = "Evaluate Constraints";

    KORALI_START(sample);
    KORALI_WAIT(sample);
    _constraintEvaluationCount++;

    const auto cEvals = KORALI_GET(std::vector<double>, sample, "Constraint Evaluations");

    for (size_t c = 0; c < _constraintEvaluations.size(); c++)
      _constraintEvaluations[c][i] = cEvals[c];
  }

  _maxConstraintViolationCount = 0;

  for (size_t c = 0; c < _constraintEvaluations.size(); c++)
  {
    double maxviolation = 0.0;
    for (size_t i = 0; i < _currentPopulationSize; ++i)
    {
      if (_constraintEvaluations[c][i] > maxviolation) maxviolation = _constraintEvaluations[c][i];
      if (_k->_currentGeneration == 1 && _isViabilityRegime) _viabilityBoundaries[c] = maxviolation;

      if (_constraintEvaluations[c][i] > _viabilityBoundaries[c] + 1e-12) _sampleConstraintViolationCounts[i]++;
      if (_sampleConstraintViolationCounts[i] > _maxConstraintViolationCount) _maxConstraintViolationCount = _sampleConstraintViolationCounts[i];
    }
  }
}

void CMAES::reEvaluateConstraints()
{
  _maxConstraintViolationCount = 0;

  for (size_t i = 0; i < _currentPopulationSize; ++i)
    if (_sampleConstraintViolationCounts[i] > 0)
    {
      if (_hasDiscreteVariables) discretize(_samplePopulation[i]);

      Sample sample;
      sample["Sample Id"] = 0;
      sample["Parameters"] = _samplePopulation[i];
      sample["Module"] = "Problem";
      sample["Operation"] = "Evaluate Constraints";

      KORALI_START(sample);
      KORALI_WAIT(sample);
      _constraintEvaluationCount++;

      _sampleConstraintViolationCounts[i] = 0;

      const auto cEvals = KORALI_GET(std::vector<double>, sample, "Constraint Evaluations");

      for (size_t c = 0; c < _constraintEvaluations.size(); c++)
      {
        _constraintEvaluations[c][i] = cEvals[c];

        if (_constraintEvaluations[c][i] > _viabilityBoundaries[c] + 1e-12)
        {
          _viabilityIndicator[c][i] = true;
          _sampleConstraintViolationCounts[i]++;
        }
        else
          _viabilityIndicator[c][i] = false;
      }
      if (_sampleConstraintViolationCounts[i] > _maxConstraintViolationCount) _maxConstraintViolationCount = _sampleConstraintViolationCounts[i];
    }
}

void CMAES::updateViabilityBoundaries()
{
  for (size_t c = 0; c < _constraintEvaluations.size(); c++)
  {
    double maxviolation = 0.0;
    for (size_t i = 0; i < _currentMuValue /* _currentPopulationSize alternative */; ++i)
      if (_constraintEvaluations[c][_sortingIndex[i]] > maxviolation)
        maxviolation = _constraintEvaluations[c][_sortingIndex[i]];

    _viabilityBoundaries[c] = std::max(0.0, std::min(_viabilityBoundaries[c], 0.5 * (maxviolation + _viabilityBoundaries[c])));
  }
}

void CMAES::prepareGeneration()
{
  updateEigensystem(_covarianceMatrix);

  if (_mirroredSampling == false)
    for (size_t i = 0; i < _currentPopulationSize; ++i)
    {
      do
      {
        std::vector<double> rands(_variableCount);
        for (size_t d = 0; d < _variableCount; ++d) rands[d] = _normalGenerator->getRandomNumber();
        sampleSingle(i, rands);

        if (_hasDiscreteVariables) discretize(_samplePopulation[i]);

      } while (isSampleFeasible(_samplePopulation[i]) == false);
    }
  else
    for (size_t i = 0; i < _currentPopulationSize; i += 2)
    {
      bool isFeasible;
      do
      {
        std::vector<double> randsOne(_variableCount);
        std::vector<double> randsTwo(_variableCount);
        for (size_t d = 0; d < _variableCount; ++d)
        {
          randsOne[d] = _normalGenerator->getRandomNumber();
          randsTwo[d] = -randsOne[d];
        }

        sampleSingle(i, randsOne);
        sampleSingle(i + 1, randsTwo);

        if (_hasDiscreteVariables)
        {
          discretize(_samplePopulation[i]);
          discretize(_samplePopulation[i + 1]);
        }

        isFeasible = isSampleFeasible(_samplePopulation[i]);
        isFeasible = isFeasible && isSampleFeasible(_samplePopulation[i + 1]);

      } while (isFeasible == false);
    }
}

void CMAES::sampleSingle(size_t sampleIdx, const std::vector<double> &randomNumbers)
{
  for (size_t d = 0; d < _variableCount; ++d)
  {
    if (_diagonalCovariance)
    {
      _bDZMatrix[sampleIdx * _variableCount + d] = _axisLengths[d] * randomNumbers[d];
      _samplePopulation[sampleIdx][d] = _currentMean[d] + _sigma * _bDZMatrix[sampleIdx * _variableCount + d];
    }
    else
      _auxiliarBDZMatrix[d] = _axisLengths[d] * randomNumbers[d];
  }

  if (!_diagonalCovariance)
    for (size_t d = 0; d < _variableCount; ++d)
    {
      _bDZMatrix[sampleIdx * _variableCount + d] = 0.0;
      for (size_t e = 0; e < _variableCount; ++e) _bDZMatrix[sampleIdx * _variableCount + d] += _covarianceEigenvectorMatrix[d * _variableCount + e] * _auxiliarBDZMatrix[e];
      _samplePopulation[sampleIdx][d] = _currentMean[d] + _sigma * _bDZMatrix[sampleIdx * _variableCount + d];
    }

  if (_hasDiscreteVariables)
  {
    if ((sampleIdx + 1) < _numberOfDiscreteMutations)
    {
      const double p_geom = std::pow(0.7, 1.0 / _numberMaskingMatrixEntries);
      size_t select = std::floor(_uniformGenerator->getRandomNumber() * _numberMaskingMatrixEntries);

      for (size_t d = 0; d < _variableCount; ++d)
        if ((_maskingMatrix[d] == 1.0) && (select-- == 0))
        {
          double dmutation = 1.0;
          while (_uniformGenerator->getRandomNumber() > p_geom) dmutation += 1.0;
          dmutation *= _k->_variables[d]->_granularity;

          if (_uniformGenerator->getRandomNumber() > 0.5) dmutation *= -1.0;
          _discreteMutations[sampleIdx * _variableCount + d] = dmutation;
          _samplePopulation[sampleIdx][d] += dmutation;
        }
    }
    else if ((sampleIdx + 1) == _numberOfDiscreteMutations)
    {
      for (size_t d = 0; d < _variableCount; ++d)
        if (_k->_variables[d]->_granularity != 0.0)
        {
          const double dmutation = std::round(_bestEverVariables[d] / _k->_variables[d]->_granularity) * _k->_variables[d]->_granularity - _samplePopulation[sampleIdx][d];
          _discreteMutations[sampleIdx * _variableCount + d] = dmutation;
          _samplePopulation[sampleIdx][d] += dmutation;
        }
    }
  }
}

void CMAES::updateDistribution()
{
  /* Generate _sortingIndex */
  sort_index(_valueVector, _sortingIndex, _currentPopulationSize);

  if ((_hasConstraints == false) || _isViabilityRegime)
    _bestValidSample = _sortingIndex[0];
  else
  {
    _bestValidSample = -1;
    for (size_t i = 0; i < _currentPopulationSize; i++)
      if (_sampleConstraintViolationCounts[_sortingIndex[i]] == 0)
      {
        _bestValidSample = _sortingIndex[i];
        break;
      }
  }

  /* update function value history */
  _previousBestValue = _currentBestValue;

  /* update current best */
  _currentBestValue = _valueVector[_bestValidSample];

  for (size_t d = 0; d < _variableCount; ++d) _currentBestVariables[d] = _samplePopulation[_bestValidSample][d];

  /* update xbestever */
  if (_currentBestValue > _bestEverValue || _k->_currentGeneration == 1)
  {
    _previousBestEverValue = _bestEverValue;
    _bestEverValue = _currentBestValue;

    for (size_t d = 0; d < _variableCount; ++d)
      _bestEverVariables[d] = _currentBestVariables[d];

    if (_hasConstraints)
      for (size_t c = 0; c < _constraintEvaluations.size(); c++)
        _bestConstraintEvaluations[c] = _constraintEvaluations[c][_bestValidSample];
  }

  /* update proportional weights and effective mu (if required) */
  if (_muType == "Proportional")
  {
    double valueSum = 0.;
    for (size_t i = 0; i < _currentMuValue; ++i)
    {
      const double value = _valueVector[_sortingIndex[i]];
      _muWeights[i] = value;
      valueSum += value;
    }

    for (size_t i = 0; i < _currentMuValue; ++i)
      _muWeights[i] /= valueSum;
  }

  /* update mean */
  for (size_t d = 0; d < _variableCount; ++d)
  {
    _previousMean[d] = _currentMean[d];
    _currentMean[d] = 0.;
    for (size_t i = 0; i < _currentMuValue; ++i)
      _currentMean[d] += _muWeights[i] * _samplePopulation[_sortingIndex[i]][d];
  }

  if (_useGradientInformation)
  {
    double l2update = 0.;
    for (size_t d = 0; d < _variableCount; ++d)
      l2update += (_currentMean[d] - _previousMean[d]) * (_currentMean[d] - _previousMean[d]);
    l2update = std::sqrt(l2update / _variableCount);

    for (size_t d = 0; d < _variableCount; ++d)
      for (size_t i = 0; i < _currentMuValue; ++i)
        _currentMean[d] += _muWeights[i] * _gradientStepSize / std::sqrt(_variableCount) * _gradients[_sortingIndex[i]][d];
  }

  for (size_t d = 0; d < _variableCount; ++d)
    _meanUpdate[d] = (_currentMean[d] - _previousMean[d]) / _sigma;

  /* calculate z := D^(-1) * B^(T) * _meanUpdate into _auxiliarBDZMatrix */
  for (size_t d = 0; d < _variableCount; ++d)
  {
    double sum = 0.0;
    if (_diagonalCovariance)
      sum = _meanUpdate[d];
    else
      for (size_t e = 0; e < _variableCount; ++e) sum += _covarianceEigenvectorMatrix[e * _variableCount + d] * _meanUpdate[e]; /* B^(T) * _meanUpdate ( iterating B[e][d] = B^(T) ) */

    _auxiliarBDZMatrix[d] = sum / _axisLengths[d]; /* D^(-1) * B^(T) * _meanUpdate */
  }

  _conjugateEvolutionPathL2Norm = 0.0;

  /* cumulation for _sigma (ps) using B*z */
  for (size_t d = 0; d < _variableCount; ++d)
  {
    double sum = 0.0;
    if (_diagonalCovariance)
      sum = _auxiliarBDZMatrix[d];
    else
      for (size_t e = 0; e < _variableCount; ++e) sum += _covarianceEigenvectorMatrix[d * _variableCount + e] * _auxiliarBDZMatrix[e];

    _conjugateEvolutionPath[d] = (1. - _sigmaCumulationFactor) * _conjugateEvolutionPath[d] + std::sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _effectiveMu) * sum;

    /* calculate norm(ps)^2 */
    _conjugateEvolutionPathL2Norm += std::pow(_conjugateEvolutionPath[d], 2.0);
  }

  /* calculate norm(ps) */
  _conjugateEvolutionPathL2Norm = std::sqrt(_conjugateEvolutionPathL2Norm);

  const int hsig = (1.4 + 2.0 / (_variableCount + 1) > _conjugateEvolutionPathL2Norm / std::sqrt(1. - std::pow(1. - _sigmaCumulationFactor, 2.0 * (1.0 + _k->_currentGeneration))) / _chiSquareNumber);

  /* cumulation for covariance matrix (pc) using B*D*z~_variableCount(0,C) */
  for (size_t d = 0; d < _variableCount; ++d)
    _evolutionPath[d] = (1. - _cumulativeCovariance) * _evolutionPath[d] + hsig * sqrt(_cumulativeCovariance * (2. - _cumulativeCovariance) * _effectiveMu) * _meanUpdate[d];

  /* update covariance matrix  */
  adaptC(hsig);

  /* update masking matrix */
  if (_hasDiscreteVariables) updateDiscreteMutationMatrix();

  /* update viability bounds */
  if (_hasConstraints && (_isViabilityRegime == true)) updateViabilityBoundaries();

  /* update sigma */
  updateSigma();

  /* numerical error management */
  numericalErrorTreatment();

  _currentMinStandardDeviation = std::numeric_limits<double>::infinity();
  _currentMaxStandardDeviation = -std::numeric_limits<double>::infinity();

  // Calculating current Minimum and Maximum STD Devs
  for (size_t i = 0; i < _variableCount; ++i)
  {
    _currentMinStandardDeviation = std::min(_currentMinStandardDeviation, _sigma * std::sqrt(_covarianceMatrix[i * _variableCount + i]));
    _currentMaxStandardDeviation = std::max(_currentMaxStandardDeviation, _sigma * std::sqrt(_covarianceMatrix[i * _variableCount + i]));
  }
}

void CMAES::adaptC(int hsig)
{
  /* definitions for speeding up inner-most loop */
  const double ccov1 = 2.0 / (std::pow(_variableCount + 1.3, 2) + _effectiveMu);
  const double ccovmu = std::min(1.0 - ccov1, 2.0 * (_effectiveMu - 2. + 1. / _effectiveMu) / (std::pow(_variableCount + 2.0, 2) + _effectiveMu));
  const double sigmasquare = _sigma * _sigma;

  /* update covariance matrix */
  for (size_t d = 0; d < _variableCount; ++d)
    for (size_t e = _diagonalCovariance ? d : 0; e <= d; ++e)
    {
      _covarianceMatrix[d * _variableCount + e] = (1 - ccov1 - ccovmu) * _covarianceMatrix[d * _variableCount + e] + ccov1 * (_evolutionPath[d] * _evolutionPath[e] + (1 - hsig) * _cumulativeCovariance * (2. - _cumulativeCovariance) * _covarianceMatrix[d * _variableCount + e]);

      for (size_t k = 0; k < _currentMuValue; ++k)
        _covarianceMatrix[d * _variableCount + e] += ccovmu * _muWeights[k] * (_samplePopulation[_sortingIndex[k]][d] - _previousMean[d]) * (_samplePopulation[_sortingIndex[k]][e] - _previousMean[e]) / sigmasquare;

      if (e < d) _covarianceMatrix[e * _variableCount + d] = _covarianceMatrix[d * _variableCount + e];
    }

  /* update maximal and minimal diagonal value */
  _maximumDiagonalCovarianceMatrixElement = _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[0];
  for (size_t d = 1; d < _variableCount; ++d)
  {
    if (_maximumDiagonalCovarianceMatrixElement < _covarianceMatrix[d * _variableCount + d])
      _maximumDiagonalCovarianceMatrixElement = _covarianceMatrix[d * _variableCount + d];
    else if (_minimumDiagonalCovarianceMatrixElement > _covarianceMatrix[d * _variableCount + d])
      _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[d * _variableCount + d];
  }
}

void CMAES::updateSigma()
{
  /* update for non-viable region */
  if (_hasConstraints && (_isViabilityRegime == true))
  {
    _globalSuccessRate = (1 - _globalSuccessLearningRate) * _globalSuccessRate;

    _sigma *= exp((_globalSuccessRate - (_targetSuccessRate / (1.0 - _targetSuccessRate)) * (1 - _globalSuccessRate)) / _dampFactor);
  }
  /* update for discrte variables */
  else if (_hasDiscreteVariables)
  {
    double pathL2 = 0.0;
    for (size_t d = 0; d < _variableCount; ++d) pathL2 += _maskingMatrixSigma[d] * _conjugateEvolutionPath[d] * _conjugateEvolutionPath[d];
    _sigma *= exp(_sigmaCumulationFactor / _dampFactor * (sqrt(pathL2) / _chiSquareNumberDiscreteMutations - 1.));
  }
  /* standard update */
  else
  {
    _sigma *= exp(_sigmaCumulationFactor / _dampFactor * (_conjugateEvolutionPathL2Norm / _chiSquareNumber - 1.));
  }

  /* escape flat evaluation */
  if (_muValue > 1 && _currentBestValue == _valueVector[_sortingIndex[_currentMuValue - 1]])
  {
    _sigma *= exp(0.2 + _sigmaCumulationFactor / _dampFactor);
    _k->_logger->logWarning("Detailed", "Sigma increased due to equal function values.\n");
  }

  /* upper bound check for _sigma */
  const double _upperBound = sqrt(_trace / _variableCount);

  if (_sigma > _upperBound)
  {
    _k->_logger->logInfo("Detailed", "Sigma exceeding inital value of _sigma (%f > %f), increase Initial Standard Deviation of variables.\n", _sigma, _upperBound);
    if (_isSigmaBounded)
    {
      _sigma = _upperBound;
      _k->_logger->logInfo("Detailed", "Sigma set to upper bound (%f) due to solver configuration 'Is Sigma Bounded' = 'true'.\n", _sigma);
    }
  }
}

void CMAES::numericalErrorTreatment()
{
  // treat minimal standard deviations
  for (size_t d = 0; d < _variableCount; ++d)
    if (_sigma * sqrt(_covarianceMatrix[d * _variableCount + d]) < _k->_variables[d]->_minimumStandardDeviationUpdate)
    {
      _sigma = (_k->_variables[d]->_minimumStandardDeviationUpdate) / sqrt(_covarianceMatrix[d * _variableCount + d]) * exp(0.05 + _sigmaCumulationFactor / _dampFactor);
      _k->_logger->logWarning("Detailed", "Sigma increased due to minimal standard deviation.\n");
    }
}

void CMAES::handleConstraints()
{
  while (_maxConstraintViolationCount > 0)
  {
    _auxiliarCovarianceMatrix = _covarianceMatrix;

    for (size_t i = 0; i < _currentPopulationSize; ++i)
      if (_sampleConstraintViolationCounts[i] > 0)
      {
        // update constraint normal
        for (size_t c = 0; c < _constraintEvaluations.size(); c++)
          if (_viabilityIndicator[c][i] == true)
          {
            _covarianceMatrixAdaptationCount++;

            if (_covarianceMatrixAdaptationCount > _maxCovarianceMatrixCorrections)
            {
              _k->_logger->logWarning("Detailed", "Exiting adaption loop, max adaptions (%zu) reached.\n", _maxCovarianceMatrixCorrections);
              return;
            }

            double v2 = 0;
            for (size_t d = 0; d < _variableCount; ++d)
            {
              _normalConstraintApproximation[c][d] = (1.0 - _normalVectorLearningRate) * _normalConstraintApproximation[c][d] + _normalVectorLearningRate * _bDZMatrix[i * _variableCount + d];
              v2 += _normalConstraintApproximation[c][d] * _normalConstraintApproximation[c][d];
            }
            for (size_t d = 0; d < _variableCount; ++d)
              for (size_t e = 0; e < _variableCount; ++e)
                _auxiliarCovarianceMatrix[d * _variableCount + e] = _auxiliarCovarianceMatrix[d * _variableCount + e] - ((_covarianceMatrixAdaptionFactor * _covarianceMatrixAdaptionFactor * _normalConstraintApproximation[c][d] * _normalConstraintApproximation[c][e]) / (v2 * _sampleConstraintViolationCounts[i] * _sampleConstraintViolationCounts[i]));
          }
      }

    updateEigensystem(_auxiliarCovarianceMatrix);

    // resample invalid points
    for (size_t i = 0; i < _currentPopulationSize; ++i)
      if (_sampleConstraintViolationCounts[i] > 0)
      {
        Sample sample;
        bool isFeasible;

        do
        {
          _resampledParameterCount++;
          std::vector<double> rands(_variableCount);
          for (size_t d = 0; d < _variableCount; ++d) rands[d] = _normalGenerator->getRandomNumber();
          sampleSingle(i, rands);

          if (_hasDiscreteVariables) discretize(_samplePopulation[i]);
          isFeasible = isSampleFeasible(_samplePopulation[i]);

        } while (isFeasible == false && _resampledParameterCount < _maxInfeasibleResamplings);
      }

    reEvaluateConstraints();

  } // while _maxConstraintViolationCount > 0
}

void CMAES::updateDiscreteMutationMatrix()
{
  // implemented based on 'A CMA-ES for Mixed-Integer Nonlinear Optimization' by
  // Hansen2011

  size_t entries = _variableCount + 1; // +1 to prevent 0-ness
  std::fill(std::begin(_maskingMatrixSigma), std::end(_maskingMatrixSigma), 1.0);
  for (size_t d = 0; d < _variableCount; ++d)
    if (_sigma * std::sqrt(_covarianceMatrix[d * _variableCount + d]) / std::sqrt(_sigmaCumulationFactor) < 0.2 * _k->_variables[d]->_granularity)
    {
      _maskingMatrixSigma[d] = 0.0;
      entries--;
    }
  _chiSquareNumberDiscreteMutations = sqrt((double)entries) * (1. - 1. / (4. * entries) + 1. / (21. * entries * entries));

  _numberMaskingMatrixEntries = 0;
  std::fill(std::begin(_maskingMatrix), std::end(_maskingMatrix), 0.0);
  for (size_t d = 0; d < _variableCount; ++d)
    if (2.0 * _sigma * std::sqrt(_covarianceMatrix[d * _variableCount + d]) < _k->_variables[d]->_granularity)
    {
      _maskingMatrix[d] = 1.0;
      _numberMaskingMatrixEntries++;
    }

  _numberOfDiscreteMutations = std::min(std::round(_populationSize / 10.0 + _numberMaskingMatrixEntries + 1), std::floor(_populationSize / 2.0) - 1);
  std::fill(std::begin(_discreteMutations), std::end(_discreteMutations), 0.0);
}

void CMAES::discretize(std::vector<double> &sample)
{
  for (size_t d = 0; d < _variableCount; ++d)
    if (_k->_variables[d]->_granularity != 0.0)
      sample[d] = std::round(sample[d] / _k->_variables[d]->_granularity) * _k->_variables[d]->_granularity;
}

void CMAES::updateEigensystem(const std::vector<double> &M)
{
  eigen(_variableCount, M, _auxiliarAxisLengths, _auxiliarCovarianceEigenvectorMatrix);

  /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
  const double minCovEV = *std::min_element(std::begin(_auxiliarAxisLengths), std::end(_auxiliarAxisLengths));
  const double maxCovEV = *std::max_element(std::begin(_auxiliarAxisLengths), std::end(_auxiliarAxisLengths));
  if (minCovEV <= 0.0)
  {
    _k->_logger->logWarning("Detailed", "Min Eigenvalue smaller or equal 0.0 (%+6.3e) after Eigen decomp (no update possible).\n", _minimumCovarianceEigenvalue);
    return;
  }

  for (size_t d = 0; d < _variableCount; ++d) _auxiliarAxisLengths[d] = sqrt(_auxiliarAxisLengths[d]);

  _minimumCovarianceEigenvalue = minCovEV;
  _maximumCovarianceEigenvalue = maxCovEV;

  /* write back */
  for (size_t d = 0; d < _variableCount; ++d) _axisLengths[d] = _auxiliarAxisLengths[d];
  _covarianceEigenvectorMatrix.assign(std::begin(_auxiliarCovarianceEigenvectorMatrix), std::end(_auxiliarCovarianceEigenvectorMatrix));
}

/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/

void CMAES::eigen(size_t size, const std::vector<double> &M, std::vector<double> &diag, std::vector<double> &Q) const
{
  if (_diagonalCovariance)
  {
    std::fill(Q.begin(), Q.end(), 0);
    for (size_t i = 0; i < size; ++i) Q[i * size + i] = 1.;
    for (size_t i = 0; i < size; ++i) diag[i] = M[i * size + i];
  }
  else
  {
    std::vector<double> data(size * size);

    for (size_t i = 0; i < size; i++)
      for (size_t j = 0; j <= i; j++)
      {
        data[i * size + j] = M[i * size + j];
        data[j * size + i] = M[i * size + j];
      }

    // GSL Workspace

    gsl_vector *gsl_eval = gsl_vector_alloc(size);
    gsl_matrix *gsl_evec = gsl_matrix_alloc(size, size);
    gsl_eigen_symmv_workspace *gsl_work = gsl_eigen_symmv_alloc(size);

    gsl_matrix_view m = gsl_matrix_view_array(data.data(), size, size);

    gsl_eigen_symmv(&m.matrix, gsl_eval, gsl_evec, gsl_work);
    gsl_eigen_symmv_sort(gsl_eval, gsl_evec, GSL_EIGEN_SORT_ABS_ASC);

    for (size_t i = 0; i < size; i++)
    {
      gsl_vector_view gsl_evec_i = gsl_matrix_column(gsl_evec, i);
      for (size_t j = 0; j < size; j++) Q[j * size + i] = gsl_vector_get(&gsl_evec_i.vector, j);
    }

    for (size_t i = 0; i < size; i++) diag[i] = gsl_vector_get(gsl_eval, i);

    gsl_vector_free(gsl_eval);
    gsl_matrix_free(gsl_evec);
    gsl_eigen_symmv_free(gsl_work);
  }
}

void CMAES::sort_index(const std::vector<double> &vec, std::vector<size_t> &sortingIndex, size_t N) const
{
  // initialize original sortingIndex locations
  std::iota(std::begin(sortingIndex), std::begin(sortingIndex) + N, (size_t)0);

  // clang-format off
  // sort indexes based on comparing values in vec
  std::sort(std::begin(sortingIndex), std::begin(sortingIndex) + N, [vec](size_t i1, size_t i2)
            {
              return vec[i1] > vec[i2];
            });
  // clang-format on
}

void CMAES::printGenerationBefore() { return; }

void CMAES::printGenerationAfter()
{
  if (_hasConstraints && _isViabilityRegime)
  {
    _k->_logger->logInfo("Normal", "Searching start (MeanX violates constraints) .. \n");
    _k->_logger->logInfo("Normal", "Viability Bounds:\n");
    for (size_t c = 0; c < _constraintEvaluations.size(); c++) _k->_logger->logData("Normal", "         (%+6.3e)\n", _viabilityBoundaries[c]);
    _k->_logger->logInfo("Normal", "\n");
  }

  _k->_logger->logInfo("Normal", "Sigma:                        %+6.3e\n", _sigma);
  _k->_logger->logInfo("Normal", "Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
  _k->_logger->logInfo("Normal", "Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", _minimumDiagonalCovarianceMatrixElement, _maximumDiagonalCovarianceMatrixElement);
  _k->_logger->logInfo("Normal", "Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", _minimumCovarianceEigenvalue, _maximumCovarianceEigenvalue);

  _k->_logger->logInfo("Detailed", "Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _variableCount; d++) _k->_logger->logData("Detailed", "         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _currentMean[d], _bestEverVariables[d]);

  if (_hasConstraints)
  {
    _k->_logger->logInfo("Detailed", "Constraint Evaluation at Current Function Value:\n");
    if (_bestValidSample >= 0)
      for (size_t c = 0; c < _constraintEvaluations.size(); c++) _k->_logger->logData("Detailed", "         ( %+6.3e )\n", _constraintEvaluations[c][_bestValidSample]);
  }

  _k->_logger->logInfo("Detailed", "Covariance Matrix:\n");
  for (size_t d = 0; d < _variableCount; d++)
  {
    for (size_t e = 0; e <= d; e++) _k->_logger->logData("Detailed", "   %+6.3e  ", _covarianceMatrix[d * _variableCount + e]);
    _k->_logger->logInfo("Detailed", "\n");
  }

  _k->_logger->logInfo("Detailed", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
  if (_hasConstraints)
  {
    _k->_logger->logInfo("Detailed", "Number of Constraint Evaluations: %zu\n", _constraintEvaluationCount);
    _k->_logger->logInfo("Detailed", "Number of Matrix Corrections: %zu\n", _covarianceMatrixAdaptationCount);
  }
}

void CMAES::finalize()
{
  // Updating Results
  (*_k)["Results"]["Best Sample"]["F(x)"] = _bestEverValue;
  (*_k)["Results"]["Best Sample"]["Parameters"] = _bestEverVariables;

  _k->_logger->logInfo("Minimal", "Optimum found at:\n");
  for (size_t d = 0; d < _variableCount; ++d) _k->_logger->logData("Minimal", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverVariables[d]);
  _k->_logger->logInfo("Minimal", "Optimum found: %e\n", _bestEverValue);
  if (_hasConstraints)
  {
    _k->_logger->logInfo("Minimal", "Constraint Evaluation at Optimum:\n");
    for (size_t c = 0; c < _constraintEvaluations.size(); c++)
      _k->_logger->logData("Minimal", "         ( %+6.3e )\n", _bestConstraintEvaluations[c]);
  }
  _k->_logger->logInfo("Minimal", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
}

void CMAES::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Normal Generator"))
 {
 _normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(js["Normal Generator"], _k));
 _normalGenerator->applyVariableDefaults();
 _normalGenerator->applyModuleDefaults(js["Normal Generator"]);
 _normalGenerator->setConfiguration(js["Normal Generator"]);
   eraseValue(js, "Normal Generator");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Is Viability Regime"))
 {
 try { _isViabilityRegime = js["Is Viability Regime"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Is Viability Regime']\n%s", e.what()); } 
   eraseValue(js, "Is Viability Regime");
 }

 if (isDefined(js, "Value Vector"))
 {
 try { _valueVector = js["Value Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Value Vector']\n%s", e.what()); } 
   eraseValue(js, "Value Vector");
 }

 if (isDefined(js, "Gradients"))
 {
 try { _gradients = js["Gradients"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Gradients']\n%s", e.what()); } 
   eraseValue(js, "Gradients");
 }

 if (isDefined(js, "Current Population Size"))
 {
 try { _currentPopulationSize = js["Current Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Current Population Size']\n%s", e.what()); } 
   eraseValue(js, "Current Population Size");
 }

 if (isDefined(js, "Current Mu Value"))
 {
 try { _currentMuValue = js["Current Mu Value"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Current Mu Value']\n%s", e.what()); } 
   eraseValue(js, "Current Mu Value");
 }

 if (isDefined(js, "Mu Weights"))
 {
 try { _muWeights = js["Mu Weights"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Mu Weights']\n%s", e.what()); } 
   eraseValue(js, "Mu Weights");
 }

 if (isDefined(js, "Effective Mu"))
 {
 try { _effectiveMu = js["Effective Mu"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Effective Mu']\n%s", e.what()); } 
   eraseValue(js, "Effective Mu");
 }

 if (isDefined(js, "Sigma Cumulation Factor"))
 {
 try { _sigmaCumulationFactor = js["Sigma Cumulation Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Sigma Cumulation Factor']\n%s", e.what()); } 
   eraseValue(js, "Sigma Cumulation Factor");
 }

 if (isDefined(js, "Damp Factor"))
 {
 try { _dampFactor = js["Damp Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Damp Factor']\n%s", e.what()); } 
   eraseValue(js, "Damp Factor");
 }

 if (isDefined(js, "Cumulative Covariance"))
 {
 try { _cumulativeCovariance = js["Cumulative Covariance"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Cumulative Covariance']\n%s", e.what()); } 
   eraseValue(js, "Cumulative Covariance");
 }

 if (isDefined(js, "Chi Square Number"))
 {
 try { _chiSquareNumber = js["Chi Square Number"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Chi Square Number']\n%s", e.what()); } 
   eraseValue(js, "Chi Square Number");
 }

 if (isDefined(js, "Covariance Eigenvalue Evaluation Frequency"))
 {
 try { _covarianceEigenvalueEvaluationFrequency = js["Covariance Eigenvalue Evaluation Frequency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Covariance Eigenvalue Evaluation Frequency']\n%s", e.what()); } 
   eraseValue(js, "Covariance Eigenvalue Evaluation Frequency");
 }

 if (isDefined(js, "Sigma"))
 {
 try { _sigma = js["Sigma"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Sigma']\n%s", e.what()); } 
   eraseValue(js, "Sigma");
 }

 if (isDefined(js, "Trace"))
 {
 try { _trace = js["Trace"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Trace']\n%s", e.what()); } 
   eraseValue(js, "Trace");
 }

 if (isDefined(js, "Sample Population"))
 {
 try { _samplePopulation = js["Sample Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Sample Population']\n%s", e.what()); } 
   eraseValue(js, "Sample Population");
 }

 if (isDefined(js, "Finished Sample Count"))
 {
 try { _finishedSampleCount = js["Finished Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Finished Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Finished Sample Count");
 }

 if (isDefined(js, "Current Best Variables"))
 {
 try { _currentBestVariables = js["Current Best Variables"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Current Best Variables']\n%s", e.what()); } 
   eraseValue(js, "Current Best Variables");
 }

 if (isDefined(js, "Previous Best Ever Value"))
 {
 try { _previousBestEverValue = js["Previous Best Ever Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Previous Best Ever Value']\n%s", e.what()); } 
   eraseValue(js, "Previous Best Ever Value");
 }

 if (isDefined(js, "Sorting Index"))
 {
 try { _sortingIndex = js["Sorting Index"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Sorting Index']\n%s", e.what()); } 
   eraseValue(js, "Sorting Index");
 }

 if (isDefined(js, "Covariance Matrix"))
 {
 try { _covarianceMatrix = js["Covariance Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Covariance Matrix");
 }

 if (isDefined(js, "Auxiliar Covariance Matrix"))
 {
 try { _auxiliarCovarianceMatrix = js["Auxiliar Covariance Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Auxiliar Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Auxiliar Covariance Matrix");
 }

 if (isDefined(js, "Covariance Eigenvector Matrix"))
 {
 try { _covarianceEigenvectorMatrix = js["Covariance Eigenvector Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Covariance Eigenvector Matrix']\n%s", e.what()); } 
   eraseValue(js, "Covariance Eigenvector Matrix");
 }

 if (isDefined(js, "Auxiliar Covariance Eigenvector Matrix"))
 {
 try { _auxiliarCovarianceEigenvectorMatrix = js["Auxiliar Covariance Eigenvector Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Auxiliar Covariance Eigenvector Matrix']\n%s", e.what()); } 
   eraseValue(js, "Auxiliar Covariance Eigenvector Matrix");
 }

 if (isDefined(js, "Axis Lengths"))
 {
 try { _axisLengths = js["Axis Lengths"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Axis Lengths']\n%s", e.what()); } 
   eraseValue(js, "Axis Lengths");
 }

 if (isDefined(js, "Auxiliar Axis Lengths"))
 {
 try { _auxiliarAxisLengths = js["Auxiliar Axis Lengths"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Auxiliar Axis Lengths']\n%s", e.what()); } 
   eraseValue(js, "Auxiliar Axis Lengths");
 }

 if (isDefined(js, "BDZ Matrix"))
 {
 try { _bDZMatrix = js["BDZ Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['BDZ Matrix']\n%s", e.what()); } 
   eraseValue(js, "BDZ Matrix");
 }

 if (isDefined(js, "Auxiliar BDZ Matrix"))
 {
 try { _auxiliarBDZMatrix = js["Auxiliar BDZ Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Auxiliar BDZ Matrix']\n%s", e.what()); } 
   eraseValue(js, "Auxiliar BDZ Matrix");
 }

 if (isDefined(js, "Current Mean"))
 {
 try { _currentMean = js["Current Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Current Mean']\n%s", e.what()); } 
   eraseValue(js, "Current Mean");
 }

 if (isDefined(js, "Previous Mean"))
 {
 try { _previousMean = js["Previous Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Previous Mean']\n%s", e.what()); } 
   eraseValue(js, "Previous Mean");
 }

 if (isDefined(js, "Mean Update"))
 {
 try { _meanUpdate = js["Mean Update"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Mean Update']\n%s", e.what()); } 
   eraseValue(js, "Mean Update");
 }

 if (isDefined(js, "Evolution Path"))
 {
 try { _evolutionPath = js["Evolution Path"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Evolution Path']\n%s", e.what()); } 
   eraseValue(js, "Evolution Path");
 }

 if (isDefined(js, "Conjugate Evolution Path"))
 {
 try { _conjugateEvolutionPath = js["Conjugate Evolution Path"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Conjugate Evolution Path']\n%s", e.what()); } 
   eraseValue(js, "Conjugate Evolution Path");
 }

 if (isDefined(js, "Conjugate Evolution Path L2 Norm"))
 {
 try { _conjugateEvolutionPathL2Norm = js["Conjugate Evolution Path L2 Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Conjugate Evolution Path L2 Norm']\n%s", e.what()); } 
   eraseValue(js, "Conjugate Evolution Path L2 Norm");
 }

 if (isDefined(js, "Maximum Diagonal Covariance Matrix Element"))
 {
 try { _maximumDiagonalCovarianceMatrixElement = js["Maximum Diagonal Covariance Matrix Element"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Maximum Diagonal Covariance Matrix Element']\n%s", e.what()); } 
   eraseValue(js, "Maximum Diagonal Covariance Matrix Element");
 }

 if (isDefined(js, "Minimum Diagonal Covariance Matrix Element"))
 {
 try { _minimumDiagonalCovarianceMatrixElement = js["Minimum Diagonal Covariance Matrix Element"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Minimum Diagonal Covariance Matrix Element']\n%s", e.what()); } 
   eraseValue(js, "Minimum Diagonal Covariance Matrix Element");
 }

 if (isDefined(js, "Maximum Covariance Eigenvalue"))
 {
 try { _maximumCovarianceEigenvalue = js["Maximum Covariance Eigenvalue"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Maximum Covariance Eigenvalue']\n%s", e.what()); } 
   eraseValue(js, "Maximum Covariance Eigenvalue");
 }

 if (isDefined(js, "Minimum Covariance Eigenvalue"))
 {
 try { _minimumCovarianceEigenvalue = js["Minimum Covariance Eigenvalue"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Minimum Covariance Eigenvalue']\n%s", e.what()); } 
   eraseValue(js, "Minimum Covariance Eigenvalue");
 }

 if (isDefined(js, "Is Eigensystem Updated"))
 {
 try { _isEigensystemUpdated = js["Is Eigensystem Updated"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Is Eigensystem Updated']\n%s", e.what()); } 
   eraseValue(js, "Is Eigensystem Updated");
 }

 if (isDefined(js, "Viability Indicator"))
 {
 try { _viabilityIndicator = js["Viability Indicator"].get<std::vector<std::vector<int>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Viability Indicator']\n%s", e.what()); } 
   eraseValue(js, "Viability Indicator");
 }

 if (isDefined(js, "Has Constraints"))
 {
 try { _hasConstraints = js["Has Constraints"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Has Constraints']\n%s", e.what()); } 
   eraseValue(js, "Has Constraints");
 }

 if (isDefined(js, "Covariance Matrix Adaption Factor"))
 {
 try { _covarianceMatrixAdaptionFactor = js["Covariance Matrix Adaption Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Covariance Matrix Adaption Factor']\n%s", e.what()); } 
   eraseValue(js, "Covariance Matrix Adaption Factor");
 }

 if (isDefined(js, "Best Valid Sample"))
 {
 try { _bestValidSample = js["Best Valid Sample"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Best Valid Sample']\n%s", e.what()); } 
   eraseValue(js, "Best Valid Sample");
 }

 if (isDefined(js, "Global Success Rate"))
 {
 try { _globalSuccessRate = js["Global Success Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Global Success Rate']\n%s", e.what()); } 
   eraseValue(js, "Global Success Rate");
 }

 if (isDefined(js, "Viability Function Value"))
 {
 try { _viabilityFunctionValue = js["Viability Function Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Viability Function Value']\n%s", e.what()); } 
   eraseValue(js, "Viability Function Value");
 }

 if (isDefined(js, "Resampled Parameter Count"))
 {
 try { _resampledParameterCount = js["Resampled Parameter Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Resampled Parameter Count']\n%s", e.what()); } 
   eraseValue(js, "Resampled Parameter Count");
 }

 if (isDefined(js, "Covariance Matrix Adaptation Count"))
 {
 try { _covarianceMatrixAdaptationCount = js["Covariance Matrix Adaptation Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Covariance Matrix Adaptation Count']\n%s", e.what()); } 
   eraseValue(js, "Covariance Matrix Adaptation Count");
 }

 if (isDefined(js, "Viability Boundaries"))
 {
 try { _viabilityBoundaries = js["Viability Boundaries"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Viability Boundaries']\n%s", e.what()); } 
   eraseValue(js, "Viability Boundaries");
 }

 if (isDefined(js, "Viability Improvement"))
 {
 try { _viabilityImprovement = js["Viability Improvement"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Viability Improvement']\n%s", e.what()); } 
   eraseValue(js, "Viability Improvement");
 }

 if (isDefined(js, "Max Constraint Violation Count"))
 {
 try { _maxConstraintViolationCount = js["Max Constraint Violation Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Max Constraint Violation Count']\n%s", e.what()); } 
   eraseValue(js, "Max Constraint Violation Count");
 }

 if (isDefined(js, "Sample Constraint Violation Counts"))
 {
 try { _sampleConstraintViolationCounts = js["Sample Constraint Violation Counts"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Sample Constraint Violation Counts']\n%s", e.what()); } 
   eraseValue(js, "Sample Constraint Violation Counts");
 }

 if (isDefined(js, "Constraint Evaluations"))
 {
 try { _constraintEvaluations = js["Constraint Evaluations"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Constraint Evaluations']\n%s", e.what()); } 
   eraseValue(js, "Constraint Evaluations");
 }

 if (isDefined(js, "Normal Constraint Approximation"))
 {
 try { _normalConstraintApproximation = js["Normal Constraint Approximation"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Normal Constraint Approximation']\n%s", e.what()); } 
   eraseValue(js, "Normal Constraint Approximation");
 }

 if (isDefined(js, "Best Constraint Evaluations"))
 {
 try { _bestConstraintEvaluations = js["Best Constraint Evaluations"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Best Constraint Evaluations']\n%s", e.what()); } 
   eraseValue(js, "Best Constraint Evaluations");
 }

 if (isDefined(js, "Has Discrete Variables"))
 {
 try { _hasDiscreteVariables = js["Has Discrete Variables"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Has Discrete Variables']\n%s", e.what()); } 
   eraseValue(js, "Has Discrete Variables");
 }

 if (isDefined(js, "Discrete Mutations"))
 {
 try { _discreteMutations = js["Discrete Mutations"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Discrete Mutations']\n%s", e.what()); } 
   eraseValue(js, "Discrete Mutations");
 }

 if (isDefined(js, "Number Of Discrete Mutations"))
 {
 try { _numberOfDiscreteMutations = js["Number Of Discrete Mutations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Number Of Discrete Mutations']\n%s", e.what()); } 
   eraseValue(js, "Number Of Discrete Mutations");
 }

 if (isDefined(js, "Number Masking Matrix Entries"))
 {
 try { _numberMaskingMatrixEntries = js["Number Masking Matrix Entries"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Number Masking Matrix Entries']\n%s", e.what()); } 
   eraseValue(js, "Number Masking Matrix Entries");
 }

 if (isDefined(js, "Masking Matrix"))
 {
 try { _maskingMatrix = js["Masking Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Masking Matrix']\n%s", e.what()); } 
   eraseValue(js, "Masking Matrix");
 }

 if (isDefined(js, "Masking Matrix Sigma"))
 {
 try { _maskingMatrixSigma = js["Masking Matrix Sigma"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Masking Matrix Sigma']\n%s", e.what()); } 
   eraseValue(js, "Masking Matrix Sigma");
 }

 if (isDefined(js, "Chi Square Number Discrete Mutations"))
 {
 try { _chiSquareNumberDiscreteMutations = js["Chi Square Number Discrete Mutations"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Chi Square Number Discrete Mutations']\n%s", e.what()); } 
   eraseValue(js, "Chi Square Number Discrete Mutations");
 }

 if (isDefined(js, "Current Min Standard Deviation"))
 {
 try { _currentMinStandardDeviation = js["Current Min Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Current Min Standard Deviation']\n%s", e.what()); } 
   eraseValue(js, "Current Min Standard Deviation");
 }

 if (isDefined(js, "Current Max Standard Deviation"))
 {
 try { _currentMaxStandardDeviation = js["Current Max Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Current Max Standard Deviation']\n%s", e.what()); } 
   eraseValue(js, "Current Max Standard Deviation");
 }

 if (isDefined(js, "Constraint Evaluation Count"))
 {
 try { _constraintEvaluationCount = js["Constraint Evaluation Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Constraint Evaluation Count']\n%s", e.what()); } 
   eraseValue(js, "Constraint Evaluation Count");
 }

 if (isDefined(js, "Population Size"))
 {
 try { _populationSize = js["Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Population Size']\n%s", e.what()); } 
   eraseValue(js, "Population Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Population Size'] required by CMAES.\n"); 

 if (isDefined(js, "Mu Value"))
 {
 try { _muValue = js["Mu Value"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Mu Value']\n%s", e.what()); } 
   eraseValue(js, "Mu Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mu Value'] required by CMAES.\n"); 

 if (isDefined(js, "Mu Type"))
 {
 try { _muType = js["Mu Type"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Mu Type']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_muType == "Linear") validOption = true; 
 if (_muType == "Equal") validOption = true; 
 if (_muType == "Logarithmic") validOption = true; 
 if (_muType == "Proportional") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mu Type'] required by CMAES.\n", _muType.c_str()); 
}
   eraseValue(js, "Mu Type");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mu Type'] required by CMAES.\n"); 

 if (isDefined(js, "Initial Sigma Cumulation Factor"))
 {
 try { _initialSigmaCumulationFactor = js["Initial Sigma Cumulation Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Initial Sigma Cumulation Factor']\n%s", e.what()); } 
   eraseValue(js, "Initial Sigma Cumulation Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Sigma Cumulation Factor'] required by CMAES.\n"); 

 if (isDefined(js, "Initial Damp Factor"))
 {
 try { _initialDampFactor = js["Initial Damp Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Initial Damp Factor']\n%s", e.what()); } 
   eraseValue(js, "Initial Damp Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Damp Factor'] required by CMAES.\n"); 

 if (isDefined(js, "Use Gradient Information"))
 {
 try { _useGradientInformation = js["Use Gradient Information"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Use Gradient Information']\n%s", e.what()); } 
   eraseValue(js, "Use Gradient Information");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Use Gradient Information'] required by CMAES.\n"); 

 if (isDefined(js, "Gradient Step Size"))
 {
 try { _gradientStepSize = js["Gradient Step Size"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Gradient Step Size']\n%s", e.what()); } 
   eraseValue(js, "Gradient Step Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Gradient Step Size'] required by CMAES.\n"); 

 if (isDefined(js, "Is Sigma Bounded"))
 {
 try { _isSigmaBounded = js["Is Sigma Bounded"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Is Sigma Bounded']\n%s", e.what()); } 
   eraseValue(js, "Is Sigma Bounded");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Is Sigma Bounded'] required by CMAES.\n"); 

 if (isDefined(js, "Initial Cumulative Covariance"))
 {
 try { _initialCumulativeCovariance = js["Initial Cumulative Covariance"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Initial Cumulative Covariance']\n%s", e.what()); } 
   eraseValue(js, "Initial Cumulative Covariance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Cumulative Covariance'] required by CMAES.\n"); 

 if (isDefined(js, "Diagonal Covariance"))
 {
 try { _diagonalCovariance = js["Diagonal Covariance"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Diagonal Covariance']\n%s", e.what()); } 
   eraseValue(js, "Diagonal Covariance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Diagonal Covariance'] required by CMAES.\n"); 

 if (isDefined(js, "Mirrored Sampling"))
 {
 try { _mirroredSampling = js["Mirrored Sampling"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Mirrored Sampling']\n%s", e.what()); } 
   eraseValue(js, "Mirrored Sampling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mirrored Sampling'] required by CMAES.\n"); 

 if (isDefined(js, "Viability Population Size"))
 {
 try { _viabilityPopulationSize = js["Viability Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Viability Population Size']\n%s", e.what()); } 
   eraseValue(js, "Viability Population Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Viability Population Size'] required by CMAES.\n"); 

 if (isDefined(js, "Viability Mu Value"))
 {
 try { _viabilityMuValue = js["Viability Mu Value"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Viability Mu Value']\n%s", e.what()); } 
   eraseValue(js, "Viability Mu Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Viability Mu Value'] required by CMAES.\n"); 

 if (isDefined(js, "Max Covariance Matrix Corrections"))
 {
 try { _maxCovarianceMatrixCorrections = js["Max Covariance Matrix Corrections"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Max Covariance Matrix Corrections']\n%s", e.what()); } 
   eraseValue(js, "Max Covariance Matrix Corrections");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Max Covariance Matrix Corrections'] required by CMAES.\n"); 

 if (isDefined(js, "Target Success Rate"))
 {
 try { _targetSuccessRate = js["Target Success Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Target Success Rate']\n%s", e.what()); } 
   eraseValue(js, "Target Success Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Target Success Rate'] required by CMAES.\n"); 

 if (isDefined(js, "Covariance Matrix Adaption Strength"))
 {
 try { _covarianceMatrixAdaptionStrength = js["Covariance Matrix Adaption Strength"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Covariance Matrix Adaption Strength']\n%s", e.what()); } 
   eraseValue(js, "Covariance Matrix Adaption Strength");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Covariance Matrix Adaption Strength'] required by CMAES.\n"); 

 if (isDefined(js, "Normal Vector Learning Rate"))
 {
 try { _normalVectorLearningRate = js["Normal Vector Learning Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Normal Vector Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Normal Vector Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Normal Vector Learning Rate'] required by CMAES.\n"); 

 if (isDefined(js, "Global Success Learning Rate"))
 {
 try { _globalSuccessLearningRate = js["Global Success Learning Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Global Success Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Global Success Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Global Success Learning Rate'] required by CMAES.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Condition Covariance Matrix"))
 {
 try { _maxConditionCovarianceMatrix = js["Termination Criteria"]["Max Condition Covariance Matrix"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Termination Criteria']['Max Condition Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Condition Covariance Matrix");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Condition Covariance Matrix'] required by CMAES.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Standard Deviation"))
 {
 try { _minStandardDeviation = js["Termination Criteria"]["Min Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Termination Criteria']['Min Standard Deviation']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Standard Deviation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Standard Deviation'] required by CMAES.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Standard Deviation"))
 {
 try { _maxStandardDeviation = js["Termination Criteria"]["Max Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Termination Criteria']['Max Standard Deviation']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Standard Deviation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Standard Deviation'] required by CMAES.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Granularity"))
 {
 try { _k->_variables[i]->_granularity = _k->_js["Variables"][i]["Granularity"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ CMAES ] \n + Key:    ['Granularity']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Granularity");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Granularity'] required by CMAES.\n"); 

 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/CMAES";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: CMAES: \n%s\n", js.dump(2).c_str());
} 

void CMAES::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Population Size"] = _populationSize;
   js["Mu Value"] = _muValue;
   js["Mu Type"] = _muType;
   js["Initial Sigma Cumulation Factor"] = _initialSigmaCumulationFactor;
   js["Initial Damp Factor"] = _initialDampFactor;
   js["Use Gradient Information"] = _useGradientInformation;
   js["Gradient Step Size"] = _gradientStepSize;
   js["Is Sigma Bounded"] = _isSigmaBounded;
   js["Initial Cumulative Covariance"] = _initialCumulativeCovariance;
   js["Diagonal Covariance"] = _diagonalCovariance;
   js["Mirrored Sampling"] = _mirroredSampling;
   js["Viability Population Size"] = _viabilityPopulationSize;
   js["Viability Mu Value"] = _viabilityMuValue;
   js["Max Covariance Matrix Corrections"] = _maxCovarianceMatrixCorrections;
   js["Target Success Rate"] = _targetSuccessRate;
   js["Covariance Matrix Adaption Strength"] = _covarianceMatrixAdaptionStrength;
   js["Normal Vector Learning Rate"] = _normalVectorLearningRate;
   js["Global Success Learning Rate"] = _globalSuccessLearningRate;
   js["Termination Criteria"]["Max Condition Covariance Matrix"] = _maxConditionCovarianceMatrix;
   js["Termination Criteria"]["Min Standard Deviation"] = _minStandardDeviation;
   js["Termination Criteria"]["Max Standard Deviation"] = _maxStandardDeviation;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Is Viability Regime"] = _isViabilityRegime;
   js["Value Vector"] = _valueVector;
   js["Gradients"] = _gradients;
   js["Current Population Size"] = _currentPopulationSize;
   js["Current Mu Value"] = _currentMuValue;
   js["Mu Weights"] = _muWeights;
   js["Effective Mu"] = _effectiveMu;
   js["Sigma Cumulation Factor"] = _sigmaCumulationFactor;
   js["Damp Factor"] = _dampFactor;
   js["Cumulative Covariance"] = _cumulativeCovariance;
   js["Chi Square Number"] = _chiSquareNumber;
   js["Covariance Eigenvalue Evaluation Frequency"] = _covarianceEigenvalueEvaluationFrequency;
   js["Sigma"] = _sigma;
   js["Trace"] = _trace;
   js["Sample Population"] = _samplePopulation;
   js["Finished Sample Count"] = _finishedSampleCount;
   js["Current Best Variables"] = _currentBestVariables;
   js["Previous Best Ever Value"] = _previousBestEverValue;
   js["Sorting Index"] = _sortingIndex;
   js["Covariance Matrix"] = _covarianceMatrix;
   js["Auxiliar Covariance Matrix"] = _auxiliarCovarianceMatrix;
   js["Covariance Eigenvector Matrix"] = _covarianceEigenvectorMatrix;
   js["Auxiliar Covariance Eigenvector Matrix"] = _auxiliarCovarianceEigenvectorMatrix;
   js["Axis Lengths"] = _axisLengths;
   js["Auxiliar Axis Lengths"] = _auxiliarAxisLengths;
   js["BDZ Matrix"] = _bDZMatrix;
   js["Auxiliar BDZ Matrix"] = _auxiliarBDZMatrix;
   js["Current Mean"] = _currentMean;
   js["Previous Mean"] = _previousMean;
   js["Mean Update"] = _meanUpdate;
   js["Evolution Path"] = _evolutionPath;
   js["Conjugate Evolution Path"] = _conjugateEvolutionPath;
   js["Conjugate Evolution Path L2 Norm"] = _conjugateEvolutionPathL2Norm;
   js["Maximum Diagonal Covariance Matrix Element"] = _maximumDiagonalCovarianceMatrixElement;
   js["Minimum Diagonal Covariance Matrix Element"] = _minimumDiagonalCovarianceMatrixElement;
   js["Maximum Covariance Eigenvalue"] = _maximumCovarianceEigenvalue;
   js["Minimum Covariance Eigenvalue"] = _minimumCovarianceEigenvalue;
   js["Is Eigensystem Updated"] = _isEigensystemUpdated;
   js["Viability Indicator"] = _viabilityIndicator;
   js["Has Constraints"] = _hasConstraints;
   js["Covariance Matrix Adaption Factor"] = _covarianceMatrixAdaptionFactor;
   js["Best Valid Sample"] = _bestValidSample;
   js["Global Success Rate"] = _globalSuccessRate;
   js["Viability Function Value"] = _viabilityFunctionValue;
   js["Resampled Parameter Count"] = _resampledParameterCount;
   js["Covariance Matrix Adaptation Count"] = _covarianceMatrixAdaptationCount;
   js["Viability Boundaries"] = _viabilityBoundaries;
   js["Viability Improvement"] = _viabilityImprovement;
   js["Max Constraint Violation Count"] = _maxConstraintViolationCount;
   js["Sample Constraint Violation Counts"] = _sampleConstraintViolationCounts;
   js["Constraint Evaluations"] = _constraintEvaluations;
   js["Normal Constraint Approximation"] = _normalConstraintApproximation;
   js["Best Constraint Evaluations"] = _bestConstraintEvaluations;
   js["Has Discrete Variables"] = _hasDiscreteVariables;
   js["Discrete Mutations"] = _discreteMutations;
   js["Number Of Discrete Mutations"] = _numberOfDiscreteMutations;
   js["Number Masking Matrix Entries"] = _numberMaskingMatrixEntries;
   js["Masking Matrix"] = _maskingMatrix;
   js["Masking Matrix Sigma"] = _maskingMatrixSigma;
   js["Chi Square Number Discrete Mutations"] = _chiSquareNumberDiscreteMutations;
   js["Current Min Standard Deviation"] = _currentMinStandardDeviation;
   js["Current Max Standard Deviation"] = _currentMaxStandardDeviation;
   js["Constraint Evaluation Count"] = _constraintEvaluationCount;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Granularity"] = _k->_variables[i]->_granularity;
 } 
 Optimizer::getConfiguration(js);
} 

void CMAES::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Population Size\": 0, \"Mu Value\": 0, \"Mu Type\": \"Logarithmic\", \"Initial Sigma Cumulation Factor\": -1.0, \"Initial Damp Factor\": -1.0, \"Is Sigma Bounded\": false, \"Initial Cumulative Covariance\": -1.0, \"Use Gradient Information\": false, \"Gradient Step Size\": 0.01, \"Diagonal Covariance\": false, \"Mirrored Sampling\": false, \"Viability Population Size\": 2, \"Viability Mu Value\": 0, \"Max Covariance Matrix Corrections\": 1000000, \"Target Success Rate\": 0.1818, \"Covariance Matrix Adaption Strength\": 0.1, \"Normal Vector Learning Rate\": -1.0, \"Global Success Learning Rate\": 0.2, \"Termination Criteria\": {\"Max Condition Covariance Matrix\": Infinity, \"Min Standard Deviation\": -Infinity, \"Max Standard Deviation\": Infinity}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Best Ever Value\": -Infinity, \"Current Min Standard Deviation\": Infinity, \"Current Max Standard Deviation\": -Infinity, \"Minimum Covariance Eigenvalue\": Infinity, \"Maximum Covariance Eigenvalue\": -Infinity}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void CMAES::applyVariableDefaults() 
{

 std::string defaultString = "{\"Granularity\": 0.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Optimizer::applyVariableDefaults();
} 

bool CMAES::checkTermination()
{
 bool hasFinished = false;

 if (_k->_currentGeneration > 1 && (_maximumCovarianceEigenvalue >= _maxConditionCovarianceMatrix * _minimumCovarianceEigenvalue))
 {
  _terminationCriteria.push_back("CMAES['Max Condition Covariance Matrix'] = " + std::to_string(_maxConditionCovarianceMatrix) + ".");
  hasFinished = true;
 }

 if (_k->_currentGeneration > 1 && (_currentMinStandardDeviation <= _minStandardDeviation))
 {
  _terminationCriteria.push_back("CMAES['Min Standard Deviation'] = " + std::to_string(_minStandardDeviation) + ".");
  hasFinished = true;
 }

 if (_k->_currentGeneration > 1 && (_currentMaxStandardDeviation >= _maxStandardDeviation))
 {
  _terminationCriteria.push_back("CMAES['Max Standard Deviation'] = " + std::to_string(_maxStandardDeviation) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Optimizer::checkTermination();
 return hasFinished;
}

;

} //optimizer
} //solver
} //korali
;
