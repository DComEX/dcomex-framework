#include "engine.hpp"
#include "modules/problem/optimization/optimization.hpp"
#include "modules/solver/optimizer/MOCMAES/MOCMAES.hpp"
#include "sample/sample.hpp"

#include <gsl/gsl_linalg.h> // Cholesky

namespace korali
{
namespace solver
{
namespace optimizer
{
;

void MOCMAES::setInitialConfiguration()
{
  knlohmann::json problemConfig = (*_k)["Problem"];
  _variableCount = _k->_variables.size();

  auto problemPtr = dynamic_cast<korali::problem::Optimization *>(_k->_problem);
  if (problemPtr == nullptr)
    KORALI_LOG_ERROR("Korali problem incompatible with MO-CMAES, must be of type 'Optimization'.\n");

  _numObjectives = problemPtr->_numObjectives;
  if (_numObjectives < 2)
    KORALI_LOG_ERROR("Problem requires multiple objectives, 'Num Objectives' is set to %zu\n.", _numObjectives);

  // Initializing Population Size
  if (_populationSize == 0) _populationSize = std::ceil(4. + floor(3. * log((double)_variableCount)));
  if (_muValue == 0) _muValue = _populationSize / 2.;
  if (_muValue > _populationSize)
    KORALI_LOG_ERROR("Number of parents ('Mu Value' %zu) must be smaller or equal with population size (%zu).\n", _muValue, _populationSize);

  _currentNonDominatedSampleCount = 1;

  // Allocating Sample and Value Memory
  _sampleCollection.resize(0);
  _sampleValueCollection.resize(0);
  _currentSamplePopulation.resize(_populationSize);
  _previousSamplePopulation.resize(_populationSize);
  _currentValues.resize(_populationSize);
  _previousValues.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++)
  {
    _currentSamplePopulation[i].resize(_variableCount);
    _previousSamplePopulation[i].resize(_variableCount);
    _currentValues[i].resize(_numObjectives, -Inf);
    _previousValues[i].resize(_numObjectives, -Inf);
  }

  // Establishing optimization goal
  _bestEverValues.resize(_numObjectives);
  _bestEverVariablesVector.resize(_numObjectives);
  _previousBestValues.resize(_numObjectives);
  _previousBestVariablesVector.resize(_numObjectives);
  _currentBestValues.resize(_numObjectives);
  _currentBestVariablesVector.resize(_numObjectives);
  for (size_t k = 0; k < _numObjectives; ++k)
  {
    _bestEverValues[k] = -Inf;
    _bestEverVariablesVector[k].resize(_variableCount);
    _previousBestValues[k] = -Inf;
    _previousBestVariablesVector[k].resize(_variableCount);
    _currentBestValues[k] = -Inf;
    _currentBestVariablesVector[k].resize(_variableCount);
  }

  // Disable general termination criterion
  _currentBestValue = +Inf;
  _previousBestValue = -Inf;

  // Initializing variable defaults
  double trace = 0.;
  double minSdev = Inf;
  double maxSdev = -Inf;
  for (size_t i = 0; i < _variableCount; i++)
  {
    if (std::isfinite(_k->_variables[i]->_initialValue) == false)
    {
      if (std::isfinite(_k->_variables[i]->_lowerBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable lower bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (std::isfinite(_k->_variables[i]->_upperBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable upper bound is not finite.\n", _k->_variables[i]->_name.c_str());
      _k->_variables[i]->_initialValue = (_k->_variables[i]->_upperBound + _k->_variables[i]->_lowerBound) * 0.5;
    }

    if (std::isfinite(_k->_variables[i]->_initialStandardDeviation) == false)
    {
      if (std::isfinite(_k->_variables[i]->_lowerBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable lower bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (std::isfinite(_k->_variables[i]->_upperBound) == false) KORALI_LOG_ERROR("Initial Standard Deviation \'%s\' not defined, and cannot be inferred because variable upper bound is not finite.\n", _k->_variables[i]->_name.c_str());
      _k->_variables[i]->_initialStandardDeviation = (_k->_variables[i]->_upperBound - _k->_variables[i]->_lowerBound) * 0.3;
    }
    trace += _k->_variables[i]->_initialStandardDeviation * _k->_variables[i]->_initialStandardDeviation;
    if (_k->_variables[i]->_initialStandardDeviation < minSdev) minSdev = _k->_variables[i]->_initialStandardDeviation;
    if (_k->_variables[i]->_initialStandardDeviation > maxSdev) maxSdev = _k->_variables[i]->_initialStandardDeviation;
  }

  // Initializing proposal
  _currentSigma.resize(_populationSize, 0.);
  _previousSigma.resize(_populationSize, 0.);
  _currentSuccessProbabilities.resize(_populationSize, 0.);
  _previousSuccessProbabilities.resize(_populationSize, 0.);
  _currentEvolutionPaths.resize(_populationSize);
  _previousEvolutionPaths.resize(_populationSize);
  _currentCovarianceMatrix.resize(_populationSize);
  _previousCovarianceMatrix.resize(_populationSize);

  // Init covariance and sigma
  for (size_t i = 0; i < _populationSize; ++i)
  {
    _currentEvolutionPaths[i].resize(_variableCount, 0.);
    _previousEvolutionPaths[i].resize(_variableCount, 0.);
    _currentCovarianceMatrix[i].resize(_variableCount * _variableCount, 0.);
    _previousCovarianceMatrix[i].resize(_variableCount * _variableCount, 0.);
  }

  // Init parent
  _parentIndex.resize(_populationSize, 0);
  _parentSuccessProbabilities.resize(_muValue, _targetSuccessRate);
  _parentSigma.resize(_muValue, std::sqrt(trace / _variableCount));
  _parentSamplePopulation.resize(_muValue);
  _parentEvolutionPaths.resize(_muValue);
  _parentCovarianceMatrix.resize(_muValue);
  for (size_t i = 0; i < _muValue; ++i)
  {
    _parentSamplePopulation[i].resize(_variableCount, 0.);
    _parentEvolutionPaths[i].resize(_variableCount, 0.);
    _parentCovarianceMatrix[i].resize(_variableCount * _variableCount, 0.);
    for (size_t d = 0; d < _variableCount; ++d)
      _parentCovarianceMatrix[i][d * _variableCount + d] = _k->_variables[d]->_initialStandardDeviation * _k->_variables[d]->_initialStandardDeviation / (_parentSigma[i] * _parentSigma[i]);
  }

  // MOCMAES variables
  if (_evolutionPathAdaptionStrength < 0.) _evolutionPathAdaptionStrength = 2. / (_variableCount + 2.);
  if (_covarianceLearningRate < 0.) _covarianceLearningRate = 2. / (_variableCount * _variableCount + 6.);
  if ((_successLearningRate <= 0.) || (_successLearningRate > 1.))
    KORALI_LOG_ERROR("Invalid Global Success Learning Rate (%f), must be greater than 0.0 and less or equal to 1.0\n", _successLearningRate);
  if ((_targetSuccessRate <= 0.) || (_targetSuccessRate > 1.))
    KORALI_LOG_ERROR("Invalid Target Success Rate (%f), must be greater than 0.0 and less or equal to 1.0\n", _targetSuccessRate);

  // Termination criteria
  _infeasibleSampleCount = 0;
  _currentMinStandardDeviations.resize(_populationSize, minSdev);
  _currentMaxStandardDeviations.resize(_populationSize, maxSdev);
  _currentBestValueDifferences.resize(_numObjectives, Inf);
  _currentBestVariableDifferences.resize(_numObjectives, Inf);
}

void MOCMAES::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  prepareGeneration();

  // Initializing Sample Evaluation
  std::vector<Sample> samples(_populationSize);
  for (size_t i = 0; i < _populationSize; i++)
  {
    samples[i]["Module"] = "Problem";
    samples[i]["Operation"] = "Evaluate Multiple";
    samples[i]["Parameters"] = _currentSamplePopulation[i];
    samples[i]["Sample Id"] = i;
    _modelEvaluationCount++;

    KORALI_START(samples[i]);
  }

  // Waiting for samples to finish
  KORALI_WAITALL(samples);

  // Gathering evaluations
  for (size_t i = 0; i < _populationSize; i++)
    _currentValues[i] = KORALI_GET(std::vector<double>, samples[i], "F(x)");

  updateDistribution();
  updateStatistics();
}

void MOCMAES::prepareGeneration()
{
  for (size_t i = 0; i < _populationSize; ++i)
  {
    _previousValues[i] = _currentValues[i];
    _previousSamplePopulation[i] = _currentSamplePopulation[i];
    _previousSigma[i] = _currentSigma[i];
    _previousCovarianceMatrix[i] = _currentCovarianceMatrix[i];
    _previousEvolutionPaths[i] = _currentEvolutionPaths[i];
    _previousSuccessProbabilities[i] = _currentSuccessProbabilities[i];
    sampleSingle(i);
  }
}

void MOCMAES::sampleSingle(size_t sampleIdx)
{
  size_t parentIdx;
  if (_muValue == _populationSize)
    parentIdx = sampleIdx;
  else
    parentIdx = std::floor(std::min(_muValue, _currentNonDominatedSampleCount) * _uniformGenerator->getRandomNumber());

  // Kepp track of parent
  _parentIndex[sampleIdx] = parentIdx;

  // Set distribution
  _multinormalGenerator->_meanVector = _parentSamplePopulation[parentIdx];
  _multinormalGenerator->_sigma = _parentCovarianceMatrix[parentIdx];
  gsl_matrix_view sig = gsl_matrix_view_array(_multinormalGenerator->_sigma.data(), _variableCount, _variableCount);

  // Calculate Cholesky decomp and scale with sigma
  int err = gsl_linalg_cholesky_decomp1(&sig.matrix);
  err = gsl_matrix_scale(&sig.matrix, _parentSigma[parentIdx]);

  if (err > 0) KORALI_LOG_ERROR("Error during Cholesky decomposition of covariance matrix.\n");
  _multinormalGenerator->updateDistribution();

  // Generate new sample
  bool isFeasible;
  do
  {
    _multinormalGenerator->getRandomVector(_currentSamplePopulation[sampleIdx].data(), _variableCount);
    isFeasible = isSampleFeasible(_currentSamplePopulation[sampleIdx]);
    // if (isFeasible == false) _infeasibleSampleCount++;

  } while (isFeasible == false);

  // Set Covariance
  _currentCovarianceMatrix[sampleIdx] = _parentCovarianceMatrix[parentIdx];

  // Set Sigma
  _currentSigma[sampleIdx] = _parentSigma[parentIdx];

  // Set success probability and evolution path
  _currentEvolutionPaths[sampleIdx] = _parentEvolutionPaths[parentIdx];
  _currentSuccessProbabilities[sampleIdx] = _parentSuccessProbabilities[parentIdx];
}

std::vector<int> MOCMAES::sortSampleIndices(const std::vector<std::vector<double>> &values) const
{
  const size_t numValues = values.size();

  // find rank based on non-dominance
  size_t minRank = numValues;
  std::vector<size_t> rank(numValues, 0);
  for (size_t r = numValues; r >= 1; --r) // start with the best ranks
  {
    size_t minMaxNBetter = _numObjectives;
    std::vector<size_t> maxNBetter(numValues, 0);
    for (size_t i = 0; i < numValues; ++i)
      if (rank[i] == 0) // sample i not yet ranked
      {
        for (size_t j = 0; j < numValues; ++j)
          if (i != j && rank[j] == 0) // compare not with yourself and not with already ranked samples
          {
            // Compare with other sample and count how many times it is better
            size_t nBetter = 0;
            for (size_t k = 0; k < _numObjectives; ++k)
              if (values[i][k] < values[j][k]) nBetter++;
            if (nBetter > maxNBetter[i]) maxNBetter[i] = nBetter;
          }

        if (maxNBetter[i] < minMaxNBetter) minMaxNBetter = maxNBetter[i];
      }

    for (size_t i = 0; i < numValues; ++i)
      if (rank[i] == 0 && maxNBetter[i] == minMaxNBetter) // sample i not yet ranked and amongst best performing
      {
        rank[i] = r;
        if (r < minRank) minRank = r;
      }
  }

  // remove offset
  const size_t maxRank = numValues - minRank;
  for (size_t i = 0; i < numValues; ++i)
    rank[i] -= minRank;

  // find reference point
  std::vector<double> reference(_numObjectives, Inf);
  for (size_t i = 0; i < numValues; ++i)
    for (size_t k = 0; k < _numObjectives; ++k)
      if (values[i][k] < reference[k])
        reference[k] = values[i][k];

  std::vector<int> sortedIndeces(numValues, -1);

  // sort samples based on contributing hypervolume
  int order = 0;
  for (size_t r = 0; r <= maxRank; ++r)
  {
    // next samples to sort
    std::vector<size_t> unsorted;
    for (size_t i = 0; i < numValues; ++i)
      if (rank[i] == r && sortedIndeces[i] == -1) unsorted.push_back(i);

    while (unsorted.size() > 0)
    {
      const size_t numUnsorted = unsorted.size();

      // init exclusive upper bounds
      std::vector<std::vector<double>> exclusiveUpperbound(numUnsorted);
      for (size_t i = 0; i < numUnsorted; ++i) exclusiveUpperbound[i] = std::vector<double>(_numObjectives, -Inf);

      // calculate exclusive upper bound
      for (size_t i = 0; i < numUnsorted; ++i)
        for (size_t j = 0; j < numUnsorted; ++j)
          if (i != j)
            for (size_t k = 0; k < _numObjectives; ++k)
              if (values[unsorted[j]][k] > exclusiveUpperbound[i][k]) exclusiveUpperbound[i][k] = values[unsorted[j]][k];

      // calculate (negative) contributing hypervolume
      std::vector<double> contributingHypervolume(numUnsorted, 0.0);
      for (size_t i = 0; i < numUnsorted; ++i)
        for (size_t k = 0; k < _numObjectives; ++k)
          contributingHypervolume[i] += (exclusiveUpperbound[i][k] - reference[k]);

      // sort samples ascending based on rank (primary) and descending based on contributing hypervolume (secondary)
      // descending because we use a different (negative) interpretation of the contributing hypervolume
      size_t nextSample = unsorted[0];
      double currentMaxContribution = contributingHypervolume[0];
      for (size_t i = 1; i < numUnsorted; ++i)
      {
        if (contributingHypervolume[i] > currentMaxContribution)
        {
          currentMaxContribution = contributingHypervolume[i];
          nextSample = unsorted[i];
        }
      }

      sortedIndeces[nextSample] = (int)order;

      unsorted.clear();
      for (size_t i = 0; i < numValues; ++i)
        if (rank[i] == r && sortedIndeces[i] == -1) unsorted.push_back(i);
      order++;
    }
  }

  return sortedIndeces;
}

void MOCMAES::updateDistribution()
{
  auto values = _currentValues;
  values.insert(std::end(values), std::begin(_previousValues), std::end(_previousValues));

  // Sort current and previous evlauations
  auto sortedIndices = sortSampleIndices(values);

  // Precompute factors
  const double evolutionPathFactor = std::sqrt(_evolutionPathAdaptionStrength * (2. - _evolutionPathAdaptionStrength));
  const double d = 1.0 + 2.0 * std::max(0.0, std::sqrt((_muValue - 1.) / (_variableCount + 1.)) - 1.0) + _evolutionPathAdaptionStrength;
  const double chiN = std::sqrt(_variableCount) * (1. - 1. / (4. * _variableCount) + 1. / (21. * _variableCount * _variableCount));

  for (size_t i = 0; i < _populationSize; ++i)
  {
    // Update Success rate
    _currentSuccessProbabilities[i] *= (1. - _successLearningRate);
    if ((size_t)sortedIndices[i] >= values.size() - _muValue) _currentSuccessProbabilities[i] += _successLearningRate; // sucess, considered a parent
    _currentSuccessProbabilities[i] = std::min(_currentSuccessProbabilities[i], 1.0);

    // Update Evolutionpath
    for (size_t d = 0; d < _variableCount; ++d)
      _currentEvolutionPaths[i][d] = (1. - _evolutionPathAdaptionStrength) * _currentEvolutionPaths[i][d];
    const auto &parent = _parentSamplePopulation[_parentIndex[i]];
    for (size_t d = 0; d < _variableCount; ++d)
      _currentEvolutionPaths[i][d] += evolutionPathFactor / std::sqrt(_currentCovarianceMatrix[i][d * _variableCount + d]) * (_currentSamplePopulation[i][d] - parent[d]) / _currentSigma[i];
    double evolutionPathLength = 0.;
    for (size_t d = 0; d < _variableCount; ++d) evolutionPathLength += _currentEvolutionPaths[i][d] * _currentEvolutionPaths[i][d];
    evolutionPathLength = std::sqrt(evolutionPathLength);

    // Update Covariance
    for (size_t d = 0; d < _variableCount; ++d)
      for (size_t e = 0; e < _variableCount; ++e)
        _currentCovarianceMatrix[i][d * _variableCount + e] = (1. - _covarianceLearningRate) * _currentCovarianceMatrix[i][d * _variableCount + e] + _covarianceLearningRate * _currentEvolutionPaths[i][d] * _currentEvolutionPaths[i][e];

    if (_currentSuccessProbabilities[i] >= _targetSuccessRate)
    {
      for (size_t d = 0; d < _variableCount; ++d)
        for (size_t e = 0; e < _variableCount; ++e)
          _currentCovarianceMatrix[i][d * _variableCount + e] += _covarianceLearningRate * evolutionPathFactor * evolutionPathFactor * _currentCovarianceMatrix[i][d * _variableCount + e];
    }

    // Update Sigma
    _currentSigma[i] *= std::exp(_evolutionPathAdaptionStrength / d * (evolutionPathLength / chiN - 1.0));
  }

  // Update parents
  for (size_t i = 0; i < sortedIndices.size(); ++i)
    if (sortedIndices[i] >= (int)(values.size() - _muValue))
    {
      // Insert parent samples descending, the 'largest' first
      size_t pidx = values.size() - sortedIndices[i] - 1;

      // Take sample from current generation
      if (i < _populationSize)
      {
        _parentSamplePopulation[pidx] = _currentSamplePopulation[i];
        _parentSigma[pidx] = _currentSigma[i];
        _parentCovarianceMatrix[pidx] = _currentCovarianceMatrix[i];
        _parentEvolutionPaths[pidx] = _currentEvolutionPaths[i];
        _parentSuccessProbabilities[pidx] = _currentSuccessProbabilities[i];
      }
      // Take sample from previous generation
      else
      {
        const size_t j = i - _populationSize;
        _parentSamplePopulation[pidx] = _previousSamplePopulation[j];
        _parentSigma[pidx] = _previousSigma[j];
        _parentCovarianceMatrix[pidx] = _previousCovarianceMatrix[j];
        _parentEvolutionPaths[pidx] = _previousEvolutionPaths[j];
        _parentSuccessProbabilities[pidx] = _previousSuccessProbabilities[j];
      }
      pidx++;
    }
}

void MOCMAES::updateStatistics()
{
  _previousBestValues = _currentBestValues;
  for (size_t k = 0; k < _numObjectives; ++k)
    _previousBestVariablesVector[k] = _currentBestVariablesVector[k];

  std::fill(_currentBestValues.begin(), _currentBestValues.end(), -Inf);
  std::fill(_currentMinStandardDeviations.begin(), _currentMinStandardDeviations.end(), Inf);
  std::fill(_currentMaxStandardDeviations.begin(), _currentMaxStandardDeviations.end(), -Inf);

  // Keep track of current best objectives and differences to previous generation
  for (size_t i = 0; i < _populationSize; ++i)
  {
    for (size_t k = 0; k < _numObjectives; ++k)
    {
      if (_currentValues[i][k] > _currentBestValues[k])
      {
        _currentBestValues[k] = _currentValues[i][k];
        _currentBestVariablesVector[k] = _currentSamplePopulation[i];
        _currentBestValueDifferences[k] = _currentBestValues[k] - _previousBestValues[k];
        double l2norm2 = 0.;
        for (size_t d = 0; d < _variableCount; ++d)
        {
          l2norm2 += std::pow(_previousBestVariablesVector[k][d] - _currentBestVariablesVector[k][d], 2.);
        }
        _currentBestVariableDifferences[k] = std::sqrt(l2norm2);
      }
    }
  }

  // Keep track of best obtained result for each objective
  for (size_t k = 0; k < _numObjectives; ++k)
  {
    if (_currentBestValues[k] > _bestEverValues[k])
    {
      _bestEverValues[k] = _currentBestValues[k];
      _bestEverVariablesVector[k] = _currentBestVariablesVector[k];
    }
  }

  // Store min / max standard deviations of offspring (sdev approximated)
  for (size_t i = 0; i < _populationSize; ++i)
  {
    for (size_t d = 0; d < _variableCount; ++d)
    {
      double sdev = _currentSigma[d] * std::sqrt(_currentCovarianceMatrix[i][d * _variableCount + d]);
      if (sdev > _currentMaxStandardDeviations[i]) _currentMaxStandardDeviations[i] = sdev;
      if (sdev < _currentMinStandardDeviations[i]) _currentMinStandardDeviations[i] = sdev;
    }
  }

  // Find non dominated samples of current generation
  std::vector<size_t> candidates;

  for (size_t i = 0; i < _populationSize; ++i)
  {
    bool dominated = false;
    for (size_t j = 0; j < _populationSize && dominated == false; ++j)
      if (j != i)
      {
        size_t numdom = 0;
        for (size_t k = 0; k < _numObjectives; ++k)
          if (_currentValues[j][k] > _currentValues[i][k]) numdom++;
        if (numdom == _numObjectives) dominated = true;
      }

    if (dominated == false) candidates.push_back(i);
  }

  _currentNonDominatedSampleCount = candidates.size();

  // Merge new candiates with sample collection
  std::vector<bool> keepCandiate(candidates.size(), true);
  std::vector<bool> keepSample(_sampleCollection.size(), true);

  for (size_t i = 0; i < candidates.size(); ++i)
  {
    for (size_t j = 0; j < _sampleCollection.size(); ++j)
    {
      size_t cdom = 0;
      size_t sdom = 0;

      for (size_t k = 0; k < _numObjectives; ++k)
      {
        if (_currentValues[candidates[i]][k] > _sampleValueCollection[j][k]) cdom++;
        if (_currentValues[candidates[i]][k] < _sampleValueCollection[j][k]) sdom++;
      }
      if (cdom == _numObjectives) keepSample[j] = false;
      if (sdom == _numObjectives) keepCandiate[i] = false;
    }
  }

  auto tmpSampleCollection = _sampleCollection;
  auto tmpSampleValueCollection = _sampleValueCollection;

  _sampleCollection.clear();
  _sampleValueCollection.clear();
  for (size_t i = 0; i < tmpSampleCollection.size(); ++i)
    if (keepSample[i])
    {
      _sampleCollection.push_back(tmpSampleCollection[i]);
      _sampleValueCollection.push_back(tmpSampleValueCollection[i]);
    }

  for (size_t i = 0; i < candidates.size(); ++i)
    if (keepCandiate[i])
    {
      _sampleCollection.push_back(_currentSamplePopulation[candidates[i]]);
      _sampleValueCollection.push_back(_currentValues[candidates[i]]);
    }
}

void MOCMAES::printGenerationBefore() { return; }

void MOCMAES::printGenerationAfter()
{
  _k->_logger->logInfo("Normal", "Current Function Values = (Max, Best):\n");
  for (size_t k = 0; k < _numObjectives; ++k) _k->_logger->logData("Normal", "                              = (%+6.3e, %+6.3e)\n", _currentBestValues[k], _bestEverValues[k]);

  double minSdev = *std::min_element(_currentMinStandardDeviations.begin(), _currentMinStandardDeviations.end());
  double maxSdev = *std::max_element(_currentMaxStandardDeviations.begin(), _currentMaxStandardDeviations.end());
  _k->_logger->logInfo("Normal", "Standard Devs:            Min = %+6.3e - Max = %+6.3e\n", minSdev, maxSdev);

  _k->_logger->logInfo("Normal", "Non Dominated Samples:   Current = %zu - Overall = %zu\n", _currentNonDominatedSampleCount, _sampleCollection.size());
  _k->_logger->logInfo("Detailed", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
}

void MOCMAES::finalize()
{
  // Updating Results
  (*_k)["Results"]["Pareto Optimal Samples"]["F(x)"] = _sampleValueCollection;
  (*_k)["Results"]["Pareto Optimal Samples"]["Parameters"] = _sampleCollection;

  /*
  _k->_logger->logInfo("Minimal", "Optimum found: %e\n", _bestEverValue);s
  _k->_logger->logInfo("Minimal", "Optimum found at:\n");
  for (size_t d = 0; d < _k->_variables.size(); ++d) _k->_logger->logData("Minimal", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverVariablesVector[d]);
  _k->_logger->logInfo("Minimal", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
  */
}

void MOCMAES::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Num Objectives"))
 {
 try { _numObjectives = js["Num Objectives"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Num Objectives']\n%s", e.what()); } 
   eraseValue(js, "Num Objectives");
 }

 if (isDefined(js, "Multinormal Generator"))
 {
 _multinormalGenerator = dynamic_cast<korali::distribution::multivariate::Normal*>(korali::Module::getModule(js["Multinormal Generator"], _k));
 _multinormalGenerator->applyVariableDefaults();
 _multinormalGenerator->applyModuleDefaults(js["Multinormal Generator"]);
 _multinormalGenerator->setConfiguration(js["Multinormal Generator"]);
   eraseValue(js, "Multinormal Generator");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Current Non Dominated Sample Count"))
 {
 try { _currentNonDominatedSampleCount = js["Current Non Dominated Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Non Dominated Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Current Non Dominated Sample Count");
 }

 if (isDefined(js, "Current Values"))
 {
 try { _currentValues = js["Current Values"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Values']\n%s", e.what()); } 
   eraseValue(js, "Current Values");
 }

 if (isDefined(js, "Previous Values"))
 {
 try { _previousValues = js["Previous Values"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Previous Values']\n%s", e.what()); } 
   eraseValue(js, "Previous Values");
 }

 if (isDefined(js, "Parent Index"))
 {
 try { _parentIndex = js["Parent Index"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Parent Index']\n%s", e.what()); } 
   eraseValue(js, "Parent Index");
 }

 if (isDefined(js, "Parent Sample Population"))
 {
 try { _parentSamplePopulation = js["Parent Sample Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Parent Sample Population']\n%s", e.what()); } 
   eraseValue(js, "Parent Sample Population");
 }

 if (isDefined(js, "Current Sample Population"))
 {
 try { _currentSamplePopulation = js["Current Sample Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Sample Population']\n%s", e.what()); } 
   eraseValue(js, "Current Sample Population");
 }

 if (isDefined(js, "Previous Sample Population"))
 {
 try { _previousSamplePopulation = js["Previous Sample Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Previous Sample Population']\n%s", e.what()); } 
   eraseValue(js, "Previous Sample Population");
 }

 if (isDefined(js, "Parent Sigma"))
 {
 try { _parentSigma = js["Parent Sigma"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Parent Sigma']\n%s", e.what()); } 
   eraseValue(js, "Parent Sigma");
 }

 if (isDefined(js, "Current Sigma"))
 {
 try { _currentSigma = js["Current Sigma"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Sigma']\n%s", e.what()); } 
   eraseValue(js, "Current Sigma");
 }

 if (isDefined(js, "Previous Sigma"))
 {
 try { _previousSigma = js["Previous Sigma"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Previous Sigma']\n%s", e.what()); } 
   eraseValue(js, "Previous Sigma");
 }

 if (isDefined(js, "Parent Covariance Matrix"))
 {
 try { _parentCovarianceMatrix = js["Parent Covariance Matrix"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Parent Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Parent Covariance Matrix");
 }

 if (isDefined(js, "Current Covariance Matrix"))
 {
 try { _currentCovarianceMatrix = js["Current Covariance Matrix"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Current Covariance Matrix");
 }

 if (isDefined(js, "Previous Covariance Matrix"))
 {
 try { _previousCovarianceMatrix = js["Previous Covariance Matrix"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Previous Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Previous Covariance Matrix");
 }

 if (isDefined(js, "Parent Evolution Paths"))
 {
 try { _parentEvolutionPaths = js["Parent Evolution Paths"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Parent Evolution Paths']\n%s", e.what()); } 
   eraseValue(js, "Parent Evolution Paths");
 }

 if (isDefined(js, "Current Evolution Paths"))
 {
 try { _currentEvolutionPaths = js["Current Evolution Paths"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Evolution Paths']\n%s", e.what()); } 
   eraseValue(js, "Current Evolution Paths");
 }

 if (isDefined(js, "Previous Evolution Paths"))
 {
 try { _previousEvolutionPaths = js["Previous Evolution Paths"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Previous Evolution Paths']\n%s", e.what()); } 
   eraseValue(js, "Previous Evolution Paths");
 }

 if (isDefined(js, "Parent Success Probabilities"))
 {
 try { _parentSuccessProbabilities = js["Parent Success Probabilities"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Parent Success Probabilities']\n%s", e.what()); } 
   eraseValue(js, "Parent Success Probabilities");
 }

 if (isDefined(js, "Current Success Probabilities"))
 {
 try { _currentSuccessProbabilities = js["Current Success Probabilities"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Success Probabilities']\n%s", e.what()); } 
   eraseValue(js, "Current Success Probabilities");
 }

 if (isDefined(js, "Previous Success Probabilities"))
 {
 try { _previousSuccessProbabilities = js["Previous Success Probabilities"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Previous Success Probabilities']\n%s", e.what()); } 
   eraseValue(js, "Previous Success Probabilities");
 }

 if (isDefined(js, "Finished Sample Count"))
 {
 try { _finishedSampleCount = js["Finished Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Finished Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Finished Sample Count");
 }

 if (isDefined(js, "Best Ever Values"))
 {
 try { _bestEverValues = js["Best Ever Values"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Best Ever Values']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Values");
 }

 if (isDefined(js, "Best Ever Variables Vector"))
 {
 try { _bestEverVariablesVector = js["Best Ever Variables Vector"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Best Ever Variables Vector']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Variables Vector");
 }

 if (isDefined(js, "Previous Best Values"))
 {
 try { _previousBestValues = js["Previous Best Values"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Previous Best Values']\n%s", e.what()); } 
   eraseValue(js, "Previous Best Values");
 }

 if (isDefined(js, "Previous Best Variables Vector"))
 {
 try { _previousBestVariablesVector = js["Previous Best Variables Vector"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Previous Best Variables Vector']\n%s", e.what()); } 
   eraseValue(js, "Previous Best Variables Vector");
 }

 if (isDefined(js, "Current Best Values"))
 {
 try { _currentBestValues = js["Current Best Values"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Best Values']\n%s", e.what()); } 
   eraseValue(js, "Current Best Values");
 }

 if (isDefined(js, "Current Best Variables Vector"))
 {
 try { _currentBestVariablesVector = js["Current Best Variables Vector"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Best Variables Vector']\n%s", e.what()); } 
   eraseValue(js, "Current Best Variables Vector");
 }

 if (isDefined(js, "Sample Collection"))
 {
 try { _sampleCollection = js["Sample Collection"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Sample Collection']\n%s", e.what()); } 
   eraseValue(js, "Sample Collection");
 }

 if (isDefined(js, "Sample Value Collection"))
 {
 try { _sampleValueCollection = js["Sample Value Collection"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Sample Value Collection']\n%s", e.what()); } 
   eraseValue(js, "Sample Value Collection");
 }

 if (isDefined(js, "Infeasible Sample Count"))
 {
 try { _infeasibleSampleCount = js["Infeasible Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Infeasible Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Infeasible Sample Count");
 }

 if (isDefined(js, "Current Best Value Differences"))
 {
 try { _currentBestValueDifferences = js["Current Best Value Differences"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Best Value Differences']\n%s", e.what()); } 
   eraseValue(js, "Current Best Value Differences");
 }

 if (isDefined(js, "Current Best Variable Differences"))
 {
 try { _currentBestVariableDifferences = js["Current Best Variable Differences"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Best Variable Differences']\n%s", e.what()); } 
   eraseValue(js, "Current Best Variable Differences");
 }

 if (isDefined(js, "Current Min Standard Deviations"))
 {
 try { _currentMinStandardDeviations = js["Current Min Standard Deviations"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Min Standard Deviations']\n%s", e.what()); } 
   eraseValue(js, "Current Min Standard Deviations");
 }

 if (isDefined(js, "Current Max Standard Deviations"))
 {
 try { _currentMaxStandardDeviations = js["Current Max Standard Deviations"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Current Max Standard Deviations']\n%s", e.what()); } 
   eraseValue(js, "Current Max Standard Deviations");
 }

 if (isDefined(js, "Population Size"))
 {
 try { _populationSize = js["Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Population Size']\n%s", e.what()); } 
   eraseValue(js, "Population Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Population Size'] required by MOCMAES.\n"); 

 if (isDefined(js, "Mu Value"))
 {
 try { _muValue = js["Mu Value"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Mu Value']\n%s", e.what()); } 
   eraseValue(js, "Mu Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mu Value'] required by MOCMAES.\n"); 

 if (isDefined(js, "Evolution Path Adaption Strength"))
 {
 try { _evolutionPathAdaptionStrength = js["Evolution Path Adaption Strength"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Evolution Path Adaption Strength']\n%s", e.what()); } 
   eraseValue(js, "Evolution Path Adaption Strength");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Evolution Path Adaption Strength'] required by MOCMAES.\n"); 

 if (isDefined(js, "Covariance Learning Rate"))
 {
 try { _covarianceLearningRate = js["Covariance Learning Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Covariance Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Covariance Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Covariance Learning Rate'] required by MOCMAES.\n"); 

 if (isDefined(js, "Target Success Rate"))
 {
 try { _targetSuccessRate = js["Target Success Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Target Success Rate']\n%s", e.what()); } 
   eraseValue(js, "Target Success Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Target Success Rate'] required by MOCMAES.\n"); 

 if (isDefined(js, "Threshold Probability"))
 {
 try { _thresholdProbability = js["Threshold Probability"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Threshold Probability']\n%s", e.what()); } 
   eraseValue(js, "Threshold Probability");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Threshold Probability'] required by MOCMAES.\n"); 

 if (isDefined(js, "Success Learning Rate"))
 {
 try { _successLearningRate = js["Success Learning Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Success Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Success Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Success Learning Rate'] required by MOCMAES.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Max Value Difference Threshold"))
 {
 try { _minMaxValueDifferenceThreshold = js["Termination Criteria"]["Min Max Value Difference Threshold"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Termination Criteria']['Min Max Value Difference Threshold']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Max Value Difference Threshold");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Max Value Difference Threshold'] required by MOCMAES.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Variable Difference Threshold"))
 {
 try { _minVariableDifferenceThreshold = js["Termination Criteria"]["Min Variable Difference Threshold"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Termination Criteria']['Min Variable Difference Threshold']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Variable Difference Threshold");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Variable Difference Threshold'] required by MOCMAES.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Standard Deviation"))
 {
 try { _minStandardDeviation = js["Termination Criteria"]["Min Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Termination Criteria']['Min Standard Deviation']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Standard Deviation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Standard Deviation'] required by MOCMAES.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Standard Deviation"))
 {
 try { _maxStandardDeviation = js["Termination Criteria"]["Max Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MOCMAES ] \n + Key:    ['Termination Criteria']['Max Standard Deviation']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Standard Deviation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Standard Deviation'] required by MOCMAES.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/MOCMAES";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: MOCMAES: \n%s\n", js.dump(2).c_str());
} 

void MOCMAES::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Population Size"] = _populationSize;
   js["Mu Value"] = _muValue;
   js["Evolution Path Adaption Strength"] = _evolutionPathAdaptionStrength;
   js["Covariance Learning Rate"] = _covarianceLearningRate;
   js["Target Success Rate"] = _targetSuccessRate;
   js["Threshold Probability"] = _thresholdProbability;
   js["Success Learning Rate"] = _successLearningRate;
   js["Termination Criteria"]["Min Max Value Difference Threshold"] = _minMaxValueDifferenceThreshold;
   js["Termination Criteria"]["Min Variable Difference Threshold"] = _minVariableDifferenceThreshold;
   js["Termination Criteria"]["Min Standard Deviation"] = _minStandardDeviation;
   js["Termination Criteria"]["Max Standard Deviation"] = _maxStandardDeviation;
   js["Num Objectives"] = _numObjectives;
 if(_multinormalGenerator != NULL) _multinormalGenerator->getConfiguration(js["Multinormal Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Current Non Dominated Sample Count"] = _currentNonDominatedSampleCount;
   js["Current Values"] = _currentValues;
   js["Previous Values"] = _previousValues;
   js["Parent Index"] = _parentIndex;
   js["Parent Sample Population"] = _parentSamplePopulation;
   js["Current Sample Population"] = _currentSamplePopulation;
   js["Previous Sample Population"] = _previousSamplePopulation;
   js["Parent Sigma"] = _parentSigma;
   js["Current Sigma"] = _currentSigma;
   js["Previous Sigma"] = _previousSigma;
   js["Parent Covariance Matrix"] = _parentCovarianceMatrix;
   js["Current Covariance Matrix"] = _currentCovarianceMatrix;
   js["Previous Covariance Matrix"] = _previousCovarianceMatrix;
   js["Parent Evolution Paths"] = _parentEvolutionPaths;
   js["Current Evolution Paths"] = _currentEvolutionPaths;
   js["Previous Evolution Paths"] = _previousEvolutionPaths;
   js["Parent Success Probabilities"] = _parentSuccessProbabilities;
   js["Current Success Probabilities"] = _currentSuccessProbabilities;
   js["Previous Success Probabilities"] = _previousSuccessProbabilities;
   js["Finished Sample Count"] = _finishedSampleCount;
   js["Best Ever Values"] = _bestEverValues;
   js["Best Ever Variables Vector"] = _bestEverVariablesVector;
   js["Previous Best Values"] = _previousBestValues;
   js["Previous Best Variables Vector"] = _previousBestVariablesVector;
   js["Current Best Values"] = _currentBestValues;
   js["Current Best Variables Vector"] = _currentBestVariablesVector;
   js["Sample Collection"] = _sampleCollection;
   js["Sample Value Collection"] = _sampleValueCollection;
   js["Infeasible Sample Count"] = _infeasibleSampleCount;
   js["Current Best Value Differences"] = _currentBestValueDifferences;
   js["Current Best Variable Differences"] = _currentBestVariableDifferences;
   js["Current Min Standard Deviations"] = _currentMinStandardDeviations;
   js["Current Max Standard Deviations"] = _currentMaxStandardDeviations;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Optimizer::getConfiguration(js);
} 

void MOCMAES::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Population Size\": 0, \"Mu Value\": 0, \"Evolution Path Adaption Strength\": -1.0, \"Covariance Learning Rate\": -1.0, \"Target Success Rate\": 0.175, \"Threshold Probability\": 0.44, \"Success Learning Rate\": 0.08, \"Termination Criteria\": {\"Min Max Value Difference Threshold\": -Infinity, \"Min Variable Difference Threshold\": -Infinity, \"Min Standard Deviation\": -Infinity, \"Max Standard Deviation\": Infinity}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Multinormal Generator\": {\"Type\": \"Multivariate/Normal\", \"Mean Vector\": [0.0], \"Sigma\": [1.0]}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void MOCMAES::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Optimizer::applyVariableDefaults();
} 

bool MOCMAES::checkTermination()
{
 bool hasFinished = false;

 if (_k->_currentGeneration > 1 && (std::abs(*std::max_element(_currentBestValueDifferences.begin(), _currentBestValueDifferences.end())) < _minValueDifferenceThreshold))
 {
  _terminationCriteria.push_back("MOCMAES['Min Max Value Difference Threshold'] = " + std::to_string(_minMaxValueDifferenceThreshold) + ".");
  hasFinished = true;
 }

 if (_k->_currentGeneration > 1 && (*std::max_element(_currentBestVariableDifferences.begin(), _currentBestVariableDifferences.end()) < _minVariableDifferenceThreshold))
 {
  _terminationCriteria.push_back("MOCMAES['Min Variable Difference Threshold'] = " + std::to_string(_minVariableDifferenceThreshold) + ".");
  hasFinished = true;
 }

 if (_k->_currentGeneration > 1 && (*std::max_element(_currentMinStandardDeviations.begin(), _currentMinStandardDeviations.end()) <= _minStandardDeviation))
 {
  _terminationCriteria.push_back("MOCMAES['Min Standard Deviation'] = " + std::to_string(_minStandardDeviation) + ".");
  hasFinished = true;
 }

 if (_k->_currentGeneration > 1 && (*std::min_element(_currentMaxStandardDeviations.begin(), _currentMaxStandardDeviations.end()) >= _maxStandardDeviation))
 {
  _terminationCriteria.push_back("MOCMAES['Max Standard Deviation'] = " + std::to_string(_maxStandardDeviation) + ".");
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
