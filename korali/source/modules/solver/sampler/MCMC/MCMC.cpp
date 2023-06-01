#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/sampler/MCMC/MCMC.hpp"
#include "sample/sample.hpp"

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

void MCMC::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  if (_chainCovarianceScaling <= 0.0) KORALI_LOG_ERROR("Chain Covariance Scaling must be larger 0.0 (is %lf).\n", _chainCovarianceScaling);
  if (_leap < 1) KORALI_LOG_ERROR("Leap must be larger 0 (is %zu).\n", _leap);
  if (_burnIn < 0) KORALI_LOG_ERROR("Burn In must be larger equal 0 (is %zu).\n", _burnIn);
  if (_rejectionLevels < 1) KORALI_LOG_ERROR("Rejection Levels must be larger 0 (is %zu).\n", _rejectionLevels);
  if (_nonAdaptionPeriod < 0) KORALI_LOG_ERROR("Non Adaption Period must be larger equal 0 (is %zu).\n", _nonAdaptionPeriod);

  // Allocating MCMC memory
  _chainCandidate.resize(_rejectionLevels);
  for (size_t i = 0; i < _rejectionLevels; i++) _chainCandidate[i].resize(_variableCount);

  _choleskyDecompositionCovariance.resize(_variableCount * _variableCount);
  _chainLeader.resize(_variableCount);
  _chainCandidatesEvaluations.resize(_rejectionLevels);
  _rejectionAlphas.resize(_rejectionLevels);
  _chainMean.resize(_variableCount);
  _chainCovariancePlaceholder.resize(_variableCount * _variableCount);
  _chainCovariance.resize(_variableCount * _variableCount);
  _choleskyDecompositionChainCovariance.resize(_variableCount * _variableCount);

  std::fill(std::begin(_choleskyDecompositionCovariance), std::end(_choleskyDecompositionCovariance), 0.0);
  std::fill(std::begin(_choleskyDecompositionChainCovariance), std::end(_choleskyDecompositionChainCovariance), 0.0);

  for (size_t i = 0; i < _variableCount; i++) _chainLeader[i] = _k->_variables[i]->_initialMean;
  for (size_t i = 0; i < _variableCount; i++) _choleskyDecompositionCovariance[i * _variableCount + i] = _k->_variables[i]->_initialStandardDeviation;

  // Init Generation
  _acceptanceCount = 0;
  _proposedSampleCount = 0;
  _chainLength = 0;
  _chainLeaderEvaluation = -std::numeric_limits<double>::infinity();
  _acceptanceRate = 1.0;
}

void MCMC::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  bool _sampleAccepted = false;

  for (size_t i = 0; i < _rejectionLevels && _sampleAccepted == false; i++)
  {
    generateCandidate(i);

    auto sample = Sample();

    _modelEvaluationCount++;
    sample["Parameters"] = _chainCandidate[i];
    sample["Sample Id"] = _sampleDatabase.size();
    sample["Module"] = "Problem";
    sample["Operation"] = "Evaluate";
    KORALI_START(sample);
    KORALI_WAIT(sample);

    double evaluation = KORALI_GET(double, sample, "logP(x)");

    _chainCandidatesEvaluations[i] = evaluation;

    // Processing Result
    double denom;
    double _rejectionAlpha = recursiveAlpha(denom, _chainLeaderEvaluation, &_chainCandidatesEvaluations[0], i);

    if (_rejectionAlpha == 1.0 || _rejectionAlpha > _uniformGenerator->getRandomNumber())
    {
      _acceptanceCount++;
      _sampleAccepted = true;
      _chainLeaderEvaluation = _chainCandidatesEvaluations[i];
      _chainLeader = _chainCandidate[i];
    }
  }

  if ((_chainLength >= _burnIn) && (_k->_currentGeneration % _leap == 0))
  {
    _sampleDatabase.push_back(_chainLeader);
    _sampleEvaluationDatabase.push_back(_chainLeaderEvaluation);
  }

  updateState();
  _chainLength++;
}

void MCMC::choleskyDecomp(const std::vector<double> &inC, std::vector<double> &outL) const
{
  gsl_matrix *A = gsl_matrix_alloc(_variableCount, _variableCount);

  for (size_t d = 0; d < _variableCount; ++d)
    for (size_t e = 0; e < d; ++e)
    {
      gsl_matrix_set(A, d, e, inC[d * _variableCount + e]);
      gsl_matrix_set(A, e, d, inC[e * _variableCount + d]);
    }
  for (size_t d = 0; d < _variableCount; ++d) gsl_matrix_set(A, d, d, inC[d * _variableCount + d]);

  int err = gsl_linalg_cholesky_decomp1(A);

  if (err == GSL_EDOM)
  {
    _k->_logger->logWarning("Normal", "Chain Covariance negative definite (not updating Cholesky Decomposition of Chain Covariance).\n");
  }
  else
  {
    for (size_t d = 0; d < _variableCount; ++d)
      for (size_t e = 0; e < d; ++e)
      {
        outL[d * _variableCount + e] = gsl_matrix_get(A, d, e);
      }
    for (size_t d = 0; d < _variableCount; ++d) outL[d * _variableCount + d] = gsl_matrix_get(A, d, d);
  }

  gsl_matrix_free(A);
}

double MCMC::recursiveAlpha(double &denominator, const double leaderLoglikelihood, const double *loglikelihoods, size_t N) const
{
  // recursive formula from Trias[2009]

  if (N == 0)
  {
    denominator = exp(leaderLoglikelihood);
    return std::min(1.0, exp(loglikelihoods[0] - leaderLoglikelihood));
  }
  else
  {
    // revert sample array
    double *reversedLogLikelihoods = new double[N];
    for (size_t i = 0; i < N; ++i) reversedLogLikelihoods[i] = loglikelihoods[N - 1 - i];

    // update numerator (w. recursive calls)
    double numerator = std::exp(loglikelihoods[N]);
    for (size_t i = 0; i < N; ++i)
    {
      double dummyDenominator;
      double alphaNumerator = recursiveAlpha(dummyDenominator, loglikelihoods[N], reversedLogLikelihoods, i);
      numerator *= (1.0 - alphaNumerator);
    }
    delete[] reversedLogLikelihoods;

    if (numerator == 0.0) return 0.0;

    // update denomiator
    double denominatorStar;
    double alphaDenominator = recursiveAlpha(denominatorStar, leaderLoglikelihood, loglikelihoods, N - 1);
    denominator = denominatorStar * (1.0 - alphaDenominator);

    return std::min(1.0, numerator / denominator);
  }
}

void MCMC::generateCandidate(size_t sampleIdx)
{
  _proposedSampleCount++;

  if (sampleIdx == 0)
    for (size_t d = 0; d < _variableCount; ++d) _chainCandidate[sampleIdx][d] = _chainLeader[d];
  else
    for (size_t d = 0; d < _variableCount; ++d) _chainCandidate[sampleIdx][d] = _chainCandidate[sampleIdx - 1][d];

  if ((_useAdaptiveSampling == false) || (_sampleDatabase.size() <= _nonAdaptionPeriod + _burnIn))
    for (size_t d = 0; d < _variableCount; ++d)
      for (size_t e = 0; e < _variableCount; ++e) _chainCandidate[sampleIdx][d] += _choleskyDecompositionCovariance[d * _variableCount + e] * _normalGenerator->getRandomNumber();
  else
    for (size_t d = 0; d < _variableCount; ++d)
      for (size_t e = 0; e < _variableCount; ++e) _chainCandidate[sampleIdx][d] += _choleskyDecompositionChainCovariance[d * _variableCount + e] * _normalGenerator->getRandomNumber();
}

void MCMC::updateState()
{
  _acceptanceRate = ((double)_acceptanceCount / (double)_chainLength);

  if (_sampleDatabase.size() == 0) return;
  if (_sampleDatabase.size() == 1)
  {
    for (size_t d = 0; d < _variableCount; d++) _chainMean[d] = _chainLeader[d];
    return;
  }

  for (size_t d = 0; d < _variableCount; d++)
    for (size_t e = 0; e < d; e++)
    {
      _chainCovariancePlaceholder[d * _variableCount + e] = (_chainMean[d] - _chainLeader[d]) * (_chainMean[e] - _chainLeader[e]);
      _chainCovariancePlaceholder[e * _variableCount + d] = (_chainMean[d] - _chainLeader[d]) * (_chainMean[e] - _chainLeader[e]);
    }
  for (size_t d = 0; d < _variableCount; d++) _chainCovariancePlaceholder[d * _variableCount + d] = (_chainMean[d] - _chainLeader[d]) * (_chainMean[d] - _chainLeader[d]);

  // Chain Mean
  for (size_t d = 0; d < _variableCount; d++) _chainMean[d] = (_chainMean[d] * (_sampleDatabase.size() - 1) + _chainLeader[d]) / _sampleDatabase.size();

  for (size_t d = 0; d < _variableCount; d++)
    for (size_t e = 0; e < d; e++)
    {
      _chainCovariance[d * _variableCount + e] = (_sampleDatabase.size() - 2.0) / (_sampleDatabase.size() - 1.0) * _chainCovariance[d * _variableCount + e] + (_chainCovarianceScaling / _sampleDatabase.size()) * _chainCovariancePlaceholder[d * _variableCount + e];
      _chainCovariance[e * _variableCount + d] = (_sampleDatabase.size() - 2.0) / (_sampleDatabase.size() - 1.0) * _chainCovariance[d * _variableCount + e] + (_chainCovarianceScaling / _sampleDatabase.size()) * _chainCovariancePlaceholder[d * _variableCount + e];
    }
  for (size_t d = 0; d < _variableCount; d++)
    _chainCovariance[d * _variableCount + d] = (_sampleDatabase.size() - 2.0) / (_sampleDatabase.size() - 1.0) * _chainCovariance[d * _variableCount + d] + (_chainCovarianceScaling / _sampleDatabase.size()) * _chainCovariancePlaceholder[d * _variableCount + d];

  if ((_useAdaptiveSampling == true) && (_sampleDatabase.size() > _nonAdaptionPeriod)) choleskyDecomp(_chainCovariance, _choleskyDecompositionChainCovariance);
}

void MCMC::printGenerationBefore() { return; }

void MCMC::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Database Entries %ld\n", _sampleDatabase.size());

  _k->_logger->logInfo("Normal", "Accepted Samples: %zu\n", _acceptanceCount);
  _k->_logger->logInfo("Normal", "Acceptance Rate Proposals: %.2f%%\n", 100 * _acceptanceRate);

  _k->_logger->logInfo("Detailed", "Current Sample:\n");
  for (size_t d = 0; d < _variableCount; d++) _k->_logger->logData("Detailed", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _chainLeader[d]);

  _k->_logger->logInfo("Detailed", "Current Chain Mean:\n");
  for (size_t d = 0; d < _variableCount; d++) _k->_logger->logData("Detailed", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _chainMean[d]);
  _k->_logger->logInfo("Detailed", "Current Chain Covariance:\n");
  for (size_t d = 0; d < _variableCount; d++)
  {
    for (size_t e = 0; e <= d; e++) _k->_logger->logData("Detailed", "         %+6.3e  ", _chainCovariance[d * _variableCount + e]);
    _k->_logger->logData("Detailed", "\n");
  }
}

void MCMC::finalize()
{
  _k->_logger->logInfo("Minimal", "Number of Generated Samples: %zu\n", _proposedSampleCount);
  _k->_logger->logInfo("Minimal", "Acceptance Rate: %.2f%%\n", 100 * _acceptanceRate);
  if (_sampleDatabase.size() == _maxSamples) _k->_logger->logInfo("Minimal", "Max Samples Reached.\n");
  (*_k)["Results"]["Sample Database"] = _sampleDatabase;
}

void MCMC::setConfiguration(knlohmann::json& js) 
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

 if (isDefined(js, "Cholesky Decomposition Covariance"))
 {
 try { _choleskyDecompositionCovariance = js["Cholesky Decomposition Covariance"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Cholesky Decomposition Covariance']\n%s", e.what()); } 
   eraseValue(js, "Cholesky Decomposition Covariance");
 }

 if (isDefined(js, "Cholesky Decomposition Chain Covariance"))
 {
 try { _choleskyDecompositionChainCovariance = js["Cholesky Decomposition Chain Covariance"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Cholesky Decomposition Chain Covariance']\n%s", e.what()); } 
   eraseValue(js, "Cholesky Decomposition Chain Covariance");
 }

 if (isDefined(js, "Chain Leader"))
 {
 try { _chainLeader = js["Chain Leader"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Leader']\n%s", e.what()); } 
   eraseValue(js, "Chain Leader");
 }

 if (isDefined(js, "Chain Leader Evaluation"))
 {
 try { _chainLeaderEvaluation = js["Chain Leader Evaluation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Leader Evaluation']\n%s", e.what()); } 
   eraseValue(js, "Chain Leader Evaluation");
 }

 if (isDefined(js, "Chain Candidate"))
 {
 try { _chainCandidate = js["Chain Candidate"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Candidate']\n%s", e.what()); } 
   eraseValue(js, "Chain Candidate");
 }

 if (isDefined(js, "Chain Candidates Evaluations"))
 {
 try { _chainCandidatesEvaluations = js["Chain Candidates Evaluations"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Candidates Evaluations']\n%s", e.what()); } 
   eraseValue(js, "Chain Candidates Evaluations");
 }

 if (isDefined(js, "Rejection Alphas"))
 {
 try { _rejectionAlphas = js["Rejection Alphas"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Rejection Alphas']\n%s", e.what()); } 
   eraseValue(js, "Rejection Alphas");
 }

 if (isDefined(js, "Acceptance Rate"))
 {
 try { _acceptanceRate = js["Acceptance Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Acceptance Rate']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Rate");
 }

 if (isDefined(js, "Acceptance Count"))
 {
 try { _acceptanceCount = js["Acceptance Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Acceptance Count']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Count");
 }

 if (isDefined(js, "Proposed Sample Count"))
 {
 try { _proposedSampleCount = js["Proposed Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Proposed Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Proposed Sample Count");
 }

 if (isDefined(js, "Sample Database"))
 {
 try { _sampleDatabase = js["Sample Database"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Sample Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Database");
 }

 if (isDefined(js, "Sample Evaluation Database"))
 {
 try { _sampleEvaluationDatabase = js["Sample Evaluation Database"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Sample Evaluation Database']\n%s", e.what()); } 
   eraseValue(js, "Sample Evaluation Database");
 }

 if (isDefined(js, "Chain Mean"))
 {
 try { _chainMean = js["Chain Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Mean']\n%s", e.what()); } 
   eraseValue(js, "Chain Mean");
 }

 if (isDefined(js, "Chain Covariance Placeholder"))
 {
 try { _chainCovariancePlaceholder = js["Chain Covariance Placeholder"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Covariance Placeholder']\n%s", e.what()); } 
   eraseValue(js, "Chain Covariance Placeholder");
 }

 if (isDefined(js, "Chain Covariance"))
 {
 try { _chainCovariance = js["Chain Covariance"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Covariance']\n%s", e.what()); } 
   eraseValue(js, "Chain Covariance");
 }

 if (isDefined(js, "Chain Length"))
 {
 try { _chainLength = js["Chain Length"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Length']\n%s", e.what()); } 
   eraseValue(js, "Chain Length");
 }

 if (isDefined(js, "Burn In"))
 {
 try { _burnIn = js["Burn In"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Burn In']\n%s", e.what()); } 
   eraseValue(js, "Burn In");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Burn In'] required by MCMC.\n"); 

 if (isDefined(js, "Leap"))
 {
 try { _leap = js["Leap"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Leap']\n%s", e.what()); } 
   eraseValue(js, "Leap");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Leap'] required by MCMC.\n"); 

 if (isDefined(js, "Rejection Levels"))
 {
 try { _rejectionLevels = js["Rejection Levels"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Rejection Levels']\n%s", e.what()); } 
   eraseValue(js, "Rejection Levels");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Rejection Levels'] required by MCMC.\n"); 

 if (isDefined(js, "Use Adaptive Sampling"))
 {
 try { _useAdaptiveSampling = js["Use Adaptive Sampling"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Use Adaptive Sampling']\n%s", e.what()); } 
   eraseValue(js, "Use Adaptive Sampling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Use Adaptive Sampling'] required by MCMC.\n"); 

 if (isDefined(js, "Non Adaption Period"))
 {
 try { _nonAdaptionPeriod = js["Non Adaption Period"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Non Adaption Period']\n%s", e.what()); } 
   eraseValue(js, "Non Adaption Period");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Non Adaption Period'] required by MCMC.\n"); 

 if (isDefined(js, "Chain Covariance Scaling"))
 {
 try { _chainCovarianceScaling = js["Chain Covariance Scaling"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Chain Covariance Scaling']\n%s", e.what()); } 
   eraseValue(js, "Chain Covariance Scaling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Chain Covariance Scaling'] required by MCMC.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Samples"))
 {
 try { _maxSamples = js["Termination Criteria"]["Max Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Termination Criteria']['Max Samples']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Samples");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Samples'] required by MCMC.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Mean"))
 {
 try { _k->_variables[i]->_initialMean = _k->_js["Variables"][i]["Initial Mean"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Initial Mean']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Mean");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Mean'] required by MCMC.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Initial Standard Deviation"))
 {
 try { _k->_variables[i]->_initialStandardDeviation = _k->_js["Variables"][i]["Initial Standard Deviation"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ MCMC ] \n + Key:    ['Initial Standard Deviation']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Standard Deviation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Standard Deviation'] required by MCMC.\n"); 

 } 
 Sampler::setConfiguration(js);
 _type = "sampler/MCMC";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: MCMC: \n%s\n", js.dump(2).c_str());
} 

void MCMC::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Burn In"] = _burnIn;
   js["Leap"] = _leap;
   js["Rejection Levels"] = _rejectionLevels;
   js["Use Adaptive Sampling"] = _useAdaptiveSampling;
   js["Non Adaption Period"] = _nonAdaptionPeriod;
   js["Chain Covariance Scaling"] = _chainCovarianceScaling;
   js["Termination Criteria"]["Max Samples"] = _maxSamples;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Cholesky Decomposition Covariance"] = _choleskyDecompositionCovariance;
   js["Cholesky Decomposition Chain Covariance"] = _choleskyDecompositionChainCovariance;
   js["Chain Leader"] = _chainLeader;
   js["Chain Leader Evaluation"] = _chainLeaderEvaluation;
   js["Chain Candidate"] = _chainCandidate;
   js["Chain Candidates Evaluations"] = _chainCandidatesEvaluations;
   js["Rejection Alphas"] = _rejectionAlphas;
   js["Acceptance Rate"] = _acceptanceRate;
   js["Acceptance Count"] = _acceptanceCount;
   js["Proposed Sample Count"] = _proposedSampleCount;
   js["Sample Database"] = _sampleDatabase;
   js["Sample Evaluation Database"] = _sampleEvaluationDatabase;
   js["Chain Mean"] = _chainMean;
   js["Chain Covariance Placeholder"] = _chainCovariancePlaceholder;
   js["Chain Covariance"] = _chainCovariance;
   js["Chain Length"] = _chainLength;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Mean"] = _k->_variables[i]->_initialMean;
   _k->_js["Variables"][i]["Initial Standard Deviation"] = _k->_variables[i]->_initialStandardDeviation;
 } 
 Sampler::getConfiguration(js);
} 

void MCMC::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Burn In\": 0, \"Leap\": 1, \"Rejection Levels\": 1, \"Use Adaptive Sampling\": false, \"Non Adaption Period\": 0, \"Chain Covariance Scaling\": 1.0, \"Termination Criteria\": {\"Max Samples\": 5000}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Sampler::applyModuleDefaults(js);
} 

void MCMC::applyVariableDefaults() 
{

 Sampler::applyVariableDefaults();
} 

bool MCMC::checkTermination()
{
 bool hasFinished = false;

 if (_sampleDatabase.size() >= _maxSamples)
 {
  _terminationCriteria.push_back("MCMC['Max Samples'] = " + std::to_string(_maxSamples) + ".");
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
