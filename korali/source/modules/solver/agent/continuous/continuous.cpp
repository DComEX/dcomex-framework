#include "engine.hpp"
#include "modules/solver/agent/continuous/continuous.hpp"
#include "sample/sample.hpp"

#include <gsl/gsl_sf_psi.h>

namespace korali
{
namespace solver
{
namespace agent
{
;

void Continuous::initializeAgent()
{
  // Getting continuous problem pointer
  _problem = dynamic_cast<problem::reinforcementLearning::Continuous *>(_k->_problem);

  // Only relevant for discrete
  _problem->_actionCount = 0;

  // Obtaining action shift and scales for bounded distributions
  _actionShifts.resize(_problem->_actionVectorSize);
  _actionScales.resize(_problem->_actionVectorSize);

  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    // For bounded distributions, infinite bounds should result in an error message
    if (_policyDistribution == "Squashed Normal" || _policyDistribution == "Beta" || _policyDistribution == "Clipped Normal" || _policyDistribution == "Truncated Normal")
    {
      if (isfinite(_actionLowerBounds[i]) == false)
        KORALI_LOG_ERROR("Provided lower bound (%f) for action variable %lu is non-finite, but the distribution (%s) is bounded.\n", _actionLowerBounds[i], i, _policyDistribution.c_str());

      if (isfinite(_actionUpperBounds[i]) == false)
        KORALI_LOG_ERROR("Provided upper bound (%f) for action variable %lu is non-finite, but the distribution (%s) is bounded.\n", _actionUpperBounds[i], i, _policyDistribution.c_str());

      _actionShifts[i] = (_actionUpperBounds[i] + _actionLowerBounds[i]) * 0.5f;
      _actionScales[i] = (_actionUpperBounds[i] - _actionLowerBounds[i]) * 0.5f;
    }
  }

  // Obtaining policy parameter transformations (depends on which policy distribution chosen)
  if (_policyDistribution == "Normal" || _policyDistribution == "Squashed Normal" || _policyDistribution == "Clipped Normal" || _policyDistribution == "Truncated Normal")
  {
    _policyParameterCount = 2 * _problem->_actionVectorSize; // Mus and Sigmas

    _policyParameterTransformationMasks.resize(_policyParameterCount);
    _policyParameterScaling.resize(_policyParameterCount);
    _policyParameterShifting.resize(_policyParameterCount);

    // Establishing transformations for the Normal policy
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const auto varIdx = _problem->_actionVectorIndexes[i];
      const float sigma = _k->_variables[varIdx]->_initialExplorationNoise;

      // Checking correct noise configuration
      if (sigma <= 0.0f) KORALI_LOG_ERROR("Provided initial noise (%f) for action variable %lu is not defined or negative.\n", sigma, varIdx);

      // Identity mask for Means
      _policyParameterTransformationMasks[i] = "Identity";
      _policyParameterScaling[i] = 1.0; //_actionScales[i];
      _policyParameterShifting[i] = _actionShifts[i];

      // Softplus mask for Sigmas
      _policyParameterTransformationMasks[_problem->_actionVectorSize + i] = "Softplus"; // f(x) = 0.5 * (x + std::sqrt(1. + x * x));
      _policyParameterScaling[_problem->_actionVectorSize + i] = 2.0f * sigma;
      _policyParameterShifting[_problem->_actionVectorSize + i] = 0.0f;
    }
  }

  if (_policyDistribution == "Beta")
  {
    _policyParameterCount = 2 * _problem->_actionVectorSize; // Mu and Variance
    _policyParameterTransformationMasks.resize(_policyParameterCount);
    _policyParameterScaling.resize(_policyParameterCount);
    _policyParameterShifting.resize(_policyParameterCount);

    // Establishing transformations for the Normal policy
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const size_t varIdx = _problem->_actionVectorIndexes[i];
      const float sigma = _k->_variables[varIdx]->_initialExplorationNoise;

      // Checking correct noise configuration
      if (sigma <= 0.0f) KORALI_LOG_ERROR("Provided initial noise (%f) for action variable %lu is not defined or negative.\n", sigma, varIdx);

      // Identity mask for Means
      _policyParameterScaling[i] = 1.0f;
      _policyParameterShifting[i] = _actionShifts[i];
      _policyParameterTransformationMasks[i] = "Identity";

      // Sigmoid Mask for Variance
      _policyParameterTransformationMasks[_problem->_actionVectorSize + i] = "Sigmoid";
      _policyParameterScaling[_problem->_actionVectorSize + i] = 2.0f * sigma;
      _policyParameterShifting[_problem->_actionVectorSize + i] = 0.0f;
    }
  }
}

void Continuous::getAction(korali::Sample &sample)
{
  // Get action for all the agents in the environment
  for (size_t i = 0; i < _problem->_agentsPerEnvironment; i++)
  {
    // Getting current state

    auto state = sample["State"][i];

    // Adding state to the state time sequence
    _stateTimeSequence[i].add(state);

    // Storage for the action to select
    std::vector<float> action(_problem->_actionVectorSize);

    // Forward state sequence to get the Gaussian means and sigmas from policy
    // in case of multiple polices it runs the i-th policy otherwise standard
    std::vector<policy_t> policy;
    if (_problem->_policiesPerEnvironment == 1)
      runPolicy({_stateTimeSequence[i].getVector()}, policy);
    else
      runPolicy({_stateTimeSequence[i].getVector()}, policy, i);

    /*****************************************************************************
     * During Training we select action according to policy's probability
     * distribution
     ****************************************************************************/

    if (sample["Mode"] == "Training") action = generateTrainingAction(policy[0]);

    /*****************************************************************************
     * During testing, we select the modes for all
     * elements of the action vector
     ****************************************************************************/

    if (sample["Mode"] == "Testing") action = generateTestingAction(policy[0]);

    /*****************************************************************************
     * Storing the action and its policy
     ****************************************************************************/

    // Check action
    for (size_t j = 0; j < _problem->_actionVectorSize; j++)
      if (std::isfinite(action[j]) == false) KORALI_LOG_ERROR("Agent %lu action %lu returned an invalid value: %f\n", i, j, action[j]);

    // Write action to sample
    sample["Action"][i] = action;
    sample["Policy"]["State Value"][i] = policy[0].stateValue;
    sample["Policy"]["Unbounded Action"][i] = policy[0].unboundedAction;
    sample["Policy"]["Distribution Parameters"][i] = policy[0].distributionParameters;
  }
}

std::vector<float> Continuous::generateTrainingAction(policy_t &curPolicy)
{
  std::vector<float> action(_problem->_actionVectorSize);

  // Creating the action based on the selected policy
  if (_policyDistribution == "Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mean = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = mean + sigma * _normalGenerator->getRandomNumber();
    }
  }

  if (_policyDistribution == "Squashed Normal")
  {
    std::vector<float> unboundedAction(_problem->_actionVectorSize);
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mu = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float scale = _actionScales[i];
      const float shift = _actionShifts[i];

      unboundedAction[i] = mu + sigma * _normalGenerator->getRandomNumber();
      action[i] = (std::tanh(unboundedAction[i]) * scale) + shift;

      // Safety check
      if (action[i] >= _actionUpperBounds[i]) action[i] = _actionUpperBounds[i];
      if (action[i] <= _actionLowerBounds[i]) action[i] = _actionLowerBounds[i];
    }
    curPolicy.unboundedAction = unboundedAction;
  }

  if (_policyDistribution == "Clipped Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mu = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = mu + sigma * _normalGenerator->getRandomNumber();

      if (action[i] >= _actionUpperBounds[i]) action[i] = _actionUpperBounds[i];
      if (action[i] <= _actionLowerBounds[i]) action[i] = _actionLowerBounds[i];
    }
  }

  if (_policyDistribution == "Truncated Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mu = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float alpha = (_actionLowerBounds[i] - mu) / sigma;
      const float beta = (_actionUpperBounds[i] - mu) / sigma;

      // Sampling via naive inverse sampling (not the safest approach due to numerical precision)
      const float u = _uniformGenerator->getRandomNumber();
      const float z = u * normalCDF(beta, 0.f, 1.f) + (1. - u) * normalCDF(alpha, 0.f, 1.f);
      action[i] = mu + M_SQRT2 * ierf(2. * z - 1.) * sigma;

      // Safety check
      if (action[i] >= _actionUpperBounds[i]) action[i] = _actionUpperBounds[i];
      if (action[i] <= _actionLowerBounds[i]) action[i] = _actionLowerBounds[i];
    }
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = ranBetaAlt(_normalGenerator->_range, curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
    }
  }

  return action;
}

std::vector<float> Continuous::generateTestingAction(const policy_t &curPolicy)
{
  std::vector<float> action(_problem->_actionVectorSize);

  if (_policyDistribution == "Normal")
  {
    // Take only the means without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
      action[i] = curPolicy.distributionParameters[i];
  }

  if (_policyDistribution == "Squashed Normal")
  {
    // Take only the transformed means without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mu = curPolicy.distributionParameters[i];
      const float scale = _actionScales[i];
      const float shift = _actionShifts[i];
      action[i] = (std::tanh(mu) * scale) + shift;
    }
  }

  if (_policyDistribution == "Clipped Normal")
  {
    // Take only the modes of the Clipped Normal without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      action[i] = curPolicy.distributionParameters[i];
      // Clip mode to bounds
      if (action[i] >= _actionUpperBounds[i]) action[i] = _actionUpperBounds[i];
      if (action[i] <= _actionLowerBounds[i]) action[i] = _actionLowerBounds[i];
    }
  }

  if (_policyDistribution == "Truncated Normal")
  {
    // Take only the modes of the Truncated Normal noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      action[i] = curPolicy.distributionParameters[i];
      // Clip mode to bounds
      if (action[i] >= _actionUpperBounds[i]) action[i] = _actionUpperBounds[i];
      if (action[i] <= _actionLowerBounds[i]) action[i] = _actionLowerBounds[i];
    }
  }

  if (_policyDistribution == "Beta")
  {
    // Take only the modes without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      action[i] = _actionLowerBounds[i] + 2.0f * _actionScales[i] * curPolicy.distributionParameters[i];
    }
  }

  return action;
}

float Continuous::calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy)
{
  float logpCurPolicy = 0.0f;
  float logpOldPolicy = 0.0f;

  if (_policyDistribution == "Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      logpCurPolicy += normalLogDensity(action[i], curMean, curSigma);
      logpOldPolicy += normalLogDensity(action[i], oldMean, oldSigma);
    }
  }

  if (_policyDistribution == "Squashed Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Importance weight of squashed normal is the importance weight of normal evaluated at unbounded action
      logpCurPolicy += normalLogDensity(oldPolicy.unboundedAction[i], curMu, curSigma);
      logpOldPolicy += normalLogDensity(oldPolicy.unboundedAction[i], oldMu, oldSigma);
    }
  }

  if (_policyDistribution == "Clipped Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      if (action[i] <= _actionLowerBounds[i])
      {
        logpCurPolicy += normalLogCDF(_actionLowerBounds[i], curMu, curSigma);
        logpOldPolicy += normalLogCDF(_actionLowerBounds[i], oldMu, oldSigma);
      }
      else if (_actionUpperBounds[i] <= action[i])
      {
        logpCurPolicy += normalLogCCDF(_actionUpperBounds[i], curMu, curSigma);
        logpOldPolicy += normalLogCCDF(_actionUpperBounds[i], oldMu, oldSigma);
      }
      else
      {
        logpCurPolicy += normalLogDensity(action[i], curMu, curSigma);
        logpOldPolicy += normalLogDensity(action[i], oldMu, oldSigma);
      }
    }
  }

  if (_policyDistribution == "Truncated Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float oldInvSig = 1.f / oldSigma;
      const float curInvSig = 1.f / curSigma;

      const float oldAlpha = (_actionLowerBounds[i] - oldMu) * oldInvSig * M_SQRT1_2;
      const float oldBeta = (_actionUpperBounds[i] - oldMu) * oldInvSig * M_SQRT1_2;

      const float curAlpha = (_actionLowerBounds[i] - curMu) * curInvSig * M_SQRT1_2;
      const float curBeta = (_actionUpperBounds[i] - curMu) * curInvSig * M_SQRT1_2;

      // log of normalization constants
      const float lCq = M_LN2 - safeLogMinus(gsl_sf_log_erfc(-curBeta), gsl_sf_log_erfc(-curAlpha));
      const float lCp = M_LN2 - safeLogMinus(gsl_sf_log_erfc(-oldBeta), gsl_sf_log_erfc(-oldAlpha));

      logpCurPolicy += lCq - std::log(curSigma) - 0.5 * (action[i] - curMu) * (action[i] - curMu) * curInvSig * curInvSig;
      logpOldPolicy += lCp - std::log(oldSigma) - 0.5 * (action[i] - oldMu) * (action[i] - oldMu) * oldInvSig * oldInvSig;
    }
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      logpCurPolicy += betaLogDensityAlt(action[i], curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
      logpOldPolicy += betaLogDensityAlt(action[i], oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
    }
  }

  // Calculating log importance weight
  float logImportanceWeight = logpCurPolicy - logpOldPolicy;

  // Normalizing extreme values to prevent loss of precision
  if (logImportanceWeight > +7.f) logImportanceWeight = +7.f;
  if (logImportanceWeight < -7.f) logImportanceWeight = -7.f;
  if (std::isfinite(logImportanceWeight) == false) KORALI_LOG_ERROR("NaN detected in the calculation of importance weight.\n");

  // Calculating actual importance weight by exp
  const float importanceWeight = std::exp(logImportanceWeight); // TODO: reuse importance weight calculation from updateExperienceReplayMetadata

  return importanceWeight;
}

std::vector<float> Continuous::calculateImportanceWeightGradient(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy, const float importanceWeight)
{
  // Storage for importance weight gradients
  std::vector<float> importanceWeightGradients(_policyParameterCount, 0.);

  if (_policyDistribution == "Normal")
  {
    // ParamsOne are the Means, ParamsTwo are the Sigmas
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Deviation from expAction and current Mean
      const float curActionDif = action[i] - curMean;

      // Inverse Variances
      const float curInvVar = 1.f / (curSigma * curSigma);

      // Gradient with respect to Mean
      importanceWeightGradients[i] = curActionDif * curInvVar;

      // Gradient with respect to Sigma
      importanceWeightGradients[_problem->_actionVectorSize + i] = (curActionDif * curActionDif) * (curInvVar / curSigma) - 1.f / curSigma;
    }

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++) importanceWeightGradients[i] *= importanceWeight;
  }

  if (_policyDistribution == "Squashed Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float unboundedAction = oldPolicy.unboundedAction[i];

      // Deviation from expAction and current Mean
      const float curActionDif = unboundedAction - curMu;

      // Inverse Variance
      const float curInvVar = 1. / (curSigma * curSigma);

      // Gradient with respect to Mean
      importanceWeightGradients[i] = curActionDif * curInvVar;

      // Gradient with respect to Sigma
      importanceWeightGradients[_problem->_actionVectorSize + i] = (curActionDif * curActionDif) * (curInvVar / curSigma) - 1.0f / curSigma;
    }

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
      importanceWeightGradients[i] *= importanceWeight;
  }

  if (_policyDistribution == "Clipped Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float curInvSig = 1.f / curSigma;

      // Deviation from expAction and current Mu
      const float curActionDif = (action[i] - curMu);

      if (action[i] <= _actionLowerBounds[i])
      {
        const float curNormalLogPdfLower = normalLogDensity(_actionLowerBounds[i], curMu, curSigma);
        const float curNormalLogCdfLower = normalLogCDF(_actionLowerBounds[i], curMu, curSigma);
        const float pdfCdfRatio = std::exp(curNormalLogPdfLower - curNormalLogCdfLower);

        // Grad wrt. curMu
        importanceWeightGradients[i] = -pdfCdfRatio;

        // Grad wrt. curSigma
        importanceWeightGradients[_problem->_actionVectorSize + i] = -curActionDif * curInvSig * pdfCdfRatio;
      }
      else if (_actionUpperBounds[i] <= action[i])
      {
        const float curNormalLogPdfUpper = normalLogDensity(_actionUpperBounds[i], curMu, curSigma);
        const float curNormalLogCCdfUpper = normalLogCCDF(_actionUpperBounds[i], curMu, curSigma);
        const float pdfCCdfRatio = std::exp(curNormalLogPdfUpper - curNormalLogCCdfUpper);

        // Grad wrt. curMu
        importanceWeightGradients[i] = pdfCCdfRatio;

        // Grad wrt. curSigma
        importanceWeightGradients[_problem->_actionVectorSize + i] = curActionDif * curInvSig * pdfCCdfRatio;
      }
      else
      {
        // Inverse Variance
        const float curInvSig3 = curInvSig * curInvSig * curInvSig;

        // Grad wrt. curMu
        importanceWeightGradients[i] = curActionDif * curInvSig * curInvSig;

        // Grad wrt. curSigma
        importanceWeightGradients[_problem->_actionVectorSize + i] = curActionDif * curActionDif * curInvSig3 - curInvSig;
      }
    }

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < _policyParameterCount; i++)
      importanceWeightGradients[i] *= importanceWeight;
  }

  if (_policyDistribution == "Truncated Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float curInvSig = 1. / curSigma;
      const float curInvVar = curInvSig * curInvSig;

      // Action differences to mu
      const float curActionDif = action[i] - curMu;

      // Scaled upper and lower bound distances from mu
      const float curAlpha = (_actionLowerBounds[i] - curMu) * curInvSig * M_SQRT1_2;
      const float curBeta = (_actionUpperBounds[i] - curMu) * curInvSig * M_SQRT1_2;

      // log of normalization constantsa
      const float lCq = M_LN2 - safeLogMinus(gsl_sf_log_erfc(-curBeta), gsl_sf_log_erfc(-curAlpha));

      // precomputing log values
      const float lPi2 = 0.5 * std::log(2. * M_PI);
      const float lCurSig = std::log(curSigma);

      // log of normalized gradients of normalization constants
      float ldCqMu = lCq - lPi2 - lCurSig;
      float dCqMu;

      const float curBeta2 = curBeta * curBeta;
      const float curAlpha2 = curAlpha * curAlpha;
      const float eps = 1e-7;

      if (definitelyLessThan(-curBeta2, -curAlpha2, eps))
      {
        ldCqMu += safeLogMinus(-curAlpha2, -curBeta2);
        dCqMu = -std::exp(ldCqMu);
      }
      else if (definitelyLessThan(-curAlpha2, -curBeta2, eps))
      {
        ldCqMu += safeLogMinus(-curBeta2, -curAlpha2);
        dCqMu = std::exp(ldCqMu);
      }
      else
      {
        ldCqMu = -100;
        dCqMu = 0.;
      }

      float dCqSig = std::exp(lCq - lPi2 - 2. * lCurSig - curAlpha2) * (curMu - _actionLowerBounds[i]) + std::exp(lCq - lPi2 - 2. * lCurSig - curBeta2) * (_actionUpperBounds[i] - curMu);

      // Gradient with respect to Mean
      importanceWeightGradients[i] = (curActionDif * curInvVar + dCqMu);
      assert(isfinite(importanceWeightGradients[i]));

      // Gradient with respect to Sigma
      importanceWeightGradients[_problem->_actionVectorSize + i] = curActionDif * curActionDif * curInvVar * curInvSig - curInvSig + dCqSig;
      assert(isfinite(importanceWeightGradients[_problem->_actionVectorSize + i]));
    }

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
    {
      importanceWeightGradients[i] *= importanceWeight;
      assert(isfinite(importanceWeightGradients[i]));
    }
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      float alphaCur;
      float betaCur;
      std::tie(alphaCur, betaCur) = betaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      float alphaOld;
      float betaOld;
      std::tie(alphaOld, betaOld) = betaParamTransformAlt(oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // Log probability of action with old policy params
      const float logpOldPolicy = betaLogDensityAlt(action[i], oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
      const float invpOldPolicy = std::exp(-logpOldPolicy);

      // Variable preparation
      const float Bab = gsl_sf_beta(alphaCur, betaCur);

      const float psiAb = gsl_sf_psi(alphaCur + betaCur);

      const float actionRange = _actionUpperBounds[i] - _actionLowerBounds[i];
      const float logscale = std::log(actionRange);
      const float powscale = std::pow(actionRange, -betaCur - alphaCur + 1.f);
      const float factor = -1.f * std::pow(action[i] - _actionLowerBounds[i], alphaCur - 1.f) * powscale * std::pow(_actionUpperBounds[i] - action[i], betaCur - 1.f) * invpOldPolicy / Bab;

      // Rho Grad wrt alpha and beta
      const float daBab = gsl_sf_psi(alphaCur) - psiAb;
      const float drhoda = ((logscale - std::log(action[i] - _actionLowerBounds[i])) + daBab) * factor;
      const float dbBab = gsl_sf_psi(betaCur) - psiAb;
      const float drhodb = (logscale - std::log(_actionUpperBounds[i] - action[i]) + dbBab) * factor;

      // Derivatives of alpha and beta wrt mu and varc
      float dadmu, dadvarc, dbdmu, dbdvarc;
      std::tie(dadmu, dadvarc, dbdmu, dbdvarc) = derivativesBetaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // Rho Grad wrt mu and varc
      importanceWeightGradients[i] = drhoda * dadmu + drhodb * dbdmu;
      importanceWeightGradients[_problem->_actionVectorSize + i] = drhoda * dadvarc + drhodb * dbdvarc;
    }
  }

  return importanceWeightGradients;
}

std::vector<float> Continuous::calculateKLDivergenceGradient(const policy_t &oldPolicy, const policy_t &curPolicy)
{
  // Storage for KL Divergence Gradients
  std::vector<float> KLDivergenceGradients(_policyParameterCount, 0.0);

  if (_policyDistribution == "Normal" || _policyDistribution == "Squashed Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float curInvSig = 1. / curSigma;
      const float curInvVar = 1. / (curSigma * curSigma);
      const float curInvSig3 = 1. / (curSigma * curSigma * curSigma);
      const float actionDiff = (curMean - oldMean);

      // KL-Gradient with respect to Mean
      KLDivergenceGradients[i] = actionDiff * curInvVar;

      // Contribution to Sigma from Trace
      const float gradTr = -curInvSig3 * oldSigma * oldSigma;

      // Contribution to Sigma from Quadratic term
      const float gradQuad = -(actionDiff * actionDiff) * curInvSig3;

      // Contribution to Sigma from Determinant
      const float gradDet = curInvSig;

      // KL-Gradient with respect to Sigma
      KLDivergenceGradients[_problem->_actionVectorSize + i] = gradTr + gradQuad + gradDet;
    }
  }

  if (_policyDistribution == "Clipped Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Precompute often used constant terms
      const float oldVar = oldSigma * oldSigma;
      const float oldInvSig = 1.f / oldSigma;
      const float curInvSig = 1.f / curSigma;
      const float curInvVar = 1.f / (curSigma * curSigma);
      const float curInvSig3 = 1.f / (curSigma * curSigma * curSigma);
      const float muDif = (oldMu - curMu);

      const float invSqrt2Pi = M_SQRT1_2 * std::sqrt(M_1_PI);

      const float oldAdjustedLb = (_actionLowerBounds[i] - oldMu) * oldInvSig;
      const float oldAdjustedUb = (_actionUpperBounds[i] - oldMu) * oldInvSig;

      const float curAdjustedLb = (_actionLowerBounds[i] - curMu) * curInvSig;
      const float curAdjustedUb = (_actionUpperBounds[i] - curMu) * curInvSig;

      const float erfLb = std::erf(M_SQRT1_2 * oldAdjustedLb);
      const float erfUb = std::erf(M_SQRT1_2 * oldAdjustedUb);

      const float expLb = std::exp(-0.5f * oldAdjustedLb * oldAdjustedLb);
      const float expUb = std::exp(-0.5f * oldAdjustedUb * oldAdjustedUb);

      const float cdfRatiosA = std::exp(normalLogCDF(_actionLowerBounds[i], oldMu, oldSigma) + normalLogDensity(_actionLowerBounds[i], curMu, curSigma) - normalLogCDF(_actionLowerBounds[i], curMu, curSigma));
      const float ccdfRatiosB = std::exp(normalLogCCDF(_actionUpperBounds[i], oldMu, oldSigma) + normalLogDensity(_actionUpperBounds[i], curMu, curSigma) - normalLogCCDF(_actionUpperBounds[i], curMu, curSigma));

      // KL-Gradient with respect to Mean
      KLDivergenceGradients[i] = cdfRatiosA;
      KLDivergenceGradients[i] -= 0.5f * muDif * curInvVar * (erfUb - erfLb);
      KLDivergenceGradients[i] += invSqrt2Pi * oldSigma * curInvVar * (expUb - expLb);
      KLDivergenceGradients[i] -= ccdfRatiosB;

      // KL-Gradient with respect to Sigma
      KLDivergenceGradients[_problem->_actionVectorSize + i] = curAdjustedLb * cdfRatiosA;
      KLDivergenceGradients[_problem->_actionVectorSize + i] += 0.5f * (curInvSig - muDif * muDif * curInvSig3 - oldVar * curInvSig3) * (erfUb - erfLb);
      KLDivergenceGradients[_problem->_actionVectorSize + i] += invSqrt2Pi * curInvSig3 * (oldVar * oldAdjustedUb + 2.f * oldSigma * muDif) * expUb;
      KLDivergenceGradients[_problem->_actionVectorSize + i] -= invSqrt2Pi * curInvSig3 * (oldVar * oldAdjustedLb + 2.f * oldSigma * muDif) * expLb;
      KLDivergenceGradients[_problem->_actionVectorSize + i] -= curAdjustedUb * ccdfRatiosB;
    }
  }

  if (_policyDistribution == "Truncated Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Precompute often used constant terms
      const float oldVar = oldSigma * oldSigma;
      const float oldInvSig = 1.f / oldSigma;

      const float curInvSig = 1.f / curSigma;
      const float curInvVar = curInvSig * curInvSig;
      const float curInvSig3 = curInvSig * curInvVar;
      const float muDif = (oldMu - curMu);

      // old scaled upper and lower bound distances from mu
      const float oldAlpha = (_actionLowerBounds[i] - oldMu) * oldInvSig * M_SQRT1_2;
      const float oldBeta = (_actionUpperBounds[i] - oldMu) * oldInvSig * M_SQRT1_2;

      // current scaled upper and lower bound distances from mu
      const float curAlpha = (_actionLowerBounds[i] - curMu) * curInvSig * M_SQRT1_2;
      const float curBeta = (_actionUpperBounds[i] - curMu) * curInvSig * M_SQRT1_2;

      // log of normalization constantsa
      const float lCq = M_LN2 - safeLogMinus(gsl_sf_log_erfc(-curBeta), gsl_sf_log_erfc(-curAlpha));
      const float lCp = M_LN2 - safeLogMinus(gsl_sf_log_erfc(-oldBeta), gsl_sf_log_erfc(-oldAlpha));

      // precomputing log values
      const float lPi2 = 0.5 * std::log(2. * M_PI);
      const float lCurSig = std::log(curSigma);
      const float lOldSig = std::log(oldSigma);

      // log of normalized gradients of normalization constants
      float ldCqMu = lCq - lPi2 - lCurSig;
      float dCqMu;

      float lCps = lCp - 0.5 * lPi2 + lOldSig - lCurSig;
      float Cps;

      const float curBeta2 = curBeta * curBeta;
      const float curAlpha2 = curAlpha * curAlpha;
      const float eps = 1e-6;

      if (definitelyLessThan(-curBeta2, -curAlpha2, eps))
      {
        const float logDif = safeLogMinus(-curAlpha2, -curBeta2);
        ldCqMu += logDif;
        dCqMu = -std::exp(ldCqMu);
        lCps += logDif;
        Cps = -std::exp(lCps);
      }
      else if (definitelyLessThan(-curAlpha2, -curBeta2, eps))
      {
        const float logDif = safeLogMinus(-curBeta2, -curAlpha2);
        ldCqMu += logDif;
        dCqMu = std::exp(ldCqMu);
        lCps += logDif;
        Cps = std::exp(lCps);
      }
      else
      {
        ldCqMu = -100;
        dCqMu = 0.;
        Cps = 0.;
      }

      float dCqSig = std::exp(lCq - lPi2 - 2. * lCurSig - curAlpha2) * (curMu - _actionLowerBounds[i]) + std::exp(lCq - lPi2 - 2. * lCurSig - curBeta2) * (_actionUpperBounds[i] - curMu);

      // KL-Gradient with respect to Mu
      KLDivergenceGradients[i] = -dCqMu - muDif * curInvVar + Cps;

      // Precompute some terms
      const float sb = oldSigma * curInvSig3 * (_actionUpperBounds[i] + oldMu - 2. * curMu);
      const float sa = oldSigma * curInvSig3 * (_actionLowerBounds[i] + oldMu - 2. * curMu);

      const float lCpb = lCp - oldBeta * oldBeta - lPi2;
      const float lCpa = lCp - oldAlpha * oldAlpha - lPi2;

      const float Cpb = std::exp(lCpb);
      const float Cpa = std::exp(lCpa);

      // KL-Gradient with respect to Sigma
      KLDivergenceGradients[_problem->_actionVectorSize + i] = -muDif * muDif * curInvSig3 + curInvSig - dCqSig - oldVar * curInvSig3 + Cpb * sb - Cpa * sa;
    }
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      float alphaCur;
      float betaCur;
      std::tie(alphaCur, betaCur) = betaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      float alphaOld;
      float betaOld;
      std::tie(alphaOld, betaOld) = betaParamTransformAlt(oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // Constants involving psi function
      const float psiAbCur = gsl_sf_psi(alphaCur + betaCur);
      const float psiAbOld = gsl_sf_psi(alphaOld + betaOld);

      const float actionRange = _actionUpperBounds[i] - _actionLowerBounds[i];

      // KL Grad wrt alpha
      const float dklda = (gsl_sf_psi(alphaCur) - psiAbCur - gsl_sf_psi(alphaOld) - psiAbOld) / actionRange;

      // KL Grad wrt beta
      const float dkldb = (gsl_sf_psi(betaCur) - psiAbCur - gsl_sf_psi(betaOld) - psiAbOld) / actionRange;

      // Derivatives of alpha and beta wrt mu and varc
      float dadmu, dadvarc, dbdmu, dbdvarc;
      std::tie(dadmu, dadvarc, dbdmu, dbdvarc) = derivativesBetaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // KL Grad wrt mu and varc
      KLDivergenceGradients[i] = dklda * dadmu + dkldb * dbdmu;
      KLDivergenceGradients[_problem->_actionVectorSize + i] = dklda * dadvarc + dkldb * dbdvarc;
    }
  }

  return KLDivergenceGradients;
}

void Continuous::setConfiguration(knlohmann::json& js) 
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

 if (isDefined(js, "Action Shifts"))
 {
 try { _actionShifts = js["Action Shifts"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Action Shifts']\n%s", e.what()); } 
   eraseValue(js, "Action Shifts");
 }

 if (isDefined(js, "Action Scales"))
 {
 try { _actionScales = js["Action Scales"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Action Scales']\n%s", e.what()); } 
   eraseValue(js, "Action Scales");
 }

 if (isDefined(js, "Policy", "Parameter Transformation Masks"))
 {
 try { _policyParameterTransformationMasks = js["Policy"]["Parameter Transformation Masks"].get<std::vector<std::string>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Transformation Masks']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Transformation Masks");
 }

 if (isDefined(js, "Policy", "Parameter Scaling"))
 {
 try { _policyParameterScaling = js["Policy"]["Parameter Scaling"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Scaling']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Scaling");
 }

 if (isDefined(js, "Policy", "Parameter Shifting"))
 {
 try { _policyParameterShifting = js["Policy"]["Parameter Shifting"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Shifting']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Shifting");
 }

 if (isDefined(js, "Policy", "Distribution"))
 {
 try { _policyDistribution = js["Policy"]["Distribution"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Distribution']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_policyDistribution == "Normal") validOption = true; 
 if (_policyDistribution == "Squashed Normal") validOption = true; 
 if (_policyDistribution == "Clipped Normal") validOption = true; 
 if (_policyDistribution == "Truncated Normal") validOption = true; 
 if (_policyDistribution == "Beta") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Policy']['Distribution'] required by continuous.\n", _policyDistribution.c_str()); 
}
   eraseValue(js, "Policy", "Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Policy']['Distribution'] required by continuous.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Agent::setConfiguration(js);
 _type = "agent/continuous";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: continuous: \n%s\n", js.dump(2).c_str());
} 

void Continuous::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Policy"]["Distribution"] = _policyDistribution;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
   js["Action Shifts"] = _actionShifts;
   js["Action Scales"] = _actionScales;
   js["Policy"]["Parameter Transformation Masks"] = _policyParameterTransformationMasks;
   js["Policy"]["Parameter Scaling"] = _policyParameterScaling;
   js["Policy"]["Parameter Shifting"] = _policyParameterShifting;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Agent::getConfiguration(js);
} 

void Continuous::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Normal Generator\": {\"Name\": \"Agent / Continuous / Normal Generator\", \"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Policy\": {\"Distribution\": \"Normal\"}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Agent::applyModuleDefaults(js);
} 

void Continuous::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Agent::applyVariableDefaults();
} 

bool Continuous::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Agent::checkTermination();
 return hasFinished;
}

;

} //agent
} //solver
} //korali
;
