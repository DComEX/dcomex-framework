#include "engine.hpp"
#include "modules/solver/agent/discrete/dVRACER/dVRACER.hpp"
#ifdef _OPENMP
  #include "omp.h"
#endif
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
namespace discrete
{
;

void dVRACER::initializeAgent()
{
  // Initializing common discrete agent configuration
  Discrete::initializeAgent();

  // Init statistics
  _statisticsAverageInverseTemperature = 0.;
  _statisticsAverageActionUnlikeability = 0.;

  /*********************************************************************
   * Initializing Critic/Policy Neural Network Optimization Experiment
   *********************************************************************/
  _criticPolicyLearner.resize(_problem->_policiesPerEnvironment);
  _criticPolicyExperiment.resize(_problem->_policiesPerEnvironment);
  _criticPolicyProblem.resize(_problem->_policiesPerEnvironment);

  _effectiveMinibatchSize = _miniBatchSize * _problem->_agentsPerEnvironment;

  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
  {
    _criticPolicyExperiment[p]["Random Seed"] = _k->_randomSeed;

    _criticPolicyExperiment[p]["Problem"]["Type"] = "Supervised Learning";
    _criticPolicyExperiment[p]["Problem"]["Max Timesteps"] = _timeSequenceLength;
    _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _effectiveMinibatchSize;
    _criticPolicyExperiment[p]["Problem"]["Testing Batch Size"] = 1;
    _criticPolicyExperiment[p]["Problem"]["Input"]["Size"] = _problem->_stateVectorSize;
    _criticPolicyExperiment[p]["Problem"]["Solution"]["Size"] = 1 + _policyParameterCount; // The value function, action q values, and inverse temperatur

    _criticPolicyExperiment[p]["Solver"]["Type"] = "DeepSupervisor";
    _criticPolicyExperiment[p]["Solver"]["Mode"] = "Training";
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
    _criticPolicyExperiment[p]["Solver"]["Learning Rate"] = _currentLearningRate;
    _criticPolicyExperiment[p]["Solver"]["Loss Function"] = "Direct Gradient";
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Engine"] = _neuralNetworkEngine;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
    _criticPolicyExperiment[p]["Solver"]["Output Weights Scaling"] = 0.001;

    // No transformations for the state value output
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Identity";
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 1.0f;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = 0.0f;

    // No transformation for the q values
    for (size_t i = 0; i < _problem->_actionCount; ++i)
    {
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][i + 1] = "Identity";
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][i + 1] = 1.0f;
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][i + 1] = 0.0f;
    }

    // Transofrmation for the inverse temperature
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][1 + _problem->_actionCount] = "Softplus"; // x = 0.5 * (x + std::sqrt(1. + x * x));
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][1 + _problem->_actionCount] = 1.0f;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][1 + _problem->_actionCount] = _initialInverseTemperature - 0.5;

    // Running initialization to verify that the configuration is correct
    _criticPolicyExperiment[p].setEngine(_k->_engine);
    _criticPolicyExperiment[p].initialize();
    _criticPolicyProblem[p] = dynamic_cast<problem::SupervisedLearning *>(_criticPolicyExperiment[p]._problem);
    _criticPolicyLearner[p] = dynamic_cast<solver::DeepSupervisor *>(_criticPolicyExperiment[p]._solver);

    // Preallocating space in the underlying supervised problem's input and solution data structures (for performance, we don't reinitialize it every time)
    _criticPolicyProblem[p]->_inputData.resize(_effectiveMinibatchSize);
    _criticPolicyProblem[p]->_solutionData.resize(_effectiveMinibatchSize);
  }
}

void dVRACER::trainPolicy()
{
  // Obtaining Minibatch experience ids
  const auto miniBatch = generateMiniBatch();

  // Gathering state sequences for selected minibatch
  const auto stateSequenceBatch = getMiniBatchStateSequence(miniBatch);

  // Buffer for policy info to update experience metadata
  std::vector<policy_t> policyInfoUpdateMetadata(miniBatch.size());

  // Get number of policies
  const size_t numPolicies = _problem->_policiesPerEnvironment;

  // Update all policies using all experiences
  for (size_t p = 0; p < numPolicies; p++)
  {
    // Fill policyInfo with behavioral policy (access to availableActions)
    std::vector<policy_t> policyInfo = getPolicyInfo(miniBatch);

    // Forward NN
    runPolicy(stateSequenceBatch, policyInfo, p);

    // Using policy information to update experience's metadata
    updateExperienceMetadata(miniBatch, policyInfo);

    // Now calculating policy gradients
    calculatePolicyGradients(miniBatch, p);

    // Updating learning rate for critic/policy learner guided by REFER
    _criticPolicyLearner[p]->_learningRate = _currentLearningRate;

    // Now applying gradients to update policy NN
    _criticPolicyLearner[p]->runGeneration();

    // Store policyData for agent p for later update of metadata
    if (numPolicies > 1)
      for (size_t b = 0; b < _miniBatchSize; b++)
        policyInfoUpdateMetadata[b * numPolicies + p] = policyInfo[b * numPolicies + p];
  }

  // Correct experience metadata
  if (numPolicies > 1)
    updateExperienceMetadata(miniBatch, policyInfoUpdateMetadata);
}

void dVRACER::calculatePolicyGradients(const std::vector<std::pair<size_t, size_t>> &miniBatch, const size_t policyIdx)
{
  // Init statistics
  _statisticsAverageInverseTemperature = 0.;
  _statisticsAverageActionUnlikeability = 0.;

  const size_t miniBatchSize = miniBatch.size();
  const size_t numAgents = _problem->_agentsPerEnvironment;

#pragma omp parallel for schedule(guided, numAgents) reduction(+ \
                                                               : _statisticsAverageInverseTemperature, _statisticsAverageActionUnlikeability)
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    const size_t expId = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Getting old and current policy
    const auto &expPolicy = _expPolicyBuffer[expId][agentId];
    const auto &curPolicy = _curPolicyBuffer[expId][agentId];

    // Getting state-value and estimator
    const auto &stateValue = _stateValueBufferContiguous[expId * numAgents + agentId];
    const auto &expVtbc = _retraceValueBufferContiguous[expId * numAgents + agentId];

    // Storage for the update gradient
    std::vector<float> gradientLoss(1 + _policyParameterCount, 0.0f);

    // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
    gradientLoss[0] = expVtbc - stateValue;

    // Gradient has to be divided by Number of Agents in Cooperative models
    if (_multiAgentRelationship == "Cooperation")
      gradientLoss[0] /= numAgents;

    // Compute policy gradient only if inside trust region
    if (_isOnPolicyBuffer[expId][agentId])
    {
      // Qret for terminal state is just reward
      float Qret = getScaledReward(_rewardBufferContiguous[expId * numAgents + agentId]);

      // If experience is non-terminal, add Vtbc
      if (_terminationBuffer[expId] == e_nonTerminal)
      {
        const float nextExpVtbc = _retraceValueBufferContiguous[(expId + 1) * numAgents + agentId];

        Qret += _discountFactor * nextExpVtbc;
      }

      // If experience is truncated, add truncated state value
      if (_terminationBuffer[expId] == e_truncated)
      {
        const float nextExpVtbc = _truncatedStateValueBuffer[expId][agentId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // Compute Off-Policy Objective (eq. 5)
      float lossOffPolicy = Qret - stateValue;

      // Compute Policy Gradient wrt Params
      auto polGrad = calculateImportanceWeightGradient(curPolicy, expPolicy);

      // If multi-agent correlation, multiply with additional factor
      if (_multiAgentCorrelation)
      {
        float correlationFactor = _productImportanceWeightBuffer[expId] / _importanceWeightBuffer[expId][agentId];
        for (size_t i = 0; i < polGrad.size(); i++)
          polGrad[i] *= correlationFactor;
      }

      // Set Gradient of Loss wrt Params
      for (size_t i = 0; i < _policyParameterCount; i++)
      {
        // '-' because the optimizer is maximizing
        gradientLoss[1 + i] = _experienceReplayOffPolicyREFERCurrentBeta[agentId] * lossOffPolicy * polGrad[i];
      }
    }

    // Compute derivative of KL divergence
    auto klGrad = calculateKLDivergenceGradient(expPolicy, curPolicy);

    // Compute factor for KL penalization
    const float klGradMultiplier = -(1.0f - _experienceReplayOffPolicyREFERCurrentBeta[agentId]);

    for (size_t i = 0; i < _policyParameterCount; i++)
    {
      gradientLoss[1 + i] += klGradMultiplier * klGrad[i];

      if (std::isfinite(gradientLoss[1 + i]) == false)
        KORALI_LOG_ERROR("Gradient loss returned an invalid value: %f\n", gradientLoss[i]);
    }

    // Set Gradient of Loss as Solution
    _criticPolicyProblem[policyIdx]->_solutionData[b] = gradientLoss;

    // Update statistics
    _statisticsAverageInverseTemperature += (curPolicy.distributionParameters[_problem->_actionCount] / (float)_problem->_policiesPerEnvironment);

    float unlikeability = 1.0;
    for (size_t i = 0; i < _problem->_actionCount; ++i)
      unlikeability -= curPolicy.actionProbabilities[i] * curPolicy.actionProbabilities[i];
    _statisticsAverageActionUnlikeability += (unlikeability / (float)_problem->_policiesPerEnvironment);
  }

  // Compute statistics
  _statisticsAverageInverseTemperature /= (float)miniBatchSize;
  _statisticsAverageActionUnlikeability /= (float)miniBatchSize;
}

float dVRACER::calculateStateValue(const std::vector<std::vector<float>> &stateSequence, size_t policyIdx)
{
  // Forward the neural network for this state to get the state value
  const auto evaluation = _criticPolicyLearner[policyIdx]->getEvaluation({stateSequence});
  return evaluation[0][0];
}

void dVRACER::runPolicy(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &policyInfo, size_t policyIdx)
{
  // Getting batch size
  size_t batchSize = stateSequenceBatch.size();

  // Forward neural network
  const auto evaluation = _criticPolicyLearner[policyIdx]->getEvaluation(stateSequenceBatch);

// Update policy info
#pragma omp parallel for
  for (size_t b = 0; b < batchSize; b++)
  {
    // Getting state value
    policyInfo[b].stateValue = evaluation[b][0];

    // Get the inverse of the temperature for the softmax distribution
    const float invTemperature = evaluation[b][_policyParameterCount];

    // Storage for Q(s,a_i) and max_{a_i} Q(s,a_i)
    std::vector<float> qValAndInvTemp(_policyParameterCount);
    float maxq = -korali::Inf;

    // Get Q(s,a_i) and max_{a_i} Q(s,a_i)
    for (size_t i = 0; i < _problem->_actionCount; i++)
    {
      // Assign Q(s,a_i)
      qValAndInvTemp[i] = evaluation[b][1 + i];

      // Update max_{a_i} Q(s,a_i)
      if (qValAndInvTemp[i] > maxq && policyInfo[b].availableActions[i] == 1)
        maxq = qValAndInvTemp[i];
    }

    // Storage for action probabilities
    std::vector<float> pActions(_problem->_actionCount);

    // Storage for the normalization factor Sum_i(e^Q(s,a_i)/e^maxq)
    float sumExpQVal = 0.0;

    for (size_t i = 0; i < _problem->_actionCount; i++)
    {
      // Computing e^(beta(Q(s,a_i) - maxq))
      const float expCurQVal = policyInfo[0].availableActions[i] == 0 ? 0.0f : std::exp(invTemperature * (qValAndInvTemp[i] - maxq));

      // Computing Sum_i(e^Q(s,a_i)/e^maxq)
      sumExpQVal += expCurQVal;

      // Storing partial value of the probability of the action
      pActions[i] = expCurQVal;
    }

    // Calculating inverse of Sum_i(e^Q(s,a_i))
    const float invSumExpQVal = 1.0f / sumExpQVal;

    // Normalizing action probabilities
    for (size_t i = 0; i < _problem->_actionCount; i++)
      pActions[i] *= invSumExpQVal;

    // Set inverse temperature parameter
    qValAndInvTemp[_problem->_actionCount] = invTemperature;

    // Storing the action probabilities into the policy
    policyInfo[b].actionProbabilities = pActions;
    policyInfo[b].distributionParameters = qValAndInvTemp;
  }
}

std::vector<policy_t> dVRACER::getPolicyInfo(const std::vector<std::pair<size_t, size_t>> &miniBatch) const
{
  // Getting mini batch size
  const size_t miniBatchSize = miniBatch.size();

  // Allocating policy sequence vector
  std::vector<policy_t> policyInfo(miniBatchSize);

#pragma omp parallel for schedule(guided, _problem->_agentsPerEnvironment)
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting current expId
    const size_t expId = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Filling policy information
    policyInfo[b] = _expPolicyBuffer[expId][agentId];
  }

  return policyInfo;
}

knlohmann::json dVRACER::getPolicy()
{
  knlohmann::json hyperparameters;
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    hyperparameters["Policy Hyperparameters"][p] = _criticPolicyLearner[p]->getHyperparameters();
  return hyperparameters;
}

void dVRACER::setPolicy(const knlohmann::json &hyperparameters)
{
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->_neuralNetwork->setHyperparameters(hyperparameters[p].get<std::vector<float>>());
}

void dVRACER::printInformation()
{
  _k->_logger->logInfo("Normal", " + [dVRACER] Policy Learning Rate: %.3e\n", _currentLearningRate);
  _k->_logger->logInfo("Normal", " + [dVRACER] Average Inverse Temperature: %.3e\n", _statisticsAverageInverseTemperature);
  _k->_logger->logInfo("Normal", " + [dVRACER] Average Action Unlikeability: %.3e\n", _statisticsAverageActionUnlikeability);
}

void dVRACER::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Statistics", "Average Inverse Temperature"))
 {
 try { _statisticsAverageInverseTemperature = js["Statistics"]["Average Inverse Temperature"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ dVRACER ] \n + Key:    ['Statistics']['Average Inverse Temperature']\n%s", e.what()); } 
   eraseValue(js, "Statistics", "Average Inverse Temperature");
 }

 if (isDefined(js, "Statistics", "Average Action Unlikeability"))
 {
 try { _statisticsAverageActionUnlikeability = js["Statistics"]["Average Action Unlikeability"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ dVRACER ] \n + Key:    ['Statistics']['Average Action Unlikeability']\n%s", e.what()); } 
   eraseValue(js, "Statistics", "Average Action Unlikeability");
 }

 if (isDefined(js, "Initial Inverse Temperature"))
 {
 try { _initialInverseTemperature = js["Initial Inverse Temperature"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ dVRACER ] \n + Key:    ['Initial Inverse Temperature']\n%s", e.what()); } 
   eraseValue(js, "Initial Inverse Temperature");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Inverse Temperature'] required by dVRACER.\n"); 

 Discrete::setConfiguration(js);
 _type = "agent/discrete/dVRACER";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: dVRACER: \n%s\n", js.dump(2).c_str());
} 

void dVRACER::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Initial Inverse Temperature"] = _initialInverseTemperature;
   js["Statistics"]["Average Inverse Temperature"] = _statisticsAverageInverseTemperature;
   js["Statistics"]["Average Action Unlikeability"] = _statisticsAverageActionUnlikeability;
 Discrete::getConfiguration(js);
} 

void dVRACER::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Initial Inverse Temperature\": 1.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Discrete::applyModuleDefaults(js);
} 

void dVRACER::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Discrete::applyVariableDefaults();
} 

bool dVRACER::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Discrete::checkTermination();
 return hasFinished;
}

;

} //discrete
} //agent
} //solver
} //korali
;
