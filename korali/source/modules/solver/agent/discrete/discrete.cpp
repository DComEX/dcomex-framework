#include "engine.hpp"
#include "modules/solver/agent/discrete/discrete.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
;

void Discrete::initializeAgent()
{
  // Getting discrete problem pointer
  _problem = dynamic_cast<problem::reinforcementLearning::Discrete *>(_k->_problem);

  _problem->_actionCount = _problem->_possibleActions.size();
  _policyParameterCount = _problem->_actionCount + 1; // q values and inverseTemperature
}

void Discrete::getAction(korali::Sample &sample)
{
  // Get action for all the agents in the environment
  for (size_t i = 0; i < _problem->_agentsPerEnvironment; i++)
  {
    // Getting current state
    auto state = sample["State"][i].get<std::vector<float>>();

    // Adding state to the state time sequence
    _stateTimeSequence[i].add(state);

    // Preparing storage for policy information and flag available actions if provided
    std::vector<policy_t> policy(1);
    policy[0].availableActions = sample["Available Actions"][i].get<std::vector<size_t>>();

    // Getting the probability of the actions given by the agent's policy
    if (_problem->_policiesPerEnvironment == 1)
      runPolicy({_stateTimeSequence[i].getVector()}, policy);
    else
      runPolicy({_stateTimeSequence[i].getVector()}, policy, i);

    const auto &qValAndInvTemp = policy[0].distributionParameters;
    const auto &pActions = policy[0].actionProbabilities;

    // Storage for the action index to use
    size_t actionIdx = 0;

    /*****************************************************************************
     * During training, we follow the Epsilon-greedy strategy. Choose, given a
     * probability (pEpsilon), one from the following:
     *  - Uniformly random action among all possible actions
     *  - Sample action guided by the policy's probability distribution
     ****************************************************************************/

    if (sample["Mode"] == "Training")
    {
      // Producing random  number for the selection of an available action
      const float x = _uniformGenerator->getRandomNumber();

      // Categorical action sampled from action probabilites (from ACER paper [Wang2017])
      float curSum = 0.0;
      for (actionIdx = 0; actionIdx < _problem->_actionCount - 1; actionIdx++)
      {
        curSum += pActions[actionIdx];
        if (x < curSum) break;
      }

      // Treat rounding errors and choose action with largest pValue
      if (policy[0].availableActions.size() > 0 && policy[0].availableActions[actionIdx] == 0)
        actionIdx = std::distance(pActions.begin(), std::max_element(pActions.begin(), pActions.end()));

      // NOTE: In original DQN paper [Minh2015] we choose max
      // actionIdx = std::distance(pActions.begin(), std::max_element(pActions.begin(), pActions.end()));
    }

    /*****************************************************************************
     * During testing, we just select the action with the largest probability
     * given by the policy.
     ****************************************************************************/

    // Finding the best action index from the probabilities
    if (sample["Mode"] == "Testing")
      actionIdx = std::distance(pActions.begin(), std::max_element(pActions.begin(), pActions.end()));

    /*****************************************************************************
     * Storing the action itself
     ****************************************************************************/

    // Storing action itself, its idx, and probabilities
    sample["Action"][i] = _problem->_possibleActions[actionIdx];
    sample["Policy"]["State Value"][i] = policy[0].stateValue;
    sample["Policy"]["Action Index"][i] = actionIdx;
    sample["Policy"]["Available Actions"][i] = policy[0].availableActions;
    sample["Policy"]["Action Probabilities"][i] = pActions;
    sample["Policy"]["Distribution Parameters"][i] = qValAndInvTemp;
  }
}

float Discrete::calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy)
{
  const auto oldActionIdx = oldPolicy.actionIndex;
  const auto pCurPolicy = curPolicy.actionProbabilities[oldActionIdx];
  const auto pOldPolicy = oldPolicy.actionProbabilities[oldActionIdx];

  // Now calculating importance weight for the old s,a experience
  float constexpr epsilon = 0.00000001f;
  float importanceWeight = pCurPolicy / (pOldPolicy + epsilon);

  // Safety checks
  if (importanceWeight > 1024.0f) importanceWeight = 1024.0f;
  if (importanceWeight < -1024.0f) importanceWeight = -1024.0f;

  return importanceWeight;
}

std::vector<float> Discrete::calculateImportanceWeightGradient(const policy_t &curPolicy, const policy_t &oldPolicy)
{
  std::vector<float> grad(_problem->_actionCount + 1, 0.0);

  const float invTemperature = curPolicy.distributionParameters[_problem->_actionCount];
  const auto &curDistParams = curPolicy.distributionParameters;

  const size_t oldActionIdx = oldPolicy.actionIndex;
  const auto pCurPolicy = curPolicy.actionProbabilities[oldActionIdx];
  const auto pOldPolicy = oldPolicy.actionProbabilities[oldActionIdx];

  // Now calculating importance weight for the old s,a experience
  float constexpr epsilon = 0.00000001f;
  float importanceWeight = pCurPolicy / (pOldPolicy + epsilon);

  // Safety checks
  if (importanceWeight > 1024.0f) importanceWeight = 1024.0f;
  if (importanceWeight < -1024.0f) importanceWeight = -1024.0f;

  float qpSum = 0.;
  // calculate gradient of importance weight wrt. pvals
  for (size_t i = 0; i < _problem->_actionCount; i++)
  {
    if (i == oldActionIdx)
      grad[i] = importanceWeight * (1. - curPolicy.actionProbabilities[i]) * invTemperature;
    else
      grad[i] = -importanceWeight * curPolicy.actionProbabilities[i] * invTemperature;

    qpSum += curDistParams[i] * curPolicy.actionProbabilities[i];
  }

  // calculate gradient of importance weight wrt. inverse temperature
  grad[_problem->_actionCount] = importanceWeight * (curDistParams[oldActionIdx] - qpSum);

  return grad;
}

std::vector<float> Discrete::calculateKLDivergenceGradient(const policy_t &oldPolicy, const policy_t &curPolicy)
{
  const float invTemperature = curPolicy.distributionParameters[_problem->_actionCount];
  const auto &curDistParams = curPolicy.distributionParameters;

  std::vector<float> klGrad(_problem->_actionCount + 1, 0.0);

  // Gradient wrt NN output i (qvalue i)
  for (size_t i = 0; i < _problem->_actionCount; ++i)
  {
    // Iterate over all pvalues
    for (size_t j = 0; j < _problem->_actionCount; ++j)
    {
      if (i == j)
        klGrad[i] -= invTemperature * oldPolicy.actionProbabilities[j] * (1.0 - curPolicy.actionProbabilities[i]);
      else
        klGrad[i] += invTemperature * oldPolicy.actionProbabilities[j] * curPolicy.actionProbabilities[i];
    }
  }

  float qpSum = 0.;
  for (size_t j = 0; j < _problem->_actionCount; ++j)
    qpSum += curDistParams[j] * curPolicy.actionProbabilities[j];

  // Gradient wrt inverse temperature parameter
  for (size_t j = 0; j < _problem->_actionCount; ++j)
    klGrad[_problem->_actionCount] -= oldPolicy.actionProbabilities[j] * (curDistParams[j] - qpSum);

  return klGrad;
}

void Discrete::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Agent::setConfiguration(js);
 _type = "agent/discrete";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: discrete: \n%s\n", js.dump(2).c_str());
} 

void Discrete::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Agent::getConfiguration(js);
} 

void Discrete::applyModuleDefaults(knlohmann::json& js) 
{

 Agent::applyModuleDefaults(js);
} 

void Discrete::applyVariableDefaults() 
{

 Agent::applyVariableDefaults();
} 

bool Discrete::checkTermination()
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
