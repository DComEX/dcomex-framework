#include "auxiliar/fs.hpp"
#include "engine.hpp"
#include "modules/solver/agent/agent.hpp"
#include "sample/sample.hpp"
#include <chrono>

namespace korali
{
namespace solver
{
;

#pragma omp declare reduction(vec_int_plus        \
                              : std::vector <int> \
                              : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus <int>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void Agent::initialize()
{
  _variableCount = _k->_variables.size();

  // Getting problem pointer
  _problem = dynamic_cast<problem::ReinforcementLearning *>(_k->_problem);

  // Getting number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Formatting reward history for each agent
  _trainingReturnHistory.resize(numAgents);
  _trainingDiscountedReturnHistory.resize(numAgents);

  // Allocating and obtaining action bounds information
  _actionLowerBounds.resize(_problem->_actionVectorSize);
  _actionUpperBounds.resize(_problem->_actionVectorSize);

  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    auto varIdx = _problem->_actionVectorIndexes[i];
    _actionLowerBounds[i] = _k->_variables[varIdx]->_lowerBound;
    _actionUpperBounds[i] = _k->_variables[varIdx]->_upperBound;

    if (_actionUpperBounds[i] - _actionLowerBounds[i] <= 0.0) KORALI_LOG_ERROR("Upper (%f) and Lower Bound (%f) of action variable %lu invalid.\n", _actionUpperBounds[i], _actionLowerBounds[i], i);
  }

  if (_episodesPerGeneration < 1)
    KORALI_LOG_ERROR("Episodes Per Generation must be larger equal 1 (is %zu)", _episodesPerGeneration);

  // Initializing selected policy
  initializeAgent();

  // If not set, using heurisitc for maximum size
  if (_experienceReplayMaximumSize == 0)
    _experienceReplayMaximumSize = std::pow(2, 14) * std::sqrt(_problem->_stateVectorSize + _problem->_actionVectorSize);

  // If not set, filling ER before learning
  if (_experienceReplayStartSize == 0)
    _experienceReplayStartSize = _experienceReplayMaximumSize;

  // Initialize current beta for all agents
  _experienceReplayOffPolicyREFERCurrentBeta = std::vector<float>(numAgents, _experienceReplayOffPolicyREFERBeta);

  //  Pre-allocating space for the experience replay memory
  _stateBuffer.resize(_experienceReplayMaximumSize);
  _actionBuffer.resize(_experienceReplayMaximumSize);
  _retraceValueBufferContiguous.resize(_experienceReplayMaximumSize * numAgents);
  _rewardBufferContiguous.resize(_experienceReplayMaximumSize * numAgents);
  _stateValueBufferContiguous.resize(_experienceReplayMaximumSize * numAgents);
  _importanceWeightBuffer.resize(_experienceReplayMaximumSize);
  _truncatedImportanceWeightBufferContiguous.resize(_experienceReplayMaximumSize * numAgents);
  _productImportanceWeightBuffer.resize(_experienceReplayMaximumSize);
  _truncatedStateValueBuffer.resize(_experienceReplayMaximumSize);
  _truncatedStateBuffer.resize(_experienceReplayMaximumSize);
  _terminationBuffer.resize(_experienceReplayMaximumSize);
  _expPolicyBuffer.resize(_experienceReplayMaximumSize);
  _curPolicyBuffer.resize(_experienceReplayMaximumSize);
  _isOnPolicyBuffer.resize(_experienceReplayMaximumSize);
  _episodePosBuffer.resize(_experienceReplayMaximumSize);
  _episodeIdBuffer.resize(_experienceReplayMaximumSize);

  //  Pre-allocating space for state time sequence
  _stateTimeSequence.resize(numAgents);
  for (size_t a = 0; a < numAgents; ++a)
    _stateTimeSequence[a].resize(_timeSequenceLength);

  /*********************************************************************
   * If initial generation, set initial agent configuration
   *********************************************************************/
  if (_k->_currentGeneration == 0)
  {
    _currentEpisode = 0;
    _policyUpdateCount = 0;
    _experienceCount = 0;

    // Initializing training and episode statistics //TODO go through all
    _testingAverageReturn = -korali::Inf;
    _testingBestReturn = -korali::Inf;
    _testingWorstReturn = -korali::Inf;
    _testingBestAverageReturn = -korali::Inf;
    _testingBestEpisodeId = 0;
    _trainingBestReturn.resize(numAgents, -korali::Inf);
    _trainingBestEpisodeId.resize(numAgents, 0);
    _trainingAverageReturn.resize(numAgents, -korali::Inf);

    /* Initializing REFER information */

    // If cutoff scale is not defined, use a heuristic value [defaults to 4.0]
    if (_experienceReplayOffPolicyCutoffScale <= 0.0f)
      KORALI_LOG_ERROR("Experience Replay Cutoff Scale must be larger 0.0");
    _experienceReplayOffPolicyCount.resize(numAgents, 0);
    _experienceReplayOffPolicyRatio.resize(numAgents, 0.0f);
    if (_learningRate <= 0.0f)
      KORALI_LOG_ERROR("Learning rate must be larger 0.0");
    _currentLearningRate = _learningRate;

    _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale;

    // Rescaling information
    _stateRescalingMeans = std::vector<std::vector<float>>(numAgents, std::vector<float>(_problem->_stateVectorSize, 0.0f));
    _stateRescalingSigmas = std::vector<std::vector<float>>(numAgents, std::vector<float>(_problem->_stateVectorSize, 1.0f));
    _rewardRescalingSigma = 1.;
    _rewardRescalingSumSquaredRewards = 0.;

    // If not given, get agent's initial policy
    if (not isDefined(_trainingCurrentPolicies, "Policy Hyperparameters"))
      _trainingCurrentPolicies = getPolicy();
  }

  // Setting current agent's training state
  setPolicy(_trainingCurrentPolicies["Policy Hyperparameters"]);

  // If this continues a previous training run, deserialize previous input experience replay. Only for the root (engine) rank
  if (_k->_currentGeneration > 0)
    if (_mode == "Training")
      if (_k->_engine->_conduit != NULL)
        deserializeExperienceReplay();

  // Initializing session-wise profiling timers
  _sessionRunningTime = 0.0;
  _sessionSerializationTime = 0.0;
  _sessionWorkerComputationTime = 0.0;
  _sessionWorkerCommunicationTime = 0.0;
  _sessionPolicyEvaluationTime = 0.0;
  _sessionPolicyUpdateTime = 0.0;
  _sessionWorkerAttendingTime = 0.0;

  // Initializing session-specific counters
  _sessionExperienceCount = 0;
  _sessionEpisodeCount = 0;
  _sessionGeneration = 1;
  _sessionPolicyUpdateCount = 0;

  // Calculating how many more experiences do we need in this session to reach the starting size
  _sessionExperiencesUntilStartSize = _stateBuffer.size() > _experienceReplayStartSize ? 0 : _experienceReplayStartSize - _stateBuffer.size();

  if (_mode == "Training")
  {
    // Creating storate for _agents and their status
    _workers.resize(_concurrentWorkers);
    _isWorkerRunning.resize(_concurrentWorkers, false);

    // In case the agent was tested before, remove _testingCurrentPolicies
    _testingCurrentPolicies.clear();
  }

  if (_mode == "Testing")
  {
    // Fixing termination criteria for testing mode
    _maxGenerations = _k->_currentGeneration + 1;

    // Setting testing policy to best testing hyperparameters if not custom-set by the user
    if (_testingCurrentPolicies.empty())
    {
      // Checking if testing policies have been generated
      if (_testingBestPolicies.empty())
      {
        _k->_logger->logInfo("Minimal", "Using current training policy for testing.\n");
        _testingCurrentPolicies = _trainingCurrentPolicies;
      }
      else
      {
        _k->_logger->logInfo("Minimal", "Using best testing policy for test-run.\n");
        _testingCurrentPolicies = _testingBestPolicies;
      }
    }

    // Checking if there's testing samples defined
    if (_testingSampleIds.size() == 0)
      KORALI_LOG_ERROR("For testing, you need to indicate the sample ids to run in the ['Testing']['Sample Ids'] field.\n");

    // Prepare storage for rewards from tested samples
    _testingReturn.resize(_testingSampleIds.size());
  }
}

void Agent::runGeneration()
{
  if (_mode == "Training") trainingGeneration();
  if (_mode == "Testing") testingGeneration();
}

void Agent::trainingGeneration()
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Setting generation-specific timers
  _generationRunningTime = 0.0;
  _generationSerializationTime = 0.0;
  _generationWorkerComputationTime = 0.0;
  _generationWorkerCommunicationTime = 0.0;
  _generationPolicyEvaluationTime = 0.0;
  _generationPolicyUpdateTime = 0.0;
  _generationWorkerAttendingTime = 0.0;

  // Running until all _workers have finished
  while (_sessionEpisodeCount < _episodesPerGeneration * _sessionGeneration)
  {
    // Launching (or re-launching) agents
    for (size_t workerId = 0; workerId < _concurrentWorkers; workerId++)
      if (_isWorkerRunning[workerId] == false)
      {
        _workers[workerId]["Sample Id"] = _currentEpisode++;
        _workers[workerId]["Module"] = "Problem";
        _workers[workerId]["Operation"] = "Run Training Episode";
        for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
          _workers[workerId]["Policy Hyperparameters"][p] = _trainingCurrentPolicies["Policy Hyperparameters"][p];
        _workers[workerId]["State Rescaling"]["Means"] = _stateRescalingMeans;
        _workers[workerId]["State Rescaling"]["Standard Deviations"] = _stateRescalingSigmas;

        KORALI_START(_workers[workerId]);

        _isWorkerRunning[workerId] = true;
      }

    // Listening to _workers for incoming experiences
    KORALI_LISTEN(_workers);

    // Attending to running agents, checking if any experience has been received
    for (size_t workerId = 0; workerId < _concurrentWorkers; workerId++)
      if (_isWorkerRunning[workerId] == true)
        attendWorker(workerId);

    // Perform optimization steps on the critic/policy, if reached the minimum replay memory size
    if (_experienceCount >= _experienceReplayStartSize)
    {
      // If we accumulated enough experiences, we rescale the states (once)
      if (_stateRescalingEnabled == true)
        if (_policyUpdateCount == 0)
          rescaleStates();

      // If we accumulated enough experiences between updates in this session, update now
      while (_sessionExperienceCount > (_experiencesBetweenPolicyUpdates * _sessionPolicyUpdateCount + _sessionExperiencesUntilStartSize))
      {
        auto beginTime = std::chrono::steady_clock::now(); // Profiling

        // Calling the algorithm specific policy training algorithm
        trainPolicy();

        auto endTime = std::chrono::steady_clock::now();                                                                  // Profiling
        _sessionPolicyUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
        _generationPolicyUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling

        // Increasing policy update counters
        _policyUpdateCount++;
        _sessionPolicyUpdateCount++;

        // Updating the off policy cutoff
        _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale / (1.0f + _experienceReplayOffPolicyAnnealingRate * (float)_policyUpdateCount);

        for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
        {
          // Updating REFER learning rate and beta parameters
          _currentLearningRate = _learningRate / (1.0f + _experienceReplayOffPolicyAnnealingRate * (float)_policyUpdateCount);
          if (_experienceReplayOffPolicyRatio[a] > _experienceReplayOffPolicyTarget)
            _experienceReplayOffPolicyREFERCurrentBeta[a] = (1.0f - _currentLearningRate) * _experienceReplayOffPolicyREFERCurrentBeta[a];
          else
            _experienceReplayOffPolicyREFERCurrentBeta[a] = (1.0f - _currentLearningRate) * _experienceReplayOffPolicyREFERCurrentBeta[a] + _currentLearningRate;
        }
      }

      // Getting new policy hyperparameters (for agents to generate actions)
      _trainingCurrentPolicies = getPolicy();
    }
  }

  // Now serializing experience replay database
  if (_experienceReplaySerialize == true)
    if (_k->_fileOutputEnabled)
      if (_k->_fileOutputFrequency > 0)
        if (_k->_currentGeneration % _k->_fileOutputFrequency == 0)
          serializeExperienceReplay();

  // Measuring generation time
  auto endTime = std::chrono::steady_clock::now();                                                             // Profiling
  _sessionRunningTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationRunningTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling

  /*********************************************************************
   * Updating statistics/bookkeeping
   *********************************************************************/

  // Updating average cumulative reward statistics
  _trainingAverageReturn = std::vector<float>(_problem->_agentsPerEnvironment, 0.0f);
  for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
  {
    ssize_t startEpisodeId = _trainingReturnHistory[a].size() - _trainingAverageDepth;
    ssize_t endEpisodeId = _trainingReturnHistory[a].size() - 1;
    if (startEpisodeId < 0) startEpisodeId = 0;
    for (ssize_t e = startEpisodeId; e <= endEpisodeId; e++)
      _trainingAverageReturn[a] += _trainingReturnHistory[a][e];
    _trainingAverageReturn[a] /= (float)(endEpisodeId - startEpisodeId + 1);
  }

  // Increasing session's generation count
  _sessionGeneration++;
}

void Agent::testingGeneration()
{
  // Allocating testing agents
  std::vector<Sample> testingWorkers(_testingSampleIds.size());

  // Launching  agents
  for (size_t workerId = 0; workerId < _testingSampleIds.size(); workerId++)
  {
    testingWorkers[workerId]["Sample Id"] = _testingSampleIds[workerId];
    testingWorkers[workerId]["Module"] = "Problem";
    testingWorkers[workerId]["Operation"] = "Run Testing Episode";
    for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
      testingWorkers[workerId]["Policy Hyperparameters"][p] = _testingCurrentPolicies["Policy Hyperparameters"][p];
    testingWorkers[workerId]["State Rescaling"]["Means"] = _stateRescalingMeans;
    testingWorkers[workerId]["State Rescaling"]["Standard Deviations"] = _stateRescalingSigmas;

    KORALI_START(testingWorkers[workerId]);
  }

  KORALI_WAITALL(testingWorkers);

  for (size_t workerId = 0; workerId < _testingSampleIds.size(); workerId++)
    _testingReturn[workerId] = KORALI_GET(float, testingWorkers[workerId], "Testing Reward");
}

void Agent::rescaleStates()
{
  // Calculation of state moments
  std::vector<std::vector<float>> sumStates(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_stateVectorSize, 0.0f));
  std::vector<std::vector<float>> squaredSumStates(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_stateVectorSize, 0.0f));

  for (size_t i = 0; i < _stateBuffer.size(); ++i)
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
      for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
      {
        sumStates[a][d] += _stateBuffer[i][a][d];
        squaredSumStates[a][d] += _stateBuffer[i][a][d] * _stateBuffer[i][a][d];
      }

  _k->_logger->logInfo("Detailed", " + Using State Normalization N(Mean, Sigma):\n");

  for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
    for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
    {
      _stateRescalingMeans[a][d] = sumStates[a][d] / (float)_stateBuffer.size();
      if (std::isfinite(_stateRescalingMeans[a][d]) == false) _stateRescalingMeans[a][d] = 0.0f;

      _stateRescalingSigmas[a][d] = std::sqrt(squaredSumStates[a][d] / (float)_stateBuffer.size() - _stateRescalingMeans[a][d] * _stateRescalingMeans[a][d]);
      if (std::isfinite(_stateRescalingSigmas[a][d]) == false) _stateRescalingSigmas[a][d] = 1.0f;
      if (_stateRescalingSigmas[a][d] <= 1e-9) _stateRescalingSigmas[a][d] = 1.0f;

      _k->_logger->logInfo("Detailed", " + State [%zu]: N(%f, %f)\n", d, _stateRescalingMeans[a][d], _stateRescalingSigmas[a][d]);
    }

  // Actual rescaling of initial states
  for (size_t i = 0; i < _stateBuffer.size(); ++i)
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
      for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
        _stateBuffer[i][a][d] = (_stateBuffer[i][a][d] - _stateRescalingMeans[a][d]) / _stateRescalingSigmas[a][d];
}

void Agent::attendWorker(size_t workerId)
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Storage for the incoming message
  knlohmann::json message;

  // Retrieving the experience, if any has arrived for the current agent.
  if (_workers[workerId].retrievePendingMessage(message))
  {
    // Getting episode Id
    size_t episodeId = message["Sample Id"];
    message["Episodes"]["Sample Id"] = episodeId;

    // If agent requested new policy, send the new hyperparameters
    if (message["Action"] == "Request New Policy")
    {
      KORALI_SEND_MSG_TO_SAMPLE(_workers[workerId], _trainingCurrentPolicies["Policy Hyperparameters"]);
    }

    // Process episode(s) incoming from the agent(s)
    if (message["Action"] == "Send Episodes")
    {
      // Process every episode received and its experiences (add them to replay memory)
      processEpisode(message["Episodes"]);

      // Increasing total experience counters
      _experienceCount += message["Episodes"]["Experiences"].size();
      _sessionExperienceCount += message["Episodes"]["Experiences"].size();

      // Waiting for the agent to come back with all the information
      KORALI_WAIT(_workers[workerId]);

      // Getting the training reward of the latest episodes
      _trainingLastReturn = KORALI_GET(std::vector<float>, _workers[workerId], "Training Rewards");

      // Keeping training statistics. Updating if exceeded best training policy so far.
      for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
      {
        if (_trainingLastReturn[a] > _trainingBestReturn[a])
        {
          _trainingBestReturn[a] = _trainingLastReturn[a];
          _trainingBestEpisodeId[a] = episodeId;
        }
        _trainingReturnHistory[a].push_back(_trainingLastReturn[a]);
      }
      // Storing bookkeeping information
      _trainingExperienceHistory.push_back(message["Episodes"]["Experiences"].size());

      // If the policy has exceeded the threshold during training, we gather its statistics
      if (_workers[workerId]["Tested Policy"] == true)
      {
        _testingCandidateCount++;
        _testingBestReturn = KORALI_GET(float, _workers[workerId], "Best Testing Reward");
        _testingWorstReturn = KORALI_GET(float, _workers[workerId], "Worst Testing Reward");
        _testingAverageReturn = KORALI_GET(float, _workers[workerId], "Average Testing Reward");
        _testingAverageReturnHistory.push_back(_testingAverageReturn);

        // If the average testing reward is better than the previous best, replace it
        // and store hyperparameters as best so far.
        if (_testingAverageReturn > _testingBestAverageReturn)
        {
          _testingBestAverageReturn = _testingAverageReturn;
          _testingBestEpisodeId = episodeId;
          for (size_t d = 0; d < _problem->_policiesPerEnvironment; ++d)
            _testingBestPolicies["Policy Hyperparameters"][d] = _workers[workerId]["Policy Hyperparameters"][d];
        }
      }

      // Obtaining profiling information
      _sessionWorkerComputationTime += KORALI_GET(double, _workers[workerId], "Computation Time");
      _sessionWorkerCommunicationTime += KORALI_GET(double, _workers[workerId], "Communication Time");
      _sessionPolicyEvaluationTime += KORALI_GET(double, _workers[workerId], "Policy Evaluation Time");
      _generationWorkerComputationTime += KORALI_GET(double, _workers[workerId], "Computation Time");
      _generationWorkerCommunicationTime += KORALI_GET(double, _workers[workerId], "Communication Time");
      _generationPolicyEvaluationTime += KORALI_GET(double, _workers[workerId], "Policy Evaluation Time");

      // Set agent as finished
      _isWorkerRunning[workerId] = false;

      // Increasing session episode count
      _sessionEpisodeCount++;
    }
  }

  auto endTime = std::chrono::steady_clock::now();                                                                     // Profiling
  _sessionWorkerAttendingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationWorkerAttendingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling
}

void Agent::processEpisode(knlohmann::json &episode)
{
  /*********************************************************************
   * Adding episode's experiences into the replay memory
   *********************************************************************/
  const size_t episodeId = episode["Sample Id"];
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Storage for the episode's discounted cumulative reward
  float discountFactor = 1;
  std::vector<float> discountedCumulativeReward(numAgents, 0.0);

  // Go over experiences in episode
  const size_t episodeExperienceCount = episode["Experiences"].size();
  for (size_t expId = 0; expId < episodeExperienceCount; expId++)
  {
    // Put state to replay memory
    _stateBuffer.add(episode["Experiences"][expId]["State"].get<std::vector<std::vector<float>>>());

    // Get action and put it to replay memory
    _actionBuffer.add(episode["Experiences"][expId]["Action"].get<std::vector<std::vector<float>>>());

    // Get reward
    std::vector<float> reward = episode["Experiences"][expId]["Reward"].get<std::vector<float>>();

    // For cooporative multi-agent model rewards are averaged
    if (_multiAgentRelationship == "Cooperation")
    {
      float avgReward = std::accumulate(reward.begin(), reward.end(), 0.);
      avgReward /= numAgents;
      reward = std::vector<float>(numAgents, avgReward);
    }

    // Update reward rescaling moments
    if (_rewardRescalingEnabled)
    {
      if (_rewardBufferContiguous.size() >= _experienceReplayMaximumSize * numAgents)
      {
        for (size_t a = 0; a < numAgents; a++)
          _rewardRescalingSumSquaredRewards -= _rewardBufferContiguous[a] * _rewardBufferContiguous[a];
      }
      for (size_t a = 0; a < numAgents; a++)
      {
        _rewardRescalingSumSquaredRewards += reward[a] * reward[a];
      }
    }

    // Put reward to replay memory
    for (size_t a = 0; a < numAgents; a++)
      _rewardBufferContiguous.add(reward[a]);

    // Keeping statistics
    for (size_t a = 0; a < numAgents; a++)
      discountedCumulativeReward[a] += discountFactor * reward[a];

    // Checking and adding experience termination status and truncated state to replay memory
    termination_t termination;
    std::vector<std::vector<float>> truncatedState;
    std::vector<float> truncatedStateValue;

    if (episode["Experiences"][expId]["Termination"] == "Non Terminal") termination = e_nonTerminal;
    if (episode["Experiences"][expId]["Termination"] == "Terminal") termination = e_terminal;
    if (episode["Experiences"][expId]["Termination"] == "Truncated")
    {
      termination = e_truncated;
      truncatedState = episode["Experiences"][expId]["Truncated State"].get<std::vector<std::vector<float>>>();
    }

    _terminationBuffer.add(termination);
    _truncatedStateBuffer.add(truncatedState);
    _truncatedStateValueBuffer.add(truncatedStateValue);

    // Getting policy information and state value
    std::vector<policy_t> expPolicy(numAgents);
    std::vector<float> stateValue(numAgents);

    if (isDefined(episode["Experiences"][expId], "Policy", "State Value"))
    {
      stateValue = episode["Experiences"][expId]["Policy"]["State Value"].get<std::vector<float>>();
      for (size_t a = 0; a < numAgents; a++)
      {
        expPolicy[a].stateValue = stateValue[a];
      }
    }
    else
    {
      KORALI_LOG_ERROR("Policy has not produced state value for the current experience.\n");
    }
    for (size_t a = 0; a < numAgents; a++)
      _stateValueBufferContiguous.add(stateValue[a]);

    /* Story policy information for continuous action spaces */
    if (isDefined(episode["Experiences"][expId], "Policy", "Distribution Parameters"))
    {
      const auto distParams = episode["Experiences"][expId]["Policy"]["Distribution Parameters"].get<std::vector<std::vector<float>>>();
      for (size_t a = 0; a < numAgents; a++)
        expPolicy[a].distributionParameters = distParams[a];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Unbounded Action"))
    {
      const auto unboundedAc = episode["Experiences"][expId]["Policy"]["Unbounded Action"].get<std::vector<std::vector<float>>>();
      for (size_t a = 0; a < numAgents; a++)
        expPolicy[a].unboundedAction = unboundedAc[a];
    }

    /* Story policy information for discrete action spaces */
    if (isDefined(episode["Experiences"][expId], "Policy", "Action Index"))
    {
      const auto actIdx = episode["Experiences"][expId]["Policy"]["Action Index"].get<std::vector<size_t>>();
      for (size_t a = 0; a < numAgents; a++)
        expPolicy[a].actionIndex = actIdx[a];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Action Probabilities"))
    {
      const auto actProb = episode["Experiences"][expId]["Policy"]["Action Probabilities"].get<std::vector<std::vector<float>>>();
      for (size_t a = 0; a < numAgents; a++)
        expPolicy[a].actionProbabilities = actProb[a];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Available Actions"))
    {
      const auto availAct = episode["Experiences"][expId]["Policy"]["Available Actions"].get<std::vector<std::vector<size_t>>>();
      for (size_t a = 0; a < numAgents; a++)
      {
        expPolicy[a].availableActions = availAct[a];
        if (std::accumulate(expPolicy[a].availableActions.begin(), expPolicy[a].availableActions.end(), 0) == 0)
          KORALI_LOG_ERROR("State with experience id %zu for agent %zu detected with no available actions.", expId, a);
      }
    }

    // Storing policy information in replay memory
    _expPolicyBuffer.add(expPolicy);
    _curPolicyBuffer.add(expPolicy);

    // Storing Episode information in replay memory
    _episodeIdBuffer.add(episodeId);
    _episodePosBuffer.add(expId);

    // Adding placeholder for retrace value
    for (size_t a = 0; a < numAgents; a++)
      _retraceValueBufferContiguous.add(0.0f);

    // If outgoing experience is off policy, subtract off policy counter
    if (_isOnPolicyBuffer.size() == _experienceReplayMaximumSize)
    {
      const auto &onPolicyBuffer = _isOnPolicyBuffer[0];

      size_t count = 1;
      // Consider all observation for the off-policy statistics
      if (_problem->_policiesPerEnvironment == 1)
        count = std::count(onPolicyBuffer.begin(), onPolicyBuffer.end(), false);

      // Update offPolicyCount
      for (size_t a = 0; a < numAgents; a++)
        if ((onPolicyBuffer[a] == false) || (_problem->_policiesPerEnvironment == 1))
        {
          // Safety check for overflow
          if (_experienceReplayOffPolicyCount[a] < count)
          {
            KORALI_LOG_ERROR("Agent %ld: _experienceReplayOffPolicyCount=%ld smaller than decrement %ld.\n", a, _experienceReplayOffPolicyCount[a], count);
            // count = _experienceReplayOffPolicyCount[d];
          }

          // Update off-policy count
          _experienceReplayOffPolicyCount[a] -= count;
        }
    }

    // Adding new experience's on policiness (by default is true when adding it to the ER)
    _isOnPolicyBuffer.add(std::vector<char>(numAgents, true));

    // Initialize experience's importance weight (1.0 because its freshly produced)
    _importanceWeightBuffer.add(std::vector<float>(numAgents, 1.0f));
    for (size_t a = 0; a < numAgents; a++)
      _truncatedImportanceWeightBufferContiguous.add(1.0f);
    _productImportanceWeightBuffer.add(1.0f);

    // Calculate running discount factor
    discountFactor *= _discountFactor;
  }

  // Storing discounted reward
  for (size_t a = 0; a < numAgents; a++)
    _trainingDiscountedReturnHistory[a].push_back(discountedCumulativeReward[a]);

  /*********************************************************************
   * Computing initial retrace value for the newly added experiences
   *********************************************************************/

  // Getting position of the final experience of the episode in the replay memory
  ssize_t endId = (ssize_t)_stateBuffer.size() - 1;

  // Getting the starting ID of the initial experience of the episode in the replay memory
  ssize_t startId = endId - episodeExperienceCount + 1;

  // Storage for the retrace value
  std::vector<float> retV(numAgents, 0.0f);

  // If it was a truncated episode, add the value function for the terminal state to retV
  if (_terminationBuffer[endId] == e_truncated)
  {
    for (size_t a = 0; a < numAgents; a++)
    {
      // Get truncated state
      auto expTruncatedStateSequence = getTruncatedStateSequence(endId, a);

      // Forward tuncated state. Take policy d if there is multiple policies, otherwise policy 0
      std::vector<policy_t> truncatedPolicy;
      if (_problem->_policiesPerEnvironment == 1)
        retV[a] = calculateStateValue(expTruncatedStateSequence);
      else
        retV[a] = calculateStateValue(expTruncatedStateSequence, a);

      // Get value of trucated state
      if (std::isfinite(retV[a]) == false)
        KORALI_LOG_ERROR("Calculated state value for truncated state returned an invalid value: %f\n", retV[a]);
    }

    // For cooporative multi-agent model truncated state-values are averaged
    if (_multiAgentRelationship == "Cooperation")
    {
      float avgRetV = std::accumulate(retV.begin(), retV.end(), 0.);
      avgRetV /= numAgents;
      retV = std::vector<float>(numAgents, avgRetV);
    }

    // The value of the truncated state equals initial retrace Value
    _truncatedStateValueBuffer[endId] = retV;
  }

  // Now going backwards, setting the retrace value of every experience
  for (ssize_t expId = endId; expId >= startId; expId--)
    for (size_t a = 0; a < numAgents; a++)
    {
      // Calculating retrace value. Importance weight is 1.0f because the policy is current.
      retV[a] = getScaledReward(_rewardBufferContiguous[expId * numAgents + a]) + _discountFactor * retV[a];
      _retraceValueBufferContiguous[expId * numAgents + a] = retV[a];
    }

  // Update reward rescaling sigma
  if (_rewardRescalingEnabled)
  {
    const float sigma = std::sqrt(_rewardRescalingSumSquaredRewards / ((float)_rewardBufferContiguous.size()) + 1e-9);
    if (sigma > 1e-9) _rewardRescalingSigma = sigma;
  }
}

std::vector<std::pair<size_t, size_t>> Agent::generateMiniBatch()
{
  // Get number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Allocating storage for mini batch experiecne indexes
  std::vector<std::pair<size_t, size_t>> miniBatch(_miniBatchSize * numAgents);

  // Fill minibatch
  for (size_t b = 0; b < _miniBatchSize; b++)
  {
    // Producing random (uniform) number for the selection of the experience
    float x = _uniformGenerator->getRandomNumber();

    // Selecting experience
    size_t expId = std::floor(x * (float)(_stateBuffer.size() - 1));

    for (size_t a = 0; a < numAgents; a++)
    {
      // Setting experience
      miniBatch[b * numAgents + a].first = expId;
      miniBatch[b * numAgents + a].second = a;

      // Sample agentId
      if (_multiAgentSampling == "Experience")
      {
        // Producing random (uniform) number for the selection of the experience
        float ax = _uniformGenerator->getRandomNumber();

        // Selecting agent
        miniBatch[b * numAgents + a].second = std::floor(ax * (float)(numAgents - 1));
      }

      // Sample both
      if (_multiAgentSampling == "Baseline")
      {
        // Producing random (uniform) number for the selection of the experience
        float ex = _uniformGenerator->getRandomNumber();
        float ax = _uniformGenerator->getRandomNumber();

        // Selecting experience
        miniBatch[b * numAgents + a].first = std::floor(ex * (float)(_stateBuffer.size() - 1));

        // Selecting agent
        miniBatch[b * numAgents + a].second = std::floor(ax * (float)(numAgents - 1));
      }
    }
  }

  // clang-format off
  // Sorting minibatch: first by expId, second by agentId
  // to quickly detect duplicates when updating metadata
  std::sort(miniBatch.begin(), miniBatch.end(), [numAgents](const std::pair<size_t, size_t> &exp0, const std::pair<size_t, size_t> &exp1) -> bool {
    return exp0.first * numAgents + exp0.second < exp1.first * numAgents + exp1.second;
  });
  // clang-format on

  // Returning generated minibatch
  return miniBatch;
}

std::vector<std::vector<std::vector<float>>> Agent::getMiniBatchStateSequence(const std::vector<std::pair<size_t, size_t>> &miniBatch)
{
  // Get number of experiences in minibatch
  const size_t numExperiences = miniBatch.size();

  // Allocating state sequence vector
  std::vector<std::vector<std::vector<float>>> stateSequence(numExperiences);

#pragma omp parallel for
  for (size_t b = 0; b < numExperiences; b++)
  {
    // Getting current expId and agentId
    const size_t expId = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Getting starting expId
    const size_t startId = getTimeSequenceStartExpId(expId);

    // Calculating time sequence length
    const size_t T = expId - startId + 1;

    // Resizing state sequence vector to the correct time sequence length
    stateSequence[b].resize(T);
    for (size_t t = 0; t < T; t++)
    {
      // Now adding states
      const size_t sequenceId = startId + t;
      stateSequence[b][t].reserve(_problem->_stateVectorSize);
      stateSequence[b][t].insert(stateSequence[b][t].begin(), _stateBuffer[sequenceId][agentId].begin(), _stateBuffer[sequenceId][agentId].end());
    }
  }

  return stateSequence;
}

void Agent::updateExperienceMetadata(const std::vector<std::pair<size_t, size_t>> &miniBatch, const std::vector<policy_t> &policyData)
{
  const size_t miniBatchSize = miniBatch.size();
  const size_t numAgents = _problem->_agentsPerEnvironment;

  /* Creating a selection of unique expIds, agentIds
   * Important: this assumes the minibatch ids are sorted.
   */

  // Create Buffers
  std::vector<size_t> updateBatch;
  std::vector<std::pair<size_t, size_t>> updateMinibatch;
  std::vector<policy_t> updatePolicyData;

  // Fill updateMinibatch and updatePolicyData
  size_t b = 0;
  while (b < miniBatchSize)
  {
    // Add new unique combination
    updateBatch.push_back(b);
    updateMinibatch.push_back(miniBatch[b]);
    updatePolicyData.push_back(policyData[b]);

    size_t a = 1;

    // Iterate over experiences with same expId
    while ((miniBatch[b + a].first == miniBatch[b + a - 1].first) && (b + a < miniBatchSize))
    {
      // Add unique experiences from agents
      if (miniBatch[b + a].second != miniBatch[b + a - 1].second)
      {
        updateMinibatch.push_back(miniBatch[b + a]);
        updatePolicyData.push_back(policyData[b + a]);
      }
      a++;
    }

    // Increment batch counter by the number of same expIds
    b += a;
  }

  // Container to compute offpolicy count difference in minibatch
  std::vector<int> offPolicyCountDelta(numAgents, 0);

#pragma omp parallel for reduction(vec_int_plus \
                                   : offPolicyCountDelta)
  for (size_t i = 0; i < updateMinibatch.size(); i++)
  {
    // Get current expId and agentId
    const size_t expId = updateMinibatch[i].first;
    const size_t agentId = updateMinibatch[i].second;

    // Get and set current policy
    const auto &curPolicy = updatePolicyData[i];
    _curPolicyBuffer[expId][agentId] = curPolicy;

    // Get state value
    _stateValueBufferContiguous[expId * numAgents + agentId] = curPolicy.stateValue;
    if (std::isfinite(curPolicy.stateValue) == false)
      KORALI_LOG_ERROR("Calculated state value returned an invalid value: %f\n", curPolicy.stateValue);

    // Get action and policy for this experience
    const auto &expAction = _actionBuffer[expId][agentId];
    const auto &expPolicy = _expPolicyBuffer[expId][agentId];

    // Compute importance weight
    const float importanceWeight = calculateImportanceWeight(expAction, curPolicy, expPolicy);
    if (std::isfinite(importanceWeight) == false)
      KORALI_LOG_ERROR("Calculated value of importanceWeight returned an invalid value: %f\n", importanceWeight);

    // Set importance weight and truncated importance weight
    _importanceWeightBuffer[expId][agentId] = importanceWeight;
    _truncatedImportanceWeightBufferContiguous[expId * numAgents + agentId] = std::min(_importanceWeightTruncationLevel, importanceWeight);

    // Keep track of off-policyness (in principle only necessary for agentId==policyId)
    if (not _multiAgentCorrelation)
    {
      // Checking if experience is on policy
      const bool isOnPolicy = (importanceWeight > (1.0f / _experienceReplayOffPolicyCurrentCutoff)) && (importanceWeight < _experienceReplayOffPolicyCurrentCutoff);

      // Updating off policy count if a change is detected
      if (_isOnPolicyBuffer[expId][agentId] == true && isOnPolicy == false)
        offPolicyCountDelta[agentId]++;

      if (_isOnPolicyBuffer[expId][agentId] == false && isOnPolicy == true)
        offPolicyCountDelta[agentId]--;

      // Write to onPolicy vector
      _isOnPolicyBuffer[expId][agentId] = isOnPolicy;
    }

    // Update truncated state value
    if (_terminationBuffer[expId] == e_truncated)
    {
      // Get truncated state
      auto expTruncatedStateSequence = getTruncatedStateSequence(expId, agentId);

      // Forward tuncated state
      // TODO: other policy for exp-sharing in multi-policy case??
      float truncatedStateValue;
      if (_problem->_policiesPerEnvironment == 1)
        truncatedStateValue = calculateStateValue(expTruncatedStateSequence);
      else
        truncatedStateValue = calculateStateValue(expTruncatedStateSequence, agentId);

      // Check value of trucated state
      if (std::isfinite(truncatedStateValue) == false)
        KORALI_LOG_ERROR("Calculated state value for truncated state returned an invalid value: %f\n", truncatedStateValue);

      // Write truncated state value
      _truncatedStateValueBuffer[expId][agentId] = truncatedStateValue;
    }
  }

  /* Now taking care of advanced correlation features for MARL */

  if (_multiAgentCorrelation)
  {
#pragma omp parallel for reduction(vec_int_plus \
                                   : offPolicyCountDelta)
    for (size_t i = 0; i < updateBatch.size(); i++)
    {
      const size_t batchId = updateBatch[i];
      const size_t expId = miniBatch[batchId].first;

      // Load importance weight for expId
      const auto &importanceWeight = _importanceWeightBuffer[expId];

      // Compute product of importance weights
      float logProdImportanceWeight = 0.0f;
      for (size_t a = 0; a < numAgents; a++)
      {
        // Numerical safety [ ln(10)=2.30258509299 ]
        if (importanceWeight[a] == 0)
          logProdImportanceWeight += 2.30258509299 * std::numeric_limits<float>::min_exponent10;
        else
          logProdImportanceWeight += std::log(importanceWeight[a]);
      }

      // Compute cut-off in log-space
      const float logCutOff = (float)numAgents * std::log(_experienceReplayOffPolicyCurrentCutoff);

      // Check whether experience is onPolicy
      const bool onPolicy = (logProdImportanceWeight > (-1. * logCutOff)) && (logProdImportanceWeight < logCutOff);

      // Load isOnPolicy
      auto &isOnPolicy = _isOnPolicyBuffer[expId];

      // Write to prodImportanceWeight vector
      _productImportanceWeightBuffer[expId] = std::exp(logProdImportanceWeight);

      // Truncate product of importance weight
      const float truncatedProdImportanceWeight = std::min(_importanceWeightTruncationLevel, _productImportanceWeightBuffer[expId]);

      // Overwrite truncated importance weights
      for (size_t a = 0; a < numAgents; a++)
        _truncatedImportanceWeightBufferContiguous[expId * numAgents + a] = truncatedProdImportanceWeight;

      // Record change of on-policyness
      for (size_t d = 0; d < numAgents; d++)
      {
        if (isOnPolicy[d] == true && onPolicy == false)
          offPolicyCountDelta[d]++;

        if (isOnPolicy[d] == false && onPolicy == true)
          offPolicyCountDelta[d]--;
      }

      // Overwrite onPolicyVector
      std::fill(isOnPolicy.begin(), isOnPolicy.end(), onPolicy);
    }
  }

  // Average state values for cooperative MA
  if (_multiAgentRelationship == "Cooperation")
  {
#pragma omp parallel for
    for (size_t i = 0; i < updateBatch.size(); i++)
    {
      const size_t batchId = updateBatch[i];
      const size_t expId = miniBatch[batchId].first;

      // Average state-value
      float averageStateValue = 0.0f;
      for (size_t a = 0; a < numAgents; a++)
        averageStateValue += _stateValueBufferContiguous[expId * numAgents + a];
      averageStateValue /= numAgents;

      // Overwrite state value with average
      for (size_t a = 0; a < numAgents; a++)
        _stateValueBufferContiguous[expId * numAgents + a] = averageStateValue;

      // Same for truncated state-value
      if (_terminationBuffer[expId] == e_truncated)
      {
        // Load truncated state value
        auto &truncatedStateValue = _truncatedStateValueBuffer[expId];

        // Average truncated state value
        float averageTruncatedStateValue = std::accumulate(truncatedStateValue.begin(), truncatedStateValue.end(), 0.);
        averageTruncatedStateValue /= numAgents;

        // Overwrite truncated state value with average
        std::fill(truncatedStateValue.begin(), truncatedStateValue.end(), averageTruncatedStateValue);
      }
    }
  }

  /* Taking care of off-policy statistics */

  if (_problem->_policiesPerEnvironment == 1)
  {
    // Consider all observation for the off-policy statistics
    int sumOffPolicyCountDelta = std::accumulate(offPolicyCountDelta.begin(), offPolicyCountDelta.end(), 0.);
    offPolicyCountDelta = std::vector<int>(numAgents, sumOffPolicyCountDelta);
  }

  // Updating the off policy count and ratio
  for (size_t a = 0; a < numAgents; a++)
  {
    // Safety check for overflow
    if ((int)_experienceReplayOffPolicyCount[a] < -offPolicyCountDelta[a])
      KORALI_LOG_ERROR("Agent %ld: offPolicyCountDelta=%d bigger than _experienceReplayOffPolicyCount=%ld.\n", a, offPolicyCountDelta[a], _experienceReplayOffPolicyCount[a]);

    // Update off policy count
    _experienceReplayOffPolicyCount[a] += offPolicyCountDelta[a];
    _experienceReplayOffPolicyRatio[a] = (float)_experienceReplayOffPolicyCount[a] / (float)_isOnPolicyBuffer.size();

    // Normalize off policy Ratio
    if (_problem->_policiesPerEnvironment == 1)
      _experienceReplayOffPolicyRatio[a] /= (float)numAgents;
  }

  /* Update Retrace value */

  // Now filtering experiences from the same episode
  std::vector<size_t> retraceMiniBatch;

  // Adding last experience from the sorted minibatch
  retraceMiniBatch.push_back(miniBatch[updateBatch.back()].first);

  // Adding experiences so long as they do not repeat episodes
  for (ssize_t i = updateBatch.size() - 2; i >= 0; i--)
  {
    size_t currExpId = miniBatch[updateBatch[i]].first;
    size_t nextExpId = miniBatch[updateBatch[i + 1]].first;
    size_t curEpisode = _episodeIdBuffer[currExpId];
    size_t nextEpisode = _episodeIdBuffer[nextExpId];
    if (curEpisode != nextEpisode) retraceMiniBatch.push_back(currExpId);
  }

// Calculating retrace value for the oldest experiences of unique episodes
#pragma omp parallel for schedule(guided, 1)
  for (size_t i = 0; i < retraceMiniBatch.size(); i++)
  {
    // Determine start of the episode
    ssize_t endId = retraceMiniBatch[i];
    ssize_t startId = endId - _episodePosBuffer[endId];

    // If start of episode has been discarded, take earliest one
    if (startId < 0) startId = 0;

    // Storage for the retrace value
    std::vector<float> retV(numAgents, 0.0f);

    // For truncated episode, set truncated state value function
    if (_terminationBuffer[endId] == e_truncated)
      retV = _truncatedStateValueBuffer[endId];

    // If non-terminal state, set next retrace value
    if (_terminationBuffer[endId] == e_nonTerminal)
      for (size_t a = 0; a < numAgents; a++)
        retV[a] = _retraceValueBufferContiguous[(endId + 1) * numAgents + a];

    // Now iterating backwards and compute retrace value
    for (ssize_t curId = endId; curId >= startId; curId--)
      for (size_t a = 0; a < numAgents; a++)
      {
        // Load truncated importance weight
        const float truncatedImportanceWeight = _truncatedImportanceWeightBufferContiguous[curId * numAgents + a];

        // Load state value
        const float stateValue = _stateValueBufferContiguous[curId * numAgents + a];

        // Getting current reward, action, and state
        const float curReward = getScaledReward(_rewardBufferContiguous[curId * numAgents + a]);

        // Apply recursion
        retV[a] = stateValue + truncatedImportanceWeight * (curReward + _discountFactor * retV[a] - stateValue);

        // Store retrace value
        _retraceValueBufferContiguous[curId * numAgents + a] = retV[a];
      }
  }
}

size_t Agent::getTimeSequenceStartExpId(size_t expId)
{
  const size_t episodePos = _episodePosBuffer[expId];

  // Determine actual length of time sequence
  const size_t lookBack = std::min(_timeSequenceLength - 1, episodePos);

  if (lookBack > expId)
    // Return start of buffer if expId is part of a cut episode
    return 0;
  else
    // Return time sequence start
    return expId - lookBack;
}

void Agent::resetTimeSequence()
{
  for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
    _stateTimeSequence[a].clear();
}

std::vector<std::vector<float>> Agent::getTruncatedStateSequence(size_t expId, size_t agentId)
{
  // Getting starting expId
  size_t startId = getTimeSequenceStartExpId(expId);

  // Creating storage for the time sequence
  std::vector<std::vector<float>> timeSequence;

  // Now adding states, except for the initial one
  for (size_t e = startId + 1; e <= expId; e++)
    timeSequence.push_back(_stateBuffer[e][agentId]);

  // Lastly, adding truncated state
  timeSequence.push_back(_truncatedStateBuffer[expId][agentId]);

  return timeSequence;
}

void Agent::finalize()
{
  if (_mode != "Training") return;

  if (_experienceReplaySerialize == true)
    if (_k->_fileOutputEnabled)
      serializeExperienceReplay();

  _k->_logger->logInfo("Normal", "Waiting for pending agents to finish...\n");

  // Waiting for pending agents to finish
  bool agentsRemain = true;
  do
  {
    agentsRemain = false;
    for (size_t workerId = 0; workerId < _concurrentWorkers; workerId++)
      if (_isWorkerRunning[workerId] == true)
      {
        attendWorker(workerId);
        agentsRemain = true;
      }

    if (agentsRemain) KORALI_LISTEN(_workers);
  } while (agentsRemain == true);
}

void Agent::serializeExperienceReplay()
{
  _k->_logger->logInfo("Detailed", "Serializing Training State...\n");
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Creating JSON storage variable
  knlohmann::json stateJson;

  // Get number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Serializing agent's database into the JSON storage
  for (size_t i = 0; i < _stateBuffer.size(); i++)
  {
    stateJson["Experience Replay"][i]["Episode Id"] = _episodeIdBuffer[i];
    stateJson["Experience Replay"][i]["Episode Pos"] = _episodePosBuffer[i];
    stateJson["Experience Replay"][i]["State"] = _stateBuffer[i];
    stateJson["Experience Replay"][i]["Action"] = _actionBuffer[i];
    stateJson["Experience Replay"][i]["Importance Weight"] = _importanceWeightBuffer[i];
    stateJson["Experience Replay"][i]["Product Importance Weight"] = _productImportanceWeightBuffer[i];
    stateJson["Experience Replay"][i]["Is On Policy"] = _isOnPolicyBuffer[i];
    stateJson["Experience Replay"][i]["Truncated State"] = _truncatedStateBuffer[i];
    stateJson["Experience Replay"][i]["Truncated State Value"] = _truncatedStateValueBuffer[i];
    stateJson["Experience Replay"][i]["Termination"] = _terminationBuffer[i];

    for (size_t a = 0; a < numAgents; a++)
    {
      stateJson["Experience Replay"][i]["Reward"][a] = _rewardBufferContiguous[i * numAgents + a];
      stateJson["Experience Replay"][i]["State Value"][a] = _stateValueBufferContiguous[i * numAgents + a];
      stateJson["Experience Replay"][i]["Retrace Value"][a] = _retraceValueBufferContiguous[i * numAgents + a];
      stateJson["Experience Replay"][i]["Truncated Importance Weight"][a] = _truncatedImportanceWeightBufferContiguous[i * numAgents + a];
    }

    std::vector<float> expStateValue(numAgents, 0.0f);
    std::vector<std::vector<float>> expDistributionParameter(numAgents, std::vector<float>(_expPolicyBuffer[0][0].distributionParameters.size()));
    std::vector<size_t> expActionIdx(numAgents, 0);
    std::vector<std::vector<float>> expUnboundedAct(numAgents, std::vector<float>(_expPolicyBuffer[0][0].unboundedAction.size()));
    std::vector<std::vector<float>> expActProb(numAgents, std::vector<float>(_expPolicyBuffer[0][0].actionProbabilities.size()));
    std::vector<std::vector<size_t>> expAvailAct(numAgents, std::vector<size_t>(_expPolicyBuffer[0][0].availableActions.size()));

    std::vector<float> curStateValue(numAgents, 0.0f);
    std::vector<std::vector<float>> curDistributionParameter(numAgents, std::vector<float>(_curPolicyBuffer[0][0].distributionParameters.size()));
    std::vector<size_t> curActionIdx(numAgents, 0);
    std::vector<std::vector<float>> curUnboundedAct(numAgents, std::vector<float>(_curPolicyBuffer[0][0].unboundedAction.size()));
    std::vector<std::vector<float>> curActProb(numAgents, std::vector<float>(_curPolicyBuffer[0][0].actionProbabilities.size()));
    std::vector<std::vector<size_t>> curAvailAct(numAgents, std::vector<size_t>(_curPolicyBuffer[0][0].availableActions.size()));

    for (size_t a = 0; a < numAgents; a++)
    {
      expStateValue[a] = _expPolicyBuffer[i][a].stateValue;
      expDistributionParameter[a] = _expPolicyBuffer[i][a].distributionParameters;
      expActionIdx[a] = _expPolicyBuffer[i][a].actionIndex;
      expUnboundedAct[a] = _expPolicyBuffer[i][a].unboundedAction;
      expActProb[a] = _expPolicyBuffer[i][a].actionProbabilities;
      expAvailAct[a] = _expPolicyBuffer[i][a].availableActions;

      curStateValue[a] = _curPolicyBuffer[i][a].stateValue;
      curDistributionParameter[a] = _curPolicyBuffer[i][a].distributionParameters;
      curActionIdx[a] = _curPolicyBuffer[i][a].actionIndex;
      curUnboundedAct[a] = _curPolicyBuffer[i][a].unboundedAction;
      curActProb[a] = _curPolicyBuffer[i][a].actionProbabilities;
      curAvailAct[a] = _curPolicyBuffer[i][a].availableActions;
    }
    stateJson["Experience Replay"][i]["Experience Policy"]["State Value"] = expStateValue;
    stateJson["Experience Replay"][i]["Experience Policy"]["Distribution Parameters"] = expDistributionParameter;
    stateJson["Experience Replay"][i]["Experience Policy"]["Action Index"] = expActionIdx;
    stateJson["Experience Replay"][i]["Experience Policy"]["Unbounded Action"] = expUnboundedAct;
    stateJson["Experience Replay"][i]["Experience Policy"]["Action Probabilities"] = expActProb;
    stateJson["Experience Replay"][i]["Experience Policy"]["Available Actions"] = expAvailAct;

    stateJson["Experience Replay"][i]["Current Policy"]["State Value"] = curStateValue;
    stateJson["Experience Replay"][i]["Current Policy"]["Distribution Parameters"] = curDistributionParameter;
    stateJson["Experience Replay"][i]["Current Policy"]["Action Index"] = curActionIdx;
    stateJson["Experience Replay"][i]["Current Policy"]["Unbounded Action"] = curUnboundedAct;
    stateJson["Experience Replay"][i]["Current Policy"]["Action Probabilities"] = curActProb;
    stateJson["Experience Replay"][i]["Current Policy"]["Available Actions"] = curAvailAct;
  }

  // Serialize the optimizer
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->_optimizer->getConfiguration(stateJson["Optimizer"][p]);

  // If results directory doesn't exist, create it
  if (!dirExists(_k->_fileOutputPath)) mkdir(_k->_fileOutputPath);

  // Resolving file path
  std::string statePath = _k->_fileOutputPath + "/state.json";

  // Storing database to file
  _k->_logger->logInfo("Detailed", "Saving json..\n");
  if (saveJsonToFile(statePath.c_str(), stateJson) != 0)
    KORALI_LOG_ERROR("Could not serialize training state into file %s\n", statePath.c_str());
  _k->_logger->logInfo("Detailed", "Agent's Training State serialized\n");

  auto endTime = std::chrono::steady_clock::now();                                                                   // Profiling
  _sessionSerializationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationSerializationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling
}

void Agent::deserializeExperienceReplay()
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Creating JSON storage variable
  knlohmann::json stateJson;

  // Resolving file path
  std::string statePath = _k->_fileOutputPath + "/state.json";

  // Get number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Loading database from file
  _k->_logger->logInfo("Normal", "Loading previous run training state from file %s...\n", statePath.c_str());
  if (loadJsonFromFile(stateJson, statePath.c_str()) == false)
    KORALI_LOG_ERROR("Trying to resume training or test policy but could not find or deserialize agent's state from from file %s...\n", statePath.c_str());

  // Clearing existing database
  _stateBuffer.clear();
  _actionBuffer.clear();
  _retraceValueBufferContiguous.clear();
  _rewardBufferContiguous.clear();
  _stateValueBufferContiguous.clear();
  _importanceWeightBuffer.clear();
  _truncatedImportanceWeightBufferContiguous.clear();
  _truncatedStateValueBuffer.clear();
  _productImportanceWeightBuffer.clear();
  _truncatedStateBuffer.clear();
  _terminationBuffer.clear();
  _expPolicyBuffer.clear();
  _curPolicyBuffer.clear();
  _isOnPolicyBuffer.clear();
  _episodePosBuffer.clear();
  _episodeIdBuffer.clear();

  // Deserializing database from JSON to the agent's state
  for (size_t i = 0; i < stateJson["Experience Replay"].size(); i++)
  {
    _episodeIdBuffer.add(stateJson["Experience Replay"][i]["Episode Id"].get<size_t>());
    _episodePosBuffer.add(stateJson["Experience Replay"][i]["Episode Pos"].get<size_t>());
    _stateBuffer.add(stateJson["Experience Replay"][i]["State"].get<std::vector<std::vector<float>>>());
    _actionBuffer.add(stateJson["Experience Replay"][i]["Action"].get<std::vector<std::vector<float>>>());
    _importanceWeightBuffer.add(stateJson["Experience Replay"][i]["Importance Weight"].get<std::vector<float>>());

    _productImportanceWeightBuffer.add(stateJson["Experience Replay"][i]["Product Importance Weight"].get<float>());
    _isOnPolicyBuffer.add(stateJson["Experience Replay"][i]["Is On Policy"].get<std::vector<char>>());
    _truncatedStateBuffer.add(stateJson["Experience Replay"][i]["Truncated State"].get<std::vector<std::vector<float>>>());
    _truncatedStateValueBuffer.add(stateJson["Experience Replay"][i]["Truncated State Value"].get<std::vector<float>>());
    _terminationBuffer.add(stateJson["Experience Replay"][i]["Termination"].get<termination_t>());

    for (size_t a = 0; a < numAgents; a++)
    {
      _rewardBufferContiguous.add(stateJson["Experience Replay"][i]["Reward"][a].get<float>());
      _stateValueBufferContiguous.add(stateJson["Experience Replay"][i]["State Value"][a].get<float>());
      _retraceValueBufferContiguous.add(stateJson["Experience Replay"][i]["Retrace Value"][a].get<float>());
      _truncatedImportanceWeightBufferContiguous.add(stateJson["Experience Replay"][i]["Truncated Importance Weight"][a].get<float>());
    }

    std::vector<policy_t> expPolicy(numAgents);
    std::vector<policy_t> curPolicy(numAgents);
    for (size_t a = 0; a < numAgents; a++)
    {
      expPolicy[a].stateValue = stateJson["Experience Replay"][i]["Experience Policy"]["State Value"][a].get<float>();
      expPolicy[a].distributionParameters = stateJson["Experience Replay"][i]["Experience Policy"]["Distribution Parameters"][a].get<std::vector<float>>();
      expPolicy[a].unboundedAction = stateJson["Experience Replay"][i]["Experience Policy"]["Unbounded Action"][a].get<std::vector<float>>();
      expPolicy[a].actionIndex = stateJson["Experience Replay"][i]["Experience Policy"]["Action Index"][a].get<size_t>();
      expPolicy[a].actionProbabilities = stateJson["Experience Replay"][i]["Experience Policy"]["Action Probabilities"][a].get<std::vector<float>>();
      expPolicy[a].availableActions = stateJson["Experience Replay"][i]["Experience Policy"]["Available Actions"][a].get<std::vector<size_t>>();

      curPolicy[a].stateValue = stateJson["Experience Replay"][i]["Current Policy"]["State Value"][a].get<float>();
      curPolicy[a].distributionParameters = stateJson["Experience Replay"][i]["Current Policy"]["Distribution Parameters"][a].get<std::vector<float>>();
      curPolicy[a].actionIndex = stateJson["Experience Replay"][i]["Current Policy"]["Action Index"][a].get<size_t>();
      curPolicy[a].unboundedAction = stateJson["Experience Replay"][i]["Current Policy"]["Unbounded Action"][a].get<std::vector<float>>();
      curPolicy[a].actionProbabilities = stateJson["Experience Replay"][i]["Current Policy"]["Action Probabilities"][a].get<std::vector<float>>();
      curPolicy[a].availableActions = stateJson["Experience Replay"][i]["Current Policy"]["Available Actions"][a].get<std::vector<size_t>>();
    }
    _expPolicyBuffer.add(expPolicy);
    _curPolicyBuffer.add(curPolicy);
  }

  // Deserialize the optimizer
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->_optimizer->setConfiguration(stateJson["Optimizer"][p]);

  auto endTime = std::chrono::steady_clock::now();                                                                         // Profiling
  double deserializationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9; // Profiling
  _k->_logger->logInfo("Normal", "Took %fs to deserialize training state.\n", deserializationTime);
}

void Agent::printGenerationAfter()
{
  if (_mode == "Training")
  {
    _k->_logger->logInfo("Normal", "Experience Replay Statistics:\n");
    _k->_logger->logInfo("Normal", " + Experience Memory Size:      %lu/%lu\n", _stateBuffer.size(), _experienceReplayMaximumSize);
    if (_maxEpisodes > 0)
      _k->_logger->logInfo("Normal", " + Total Episodes Count:        %lu/%lu\n", _currentEpisode, _maxEpisodes);
    else
      _k->_logger->logInfo("Normal", " + Total Episodes Count:        %lu\n", _currentEpisode);

    if (_maxExperiences > 0)
      _k->_logger->logInfo("Normal", " + Total Experience Count:      %lu/%lu\n", _experienceCount, _maxExperiences);
    else
      _k->_logger->logInfo("Normal", " + Total Experience Count:      %lu\n", _experienceCount);

    _k->_logger->logInfo("Normal", "Training Statistics:\n");
    if (_maxPolicyUpdates > 0)
      _k->_logger->logInfo("Normal", " + Policy Update Count:         %lu/%lu\n", _policyUpdateCount, _maxPolicyUpdates);
    else
      _k->_logger->logInfo("Normal", " + Policy Update Count:         %lu\n", _policyUpdateCount);

    size_t numPolicies = _problem->_policiesPerEnvironment;
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
    {
      _k->_logger->logInfo("Normal", "Off-Policy Statistics for policy %lu: \n", a);
      _k->_logger->logInfo("Normal", " + Count (Ratio/Target):        %lu/%lu (%.3f/%.3f)\n", numPolicies > 1 ? _experienceReplayOffPolicyCount[a] : _experienceReplayOffPolicyCount[a] / _problem->_agentsPerEnvironment, _stateBuffer.size(), _experienceReplayOffPolicyRatio[a], _experienceReplayOffPolicyTarget);
      _k->_logger->logInfo("Normal", " + Importance Weight Cutoff:    [%.3f, %.3f]\n", 1.0f / _experienceReplayOffPolicyCurrentCutoff, _experienceReplayOffPolicyCurrentCutoff);
      _k->_logger->logInfo("Normal", " + REFER Beta Factor:           %f\n", _experienceReplayOffPolicyREFERCurrentBeta[a]);
      _k->_logger->logInfo("Normal", " + Latest Return for agent %lu:               %f\n", a, _trainingLastReturn[a]);
      _k->_logger->logInfo("Normal", " + %lu-Episode Average Return for agent %lu:  %f\n", _trainingAverageDepth, a, _trainingAverageReturn[a]);
      _k->_logger->logInfo("Normal", " + Best Return for agent %lu:                 %f (%lu)\n", a, _trainingBestReturn[a], _trainingBestEpisodeId[a]);
    }

    if (_rewardRescalingEnabled)
      _k->_logger->logInfo("Normal", " + Reward Rescaling: N(0.0, %.3e)\n", _rewardRescalingSigma);

    if (_problem->_testingFrequency > 0)
    {
      _k->_logger->logInfo("Normal", "Testing Statistics:\n");
      _k->_logger->logInfo("Normal", " + Best Average Return: %f (%lu)\n", _testingBestAverageReturn, _testingBestEpisodeId);
      _k->_logger->logInfo("Normal", " + Latest Average (Worst / Best) Return: %f (%f / %f)\n", _testingAverageReturn, _testingWorstReturn, _testingBestReturn);
    }

    if (_policyUpdateCount != 0)
    {
      printInformation();
      _k->_logger->logInfo("Normal", " + Current Learning Rate:           %.3e\n", _currentLearningRate);
    }

    _k->_logger->logInfo("Detailed", "Profiling Information:                    [Generation] - [Session]\n");
    _k->_logger->logInfo("Detailed", " + Experience Serialization Time:         [%5.3fs] - [%3.3fs]\n", _generationSerializationTime / 1.0e+9, _sessionSerializationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Worker Attending Time:                 [%5.3fs] - [%3.3fs]\n", _generationWorkerAttendingTime / 1.0e+9, _sessionWorkerAttendingTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Worker Computation Time:           [%5.3fs] - [%3.3fs]\n", _generationWorkerComputationTime / 1.0e+9, _sessionWorkerComputationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Worker Communication/Wait Time:    [%5.3fs] - [%3.3fs]\n", _generationWorkerCommunicationTime / 1.0e+9, _sessionWorkerCommunicationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Policy Evaluation Time:            [%5.3fs] - [%3.3fs]\n", _generationPolicyEvaluationTime / 1.0e+9, _sessionPolicyEvaluationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Policy Update Time:                    [%5.3fs] - [%3.3fs]\n", _generationPolicyUpdateTime / 1.0e+9, _sessionPolicyUpdateTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Running Time:                          [%5.3fs] - [%3.3fs]\n", _generationRunningTime / 1.0e+9, _sessionRunningTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + [I/O] Result File Saving Time:         [%5.3fs]\n", _k->_resultSavingTime / 1.0e+9);
  }

  if (_mode == "Testing")
  {
    _k->_logger->logInfo("Normal", "Testing Results:\n");
    for (size_t sampleId = 0; sampleId < _testingSampleIds.size(); sampleId++)
    {
      _k->_logger->logInfo("Normal", " + Sample %lu:\n", _testingSampleIds[sampleId]);
      _k->_logger->logInfo("Normal", "   + (Average) Return:           %f\n", _testingReturn[sampleId]);
    }
  }
}

void Agent::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Policy", "Parameter Count"))
 {
 try { _policyParameterCount = js["Policy"]["Parameter Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Policy']['Parameter Count']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Count");
 }

 if (isDefined(js, "Action Lower Bounds"))
 {
 try { _actionLowerBounds = js["Action Lower Bounds"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Action Lower Bounds']\n%s", e.what()); } 
   eraseValue(js, "Action Lower Bounds");
 }

 if (isDefined(js, "Action Upper Bounds"))
 {
 try { _actionUpperBounds = js["Action Upper Bounds"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Action Upper Bounds']\n%s", e.what()); } 
   eraseValue(js, "Action Upper Bounds");
 }

 if (isDefined(js, "Current Episode"))
 {
 try { _currentEpisode = js["Current Episode"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Current Episode']\n%s", e.what()); } 
   eraseValue(js, "Current Episode");
 }

 if (isDefined(js, "Training", "Return History"))
 {
 try { _trainingReturnHistory = js["Training"]["Return History"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Return History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Return History");
 }

 if (isDefined(js, "Training", "Discounted Return History"))
 {
 try { _trainingDiscountedReturnHistory = js["Training"]["Discounted Return History"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Discounted Return History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Discounted Return History");
 }

 if (isDefined(js, "Training", "Experience History"))
 {
 try { _trainingExperienceHistory = js["Training"]["Experience History"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Experience History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Experience History");
 }

 if (isDefined(js, "Testing", "Average Return History"))
 {
 try { _testingAverageReturnHistory = js["Testing"]["Average Return History"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Average Return History']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Average Return History");
 }

 if (isDefined(js, "Training", "Average Return"))
 {
 try { _trainingAverageReturn = js["Training"]["Average Return"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Return']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Return");
 }

 if (isDefined(js, "Training", "Last Return"))
 {
 try { _trainingLastReturn = js["Training"]["Last Return"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Last Return']\n%s", e.what()); } 
   eraseValue(js, "Training", "Last Return");
 }

 if (isDefined(js, "Training", "Best Return"))
 {
 try { _trainingBestReturn = js["Training"]["Best Return"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Best Return']\n%s", e.what()); } 
   eraseValue(js, "Training", "Best Return");
 }

 if (isDefined(js, "Training", "Best Episode Id"))
 {
 try { _trainingBestEpisodeId = js["Training"]["Best Episode Id"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Best Episode Id']\n%s", e.what()); } 
   eraseValue(js, "Training", "Best Episode Id");
 }

 if (isDefined(js, "Training", "Current Policies"))
 {
 _trainingCurrentPolicies = js["Training"]["Current Policies"].get<knlohmann::json>();

   eraseValue(js, "Training", "Current Policies");
 }

 if (isDefined(js, "Training", "Best Policies"))
 {
 _trainingBestPolicies = js["Training"]["Best Policies"].get<knlohmann::json>();

   eraseValue(js, "Training", "Best Policies");
 }

 if (isDefined(js, "Testing", "Return"))
 {
 try { _testingReturn = js["Testing"]["Return"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Return']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Return");
 }

 if (isDefined(js, "Testing", "Best Return"))
 {
 try { _testingBestReturn = js["Testing"]["Best Return"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Return']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Return");
 }

 if (isDefined(js, "Testing", "Worst Return"))
 {
 try { _testingWorstReturn = js["Testing"]["Worst Return"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Worst Return']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Worst Return");
 }

 if (isDefined(js, "Testing", "Best Episode Id"))
 {
 try { _testingBestEpisodeId = js["Testing"]["Best Episode Id"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Episode Id']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Episode Id");
 }

 if (isDefined(js, "Testing", "Candidate Count"))
 {
 try { _testingCandidateCount = js["Testing"]["Candidate Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Candidate Count']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Candidate Count");
 }

 if (isDefined(js, "Testing", "Average Return"))
 {
 try { _testingAverageReturn = js["Testing"]["Average Return"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Average Return']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Average Return");
 }

 if (isDefined(js, "Testing", "Best Average Return"))
 {
 try { _testingBestAverageReturn = js["Testing"]["Best Average Return"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Average Return']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Average Return");
 }

 if (isDefined(js, "Testing", "Best Policies"))
 {
 _testingBestPolicies = js["Testing"]["Best Policies"].get<knlohmann::json>();

   eraseValue(js, "Testing", "Best Policies");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Count"))
 {
 try { _experienceReplayOffPolicyCount = js["Experience Replay"]["Off Policy"]["Count"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Count']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Count");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Ratio"))
 {
 try { _experienceReplayOffPolicyRatio = js["Experience Replay"]["Off Policy"]["Ratio"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Ratio']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Ratio");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Current Cutoff"))
 {
 try { _experienceReplayOffPolicyCurrentCutoff = js["Experience Replay"]["Off Policy"]["Current Cutoff"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Current Cutoff']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Current Cutoff");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "REFER Current Beta"))
 {
 try { _experienceReplayOffPolicyREFERCurrentBeta = js["Experience Replay"]["Off Policy"]["REFER Current Beta"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['REFER Current Beta']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "REFER Current Beta");
 }

 if (isDefined(js, "Current Learning Rate"))
 {
 try { _currentLearningRate = js["Current Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Current Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Current Learning Rate");
 }

 if (isDefined(js, "Policy Update Count"))
 {
 try { _policyUpdateCount = js["Policy Update Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Policy Update Count']\n%s", e.what()); } 
   eraseValue(js, "Policy Update Count");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Experience Count"))
 {
 try { _experienceCount = js["Experience Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Count']\n%s", e.what()); } 
   eraseValue(js, "Experience Count");
 }

 if (isDefined(js, "Reward", "Rescaling", "Sigma"))
 {
 try { _rewardRescalingSigma = js["Reward"]["Rescaling"]["Sigma"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Sigma']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Sigma");
 }

 if (isDefined(js, "Reward", "Rescaling", "Sum Squared Rewards"))
 {
 try { _rewardRescalingSumSquaredRewards = js["Reward"]["Rescaling"]["Sum Squared Rewards"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Sum Squared Rewards']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Sum Squared Rewards");
 }

 if (isDefined(js, "State Rescaling", "Means"))
 {
 try { _stateRescalingMeans = js["State Rescaling"]["Means"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Means']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Means");
 }

 if (isDefined(js, "State Rescaling", "Sigmas"))
 {
 try { _stateRescalingSigmas = js["State Rescaling"]["Sigmas"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Sigmas']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Sigmas");
 }

 if (isDefined(js, "Effective Minibatch Size"))
 {
 try { _effectiveMinibatchSize = js["Effective Minibatch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Effective Minibatch Size']\n%s", e.what()); } 
   eraseValue(js, "Effective Minibatch Size");
 }

 if (isDefined(js, "Mode"))
 {
 try { _mode = js["Mode"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Mode']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_mode == "Training") validOption = true; 
 if (_mode == "Testing") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mode'] required by agent.\n", _mode.c_str()); 
}
   eraseValue(js, "Mode");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mode'] required by agent.\n"); 

 if (isDefined(js, "Testing", "Sample Ids"))
 {
 try { _testingSampleIds = js["Testing"]["Sample Ids"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Sample Ids']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Sample Ids");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing']['Sample Ids'] required by agent.\n"); 

 if (isDefined(js, "Testing", "Current Policies"))
 {
 _testingCurrentPolicies = js["Testing"]["Current Policies"].get<knlohmann::json>();

   eraseValue(js, "Testing", "Current Policies");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing']['Current Policies'] required by agent.\n"); 

 if (isDefined(js, "Training", "Average Depth"))
 {
 try { _trainingAverageDepth = js["Training"]["Average Depth"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Depth']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Depth");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Training']['Average Depth'] required by agent.\n"); 

 if (isDefined(js, "Concurrent Workers"))
 {
 try { _concurrentWorkers = js["Concurrent Workers"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Concurrent Workers']\n%s", e.what()); } 
   eraseValue(js, "Concurrent Workers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Concurrent Workers'] required by agent.\n"); 

 if (isDefined(js, "Episodes Per Generation"))
 {
 try { _episodesPerGeneration = js["Episodes Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Episodes Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Episodes Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Episodes Per Generation'] required by agent.\n"); 

 if (isDefined(js, "Mini Batch", "Size"))
 {
 try { _miniBatchSize = js["Mini Batch"]["Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Mini Batch']['Size']\n%s", e.what()); } 
   eraseValue(js, "Mini Batch", "Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mini Batch']['Size'] required by agent.\n"); 

 if (isDefined(js, "Time Sequence Length"))
 {
 try { _timeSequenceLength = js["Time Sequence Length"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Time Sequence Length']\n%s", e.what()); } 
   eraseValue(js, "Time Sequence Length");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Time Sequence Length'] required by agent.\n"); 

 if (isDefined(js, "Learning Rate"))
 {
 try { _learningRate = js["Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate'] required by agent.\n"); 

 if (isDefined(js, "L2 Regularization", "Enabled"))
 {
 try { _l2RegularizationEnabled = js["L2 Regularization"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['L2 Regularization']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "L2 Regularization", "Importance"))
 {
 try { _l2RegularizationImportance = js["L2 Regularization"]["Importance"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['L2 Regularization']['Importance']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Importance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Importance'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Hidden Layers"))
 {
 _neuralNetworkHiddenLayers = js["Neural Network"]["Hidden Layers"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Hidden Layers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Hidden Layers'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Optimizer"))
 {
 try { _neuralNetworkOptimizer = js["Neural Network"]["Optimizer"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Neural Network']['Optimizer']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Optimizer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Optimizer'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Engine"))
 {
 try { _neuralNetworkEngine = js["Neural Network"]["Engine"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Neural Network']['Engine']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Engine");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Engine'] required by agent.\n"); 

 if (isDefined(js, "Discount Factor"))
 {
 try { _discountFactor = js["Discount Factor"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Discount Factor']\n%s", e.what()); } 
   eraseValue(js, "Discount Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Discount Factor'] required by agent.\n"); 

 if (isDefined(js, "Importance Weight Truncation Level"))
 {
 try { _importanceWeightTruncationLevel = js["Importance Weight Truncation Level"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Importance Weight Truncation Level']\n%s", e.what()); } 
   eraseValue(js, "Importance Weight Truncation Level");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Importance Weight Truncation Level'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Serialize"))
 {
 try { _experienceReplaySerialize = js["Experience Replay"]["Serialize"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Serialize']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Serialize");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Serialize'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Start Size"))
 {
 try { _experienceReplayStartSize = js["Experience Replay"]["Start Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Start Size']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Start Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Start Size'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Maximum Size"))
 {
 try { _experienceReplayMaximumSize = js["Experience Replay"]["Maximum Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Maximum Size']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Maximum Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Maximum Size'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Cutoff Scale"))
 {
 try { _experienceReplayOffPolicyCutoffScale = js["Experience Replay"]["Off Policy"]["Cutoff Scale"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Cutoff Scale']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Cutoff Scale");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Cutoff Scale'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Target"))
 {
 try { _experienceReplayOffPolicyTarget = js["Experience Replay"]["Off Policy"]["Target"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Target']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Target");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Target'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Annealing Rate"))
 {
 try { _experienceReplayOffPolicyAnnealingRate = js["Experience Replay"]["Off Policy"]["Annealing Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Annealing Rate']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Annealing Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Annealing Rate'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "REFER Beta"))
 {
 try { _experienceReplayOffPolicyREFERBeta = js["Experience Replay"]["Off Policy"]["REFER Beta"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['REFER Beta']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "REFER Beta");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['REFER Beta'] required by agent.\n"); 

 if (isDefined(js, "Experiences Between Policy Updates"))
 {
 try { _experiencesBetweenPolicyUpdates = js["Experiences Between Policy Updates"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experiences Between Policy Updates']\n%s", e.what()); } 
   eraseValue(js, "Experiences Between Policy Updates");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experiences Between Policy Updates'] required by agent.\n"); 

 if (isDefined(js, "State Rescaling", "Enabled"))
 {
 try { _stateRescalingEnabled = js["State Rescaling"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['State Rescaling']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Reward", "Rescaling", "Enabled"))
 {
 try { _rewardRescalingEnabled = js["Reward"]["Rescaling"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward']['Rescaling']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Relationship"))
 {
 try { _multiAgentRelationship = js["Multi Agent Relationship"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Relationship']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_multiAgentRelationship == "Individual") validOption = true; 
 if (_multiAgentRelationship == "Cooperation") validOption = true; 
 if (_multiAgentRelationship == "Competition") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Multi Agent Relationship'] required by agent.\n", _multiAgentRelationship.c_str()); 
}
   eraseValue(js, "Multi Agent Relationship");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Relationship'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Correlation"))
 {
 try { _multiAgentCorrelation = js["Multi Agent Correlation"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Correlation']\n%s", e.what()); } 
   eraseValue(js, "Multi Agent Correlation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Correlation'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Sampling"))
 {
 try { _multiAgentSampling = js["Multi Agent Sampling"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Sampling']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_multiAgentSampling == "Tuple") validOption = true; 
 if (_multiAgentSampling == "Baseline") validOption = true; 
 if (_multiAgentSampling == "Experience") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Multi Agent Sampling'] required by agent.\n", _multiAgentSampling.c_str()); 
}
   eraseValue(js, "Multi Agent Sampling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Sampling'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Episodes"))
 {
 try { _maxEpisodes = js["Termination Criteria"]["Max Episodes"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Episodes']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Episodes");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Episodes'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Experiences"))
 {
 try { _maxExperiences = js["Termination Criteria"]["Max Experiences"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Experiences']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Experiences");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Experiences'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Policy Updates"))
 {
 try { _maxPolicyUpdates = js["Termination Criteria"]["Max Policy Updates"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Policy Updates']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Policy Updates");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Policy Updates'] required by agent.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Solver::setConfiguration(js);
 _type = "agent";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: agent: \n%s\n", js.dump(2).c_str());
} 

void Agent::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Mode"] = _mode;
   js["Testing"]["Sample Ids"] = _testingSampleIds;
   js["Testing"]["Current Policies"] = _testingCurrentPolicies;
   js["Training"]["Average Depth"] = _trainingAverageDepth;
   js["Concurrent Workers"] = _concurrentWorkers;
   js["Episodes Per Generation"] = _episodesPerGeneration;
   js["Mini Batch"]["Size"] = _miniBatchSize;
   js["Time Sequence Length"] = _timeSequenceLength;
   js["Learning Rate"] = _learningRate;
   js["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
   js["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
   js["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
   js["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
   js["Neural Network"]["Engine"] = _neuralNetworkEngine;
   js["Discount Factor"] = _discountFactor;
   js["Importance Weight Truncation Level"] = _importanceWeightTruncationLevel;
   js["Experience Replay"]["Serialize"] = _experienceReplaySerialize;
   js["Experience Replay"]["Start Size"] = _experienceReplayStartSize;
   js["Experience Replay"]["Maximum Size"] = _experienceReplayMaximumSize;
   js["Experience Replay"]["Off Policy"]["Cutoff Scale"] = _experienceReplayOffPolicyCutoffScale;
   js["Experience Replay"]["Off Policy"]["Target"] = _experienceReplayOffPolicyTarget;
   js["Experience Replay"]["Off Policy"]["Annealing Rate"] = _experienceReplayOffPolicyAnnealingRate;
   js["Experience Replay"]["Off Policy"]["REFER Beta"] = _experienceReplayOffPolicyREFERBeta;
   js["Experiences Between Policy Updates"] = _experiencesBetweenPolicyUpdates;
   js["State Rescaling"]["Enabled"] = _stateRescalingEnabled;
   js["Reward"]["Rescaling"]["Enabled"] = _rewardRescalingEnabled;
   js["Multi Agent Relationship"] = _multiAgentRelationship;
   js["Multi Agent Correlation"] = _multiAgentCorrelation;
   js["Multi Agent Sampling"] = _multiAgentSampling;
   js["Termination Criteria"]["Max Episodes"] = _maxEpisodes;
   js["Termination Criteria"]["Max Experiences"] = _maxExperiences;
   js["Termination Criteria"]["Max Policy Updates"] = _maxPolicyUpdates;
   js["Policy"]["Parameter Count"] = _policyParameterCount;
   js["Action Lower Bounds"] = _actionLowerBounds;
   js["Action Upper Bounds"] = _actionUpperBounds;
   js["Current Episode"] = _currentEpisode;
   js["Training"]["Return History"] = _trainingReturnHistory;
   js["Training"]["Discounted Return History"] = _trainingDiscountedReturnHistory;
   js["Training"]["Experience History"] = _trainingExperienceHistory;
   js["Testing"]["Average Return History"] = _testingAverageReturnHistory;
   js["Training"]["Average Return"] = _trainingAverageReturn;
   js["Training"]["Last Return"] = _trainingLastReturn;
   js["Training"]["Best Return"] = _trainingBestReturn;
   js["Training"]["Best Episode Id"] = _trainingBestEpisodeId;
   js["Training"]["Current Policies"] = _trainingCurrentPolicies;
   js["Training"]["Best Policies"] = _trainingBestPolicies;
   js["Testing"]["Return"] = _testingReturn;
   js["Testing"]["Best Return"] = _testingBestReturn;
   js["Testing"]["Worst Return"] = _testingWorstReturn;
   js["Testing"]["Best Episode Id"] = _testingBestEpisodeId;
   js["Testing"]["Candidate Count"] = _testingCandidateCount;
   js["Testing"]["Average Return"] = _testingAverageReturn;
   js["Testing"]["Best Average Return"] = _testingBestAverageReturn;
   js["Testing"]["Best Policies"] = _testingBestPolicies;
   js["Experience Replay"]["Off Policy"]["Count"] = _experienceReplayOffPolicyCount;
   js["Experience Replay"]["Off Policy"]["Ratio"] = _experienceReplayOffPolicyRatio;
   js["Experience Replay"]["Off Policy"]["Current Cutoff"] = _experienceReplayOffPolicyCurrentCutoff;
   js["Experience Replay"]["Off Policy"]["REFER Current Beta"] = _experienceReplayOffPolicyREFERCurrentBeta;
   js["Current Learning Rate"] = _currentLearningRate;
   js["Policy Update Count"] = _policyUpdateCount;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Experience Count"] = _experienceCount;
   js["Reward"]["Rescaling"]["Sigma"] = _rewardRescalingSigma;
   js["Reward"]["Rescaling"]["Sum Squared Rewards"] = _rewardRescalingSumSquaredRewards;
   js["State Rescaling"]["Means"] = _stateRescalingMeans;
   js["State Rescaling"]["Sigmas"] = _stateRescalingSigmas;
   js["Effective Minibatch Size"] = _effectiveMinibatchSize;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void Agent::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Episodes Per Generation\": 1, \"Concurrent Workers\": 1, \"Discount Factor\": 0.995, \"Time Sequence Length\": 1, \"Importance Weight Truncation Level\": 1.0, \"Multi Agent Relationship\": \"Individual\", \"Multi Agent Correlation\": false, \"Multi Agent Sampling\": \"Tuple\", \"State Rescaling\": {\"Enabled\": false}, \"Reward\": {\"Rescaling\": {\"Enabled\": false}}, \"Mini Batch\": {\"Size\": 256}, \"L2 Regularization\": {\"Enabled\": false, \"Importance\": 0.0001}, \"Training\": {\"Average Depth\": 100, \"Current Policies\": {}, \"Best Policies\": {}}, \"Testing\": {\"Sample Ids\": [], \"Current Policies\": {}, \"Best Policies\": {}}, \"Termination Criteria\": {\"Max Episodes\": 0, \"Max Experiences\": 0, \"Max Policy Updates\": 0}, \"Experience Replay\": {\"Serialize\": true, \"Off Policy\": {\"Cutoff Scale\": 4.0, \"Target\": 0.1, \"REFER Beta\": 0.3, \"Annealing Rate\": 0.0}}, \"Uniform Generator\": {\"Name\": \"Agent / Uniform Generator\", \"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Agent::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool Agent::checkTermination()
{
 bool hasFinished = false;

 if ((_mode == "Training") && (_maxEpisodes > 0) && (_currentEpisode >= _maxEpisodes))
 {
  _terminationCriteria.push_back("agent['Max Episodes'] = " + std::to_string(_maxEpisodes) + ".");
  hasFinished = true;
 }

 if ((_mode == "Training") && (_maxExperiences > 0) && (_experienceCount >= _maxExperiences))
 {
  _terminationCriteria.push_back("agent['Max Experiences'] = " + std::to_string(_maxExperiences) + ".");
  hasFinished = true;
 }

 if ((_mode == "Training") && (_maxPolicyUpdates > 0) && (_policyUpdateCount >= _maxPolicyUpdates))
 {
  _terminationCriteria.push_back("agent['Max Policy Updates'] = " + std::to_string(_maxPolicyUpdates) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;
