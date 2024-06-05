/** \namespace solver
* @brief Namespace declaration for modules of type: solver.
*/

/** \file
* @brief Header file for module: Agent.
*/

/** \dir solver/agent
* @brief Contains code, documentation, and scripts for module: Agent.
*/

#pragma once

#include "auxiliar/cbuffer.hpp"
#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"
#include "modules/solver/deepSupervisor/deepSupervisor.hpp"
#include "sample/sample.hpp"
#include <algorithm> // std::shuffle
#include <random>

namespace korali
{
namespace solver
{
;

/**
 * @brief This enumerator details all possible termination statuses for a given episode's experience
 */
enum termination_t
{
  /**
   * @brief The experience is non-terminal
   */
  e_nonTerminal = 0,

  /**
   * @brief This is the terminal experience in a normally executed episode
   */
  e_terminal = 1,

  /**
   * @brief This is the terminal experience in a truncated episode.
   *        (i.e., should have continued, but it was artificially truncated to limit running time)
   */
  e_truncated = 2
};

/**
 * @brief Structure to store policy information
 */
struct policy_t
{
  /**
   * @brief Contains state value (V) estimation for the given state / policy combination
   */
  float stateValue;

  /**
   * @brief Contains the parameters that define the policy distribution used to produced the action.
   *        For continuous policies, it depends on the distribution selected.
   *        For discrete policies, it contains the categorical probability of every action.
   */
  std::vector<float> distributionParameters;

  /**
   * @brief [Discrete] Stores the index of the selected experience
   */
  size_t actionIndex;

  /**
   * @brief [Discrete] Stores the action probabilities of the categorial distribution.
   */
  std::vector<float> actionProbabilities;

  /**
   * @brief [Discrete] Flags the actions that are available at the current state.
   */
  std::vector<size_t> availableActions;

  /**
   * @brief [Continuous] Stores the Unbounded Actions of the Squashed Normal Policy Distribution
   */
  std::vector<float> unboundedAction;
};

/**
* @brief Class declaration for module: Agent.
*/
class Agent : public Solver
{
  public: 
  /**
  * @brief Specifies the operation mode for the agent.
  */
   std::string _mode;
  /**
  * @brief A vector with the identifiers for the samples to test the hyperparameters with.
  */
   std::vector<size_t> _testingSampleIds;
  /**
  * @brief The current hyperparameters of the policies to test.
  */
   knlohmann::json _testingCurrentPolicies;
  /**
  * @brief Specifies the depth of the running training average to report.
  */
   size_t _trainingAverageDepth;
  /**
  * @brief Indicates the number of concurrent environments to use to collect experiences.
  */
   size_t _concurrentWorkers;
  /**
  * @brief Number of reinforcement learning episodes per Korali generation (checkpoints are generated between generations).
  */
   size_t _episodesPerGeneration;
  /**
  * @brief The number of experiences to randomly select to train the neural network(s) with.
  */
   size_t _miniBatchSize;
  /**
  * @brief Indicates the number of contiguous experiences to pass to the NN for learning. This is only useful when using recurrent NNs.
  */
   size_t _timeSequenceLength;
  /**
  * @brief The initial learning rate to use for the NN hyperparameter optimization.
  */
   float _learningRate;
  /**
  * @brief Boolean to determine if l2 regularization will be applied to the neural networks.
  */
   int _l2RegularizationEnabled;
  /**
  * @brief Coefficient for l2 regularization.
  */
   float _l2RegularizationImportance;
  /**
  * @brief Indicates the configuration of the hidden neural network layers.
  */
   knlohmann::json _neuralNetworkHiddenLayers;
  /**
  * @brief Indicates the optimizer algorithm to update the NN hyperparameters.
  */
   std::string _neuralNetworkOptimizer;
  /**
  * @brief Specifies which Neural Network backend to use.
  */
   std::string _neuralNetworkEngine;
  /**
  * @brief Represents the discount factor to weight future experiences.
  */
   float _discountFactor;
  /**
  * @brief Represents the discount factor to weight future experiences.
  */
   float _importanceWeightTruncationLevel;
  /**
  * @brief Indicates whether to serialize and store the experience replay after each generation. Disabling will reduce I/O overheads but will disable the checkpoint/resume function.
  */
   int _experienceReplaySerialize;
  /**
  * @brief The minimum number of experiences before learning starts.
  */
   size_t _experienceReplayStartSize;
  /**
  * @brief The size of the replay memory. If this number is exceeded, experiences are deleted.
  */
   size_t _experienceReplayMaximumSize;
  /**
  * @brief Initial Cut-Off to classify experiences as on- or off-policy. (c_max in https://arxiv.org/abs/1807.05827)
  */
   float _experienceReplayOffPolicyCutoffScale;
  /**
  * @brief Target fraction of off-policy experiences in the replay memory. (D in https://arxiv.org/abs/1807.05827)
  */
   float _experienceReplayOffPolicyTarget;
  /**
  * @brief Annealing rate for Off Policy Cutoff Scale and Learning Rate. (A in https://arxiv.org/abs/1807.05827)
  */
   float _experienceReplayOffPolicyAnnealingRate;
  /**
  * @brief Initial value for the penalisation coefficient for off-policiness. (beta in https://arxiv.org/abs/1807.05827)
  */
   float _experienceReplayOffPolicyREFERBeta;
  /**
  * @brief The number of experiences to receive before training/updating (real number, may be less than < 1.0, for more than one update per experience).
  */
   float _experiencesBetweenPolicyUpdates;
  /**
  * @brief Determines whether to normalize the states, such that they have mean 0 and standard deviation 1 (done only once after the initial exploration phase).
  */
   int _stateRescalingEnabled;
  /**
  * @brief Determines whether to normalize the rewards, such that they have mean 0 and standard deviation 1
  */
   int _rewardRescalingEnabled;
  /**
  * @brief Specifies whether we are in an individual setting or collaborator setting.
  */
   std::string _multiAgentRelationship;
  /**
  * @brief Specifies whether we take into account the dependencies of the agents or not.
  */
   int _multiAgentCorrelation;
  /**
  * @brief Specifies how to sample the minibatch.
  */
   std::string _multiAgentSampling;
  /**
  * @brief [Internal Use] Stores the number of parameters that determine the probability distribution for the current state sequence.
  */
   size_t _policyParameterCount;
  /**
  * @brief [Internal Use] Lower bounds for actions.
  */
   std::vector<float> _actionLowerBounds;
  /**
  * @brief [Internal Use] Upper bounds for actions.
  */
   std::vector<float> _actionUpperBounds;
  /**
  * @brief [Internal Use] Indicates the current episode being processed.
  */
   size_t _currentEpisode;
  /**
  * @brief [Internal Use] Keeps a history of all training episode returns.
  */
   std::vector<std::vector<float>> _trainingReturnHistory;
  /**
  * @brief [Internal Use] Keeps a history of all training episode discounted returns.
  */
   std::vector<std::vector<float>> _trainingDiscountedReturnHistory;
  /**
  * @brief [Internal Use] Keeps a history of all training episode experience counts.
  */
   std::vector<size_t> _trainingExperienceHistory;
  /**
  * @brief [Internal Use] Keeps a history of all training episode returns.
  */
   std::vector<float> _testingAverageReturnHistory;
  /**
  * @brief [Internal Use] Contains a running average of the training episode returns.
  */
   std::vector<float> _trainingAverageReturn;
  /**
  * @brief [Internal Use] Remembers the cumulative sum of rewards for the last training episode.
  */
   std::vector<float> _trainingLastReturn;
  /**
  * @brief [Internal Use] Remembers the best cumulative sum of rewards found so far in any episodes.
  */
   std::vector<float> _trainingBestReturn;
  /**
  * @brief [Internal Use] Remembers the episode that obtained the maximum cumulative sum of rewards found so far.
  */
   std::vector<size_t> _trainingBestEpisodeId;
  /**
  * @brief [Internal Use] Stores the current training policies configuration.
  */
   knlohmann::json _trainingCurrentPolicies;
  /**
  * @brief [Internal Use] Stores the best training policies configuration found so far.
  */
   knlohmann::json _trainingBestPolicies;
  /**
  * @brief [Internal Use] The cumulative sum of rewards obtained when evaluating the testing samples.
  */
   std::vector<float> _testingReturn;
  /**
  * @brief [Internal Use] Remembers the best cumulative sum of rewards from latest testing episodes, if any.
  */
   float _testingBestReturn;
  /**
  * @brief [Internal Use] Remembers the worst cumulative sum of rewards from latest testing episodes, if any.
  */
   float _testingWorstReturn;
  /**
  * @brief [Internal Use] Remembers the episode Id that obtained the maximum cumulative sum of rewards found so far during testing.
  */
   size_t _testingBestEpisodeId;
  /**
  * @brief [Internal Use] Remembers the number of candidate policies tested so far.
  */
   size_t _testingCandidateCount;
  /**
  * @brief [Internal Use] Remembers the average cumulative sum of rewards from latest testing episodes, if any.
  */
   float _testingAverageReturn;
  /**
  * @brief [Internal Use] Remembers the best cumulative sum of rewards found so far from testing episodes.
  */
   float _testingBestAverageReturn;
  /**
  * @brief [Internal Use] Stores the best testing policies configuration found so far.
  */
   knlohmann::json _testingBestPolicies;
  /**
  * @brief [Internal Use] Number of off-policy experiences in the experience replay.
  */
   std::vector<size_t> _experienceReplayOffPolicyCount;
  /**
  * @brief [Internal Use] Current off policy ratio in the experience replay.
  */
   std::vector<float> _experienceReplayOffPolicyRatio;
  /**
  * @brief [Internal Use] Indicates the current cutoff to classify experiences as on- or off-policy.
  */
   float _experienceReplayOffPolicyCurrentCutoff;
  /**
  * @brief [Internal Use] Vector of the current penalisation coefficient for off-policiness for each agent.
  */
   std::vector<float> _experienceReplayOffPolicyREFERCurrentBeta;
  /**
  * @brief [Internal Use] The current learning rate to use for the NN hyperparameter optimization.
  */
   float _currentLearningRate;
  /**
  * @brief [Internal Use] Keeps track of the number of policy updates that have been performed.
  */
   size_t _policyUpdateCount;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] Count of the number of experiences produced so far.
  */
   size_t _experienceCount;
  /**
  * @brief [Internal Use] Contains the standard deviation of the rewards. They will be scaled by this value in order to normalize the reward distribution in the RM.
  */
   float _rewardRescalingSigma;
  /**
  * @brief [Internal Use] Sum of squared rewards in experience replay.
  */
   float _rewardRescalingSumSquaredRewards;
  /**
  * @brief [Internal Use] Contains the mean of the states. They will be shifted by this value in order to normalize the state distribution in the RM.
  */
   std::vector<std::vector<float>> _stateRescalingMeans;
  /**
  * @brief [Internal Use] Contains the standard deviations of the states. They will be scaled by this value in order to normalize the state distribution in the RM.
  */
   std::vector<std::vector<float>> _stateRescalingSigmas;
  /**
  * @brief [Internal Use] Effective Minibatch Size in the context of MARL.
  */
   size_t _effectiveMinibatchSize;
  /**
  * @brief [Termination Criteria] The solver will stop when the given number of episodes have been run.
  */
   size_t _maxEpisodes;
  /**
  * @brief [Termination Criteria] The solver will stop when the given number of experiences have been gathered.
  */
   size_t _maxExperiences;
  /**
  * @brief [Termination Criteria] The solver will stop when the given number of optimization steps have been performed.
  */
   size_t _maxPolicyUpdates;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
   * @brief Array of workers collecting new experiences
   */
  std::vector<Sample> _workers;

  /**
   * @brief Keeps track of the workers
   */
  std::vector<bool> _isWorkerRunning;

  /**
   * @brief Pointer to training the actor network
   */
  std::vector<solver::DeepSupervisor *> _criticPolicyLearner;

  /**
   * @brief Korali experiment for obtaining the agent's action
   */
  std::vector<korali::Experiment> _criticPolicyExperiment;

  /**
   * @brief Pointer to actor's experiment problem
   */
  std::vector<problem::SupervisedLearning *> _criticPolicyProblem;

  /**
   * @brief Session-specific experience count. This is useful in case of restart: counters from the old session won't count
   */
  size_t _sessionExperienceCount;

  /**
   * @brief Session-specific episode count. This is useful in case of restart: counters from the old session won't count
   */
  size_t _sessionEpisodeCount;

  /**
   * @brief Session-specific generation count. This is useful in case of restart: counters from the old session won't count
   */
  size_t _sessionGeneration;

  /**
   * @brief Session-specific policy update count. This is useful in case of restart: counters from the old session won't count
   */
  size_t _sessionPolicyUpdateCount;

  /**
   * @brief Session-specific counter that keeps track of how many experiences need to be obtained this session to reach the start training threshold.
   */
  size_t _sessionExperiencesUntilStartSize;

  /**
   * @brief Stores the state of the experience
   */
  cBuffer<std::vector<std::vector<float>>> _stateBuffer;

  /**
   * @brief Stores the action taken by the agent
   */
  cBuffer<std::vector<std::vector<float>>> _actionBuffer;

  /**
   * @brief Stores the current sequence of states observed by the agent (limited to time sequence length defined by the user)
   */
  std::vector<cBuffer<std::vector<float>>> _stateTimeSequence;

  /**
   * @brief Episode that experience belongs to
   */
  cBuffer<size_t> _episodeIdBuffer;

  /**
   * @brief Position within the episode of this experience
   */
  cBuffer<size_t> _episodePosBuffer;

  /**
   * @brief Contains the latest calculation of the experience's importance weight
   */
  cBuffer<std::vector<float>> _importanceWeightBuffer;

  /**
   * @brief Contains the latest calculation of the experience's truncated importance weight (for cache optimzed update of retV in updateExperienceMetadata)
   */
  cBuffer<float> _truncatedImportanceWeightBufferContiguous;

  /**
   * @brief Contains the latest calculation of the product of the product of the experience's importance weights
   */
  cBuffer<float> _productImportanceWeightBuffer;

  /**
   * @brief Contains the most current policy information given the experience state
   */
  cBuffer<std::vector<policy_t>> _curPolicyBuffer;

  /**
   * @brief Contains the policy information produced at the moment of the action was taken
   */
  cBuffer<std::vector<policy_t>> _expPolicyBuffer;

  /**
   * @brief Indicates whether the experience is on policy, given the specified off-policiness criteria
   */
  cBuffer<std::vector<char>> _isOnPolicyBuffer;

  /**
   * @brief Specifies whether the experience is terminal (truncated or normal) or not.
   */
  cBuffer<termination_t> _terminationBuffer;

  /**
   * @brief Contains the result of the retrace (Vtbc) function for the currrent experience (for cache optimzed update of retV in updateExperienceMetadata)
   */
  cBuffer<float> _retraceValueBufferContiguous;

  /**
   * @brief If this is a truncated terminal experience, this contains the state value for that state
   */
  cBuffer<std::vector<float>> _truncatedStateValueBuffer;

  /**
   * @brief If this is a truncated terminal experience, the truncated state is also saved here
   */
  cBuffer<std::vector<std::vector<float>>> _truncatedStateBuffer;

  /**
   * @brief Contains the rewards of every experience (for cache optimzed update of retV in updateExperienceMetadata)
   */
  cBuffer<float> _rewardBufferContiguous;

  /**
   * @brief Contains the state value evaluation for every experience (for cache optimzed update of retV in updateExperienceMetadata)
   */
  cBuffer<float> _stateValueBufferContiguous;

  /**
   * @brief Stores the priority annealing rate.
   */
  float _priorityAnnealingRate;

  /**
   * @brief Stores the importance weight annealing factor.
   */
  float _importanceWeightAnnealingRate;

  /**
   * @brief Storage for the pointer to the learning problem
   */
  problem::ReinforcementLearning *_problem;

  /****************************************************************************************************
   * Session-wise Profiling Timers
   ***************************************************************************************************/

  /**
   * @brief [Profiling] Measures the amount of time taken by the generation
   */
  double _sessionRunningTime;

  /**
   * @brief [Profiling] Measures the amount of time taken by ER serialization
   */
  double _sessionSerializationTime;

  /**
   * @brief [Profiling] Stores the computation time per episode taken by Workers
   */
  double _sessionWorkerComputationTime;

  /**
   * @brief [Profiling] Measures the average communication time per episode taken by Workers
   */
  double _sessionWorkerCommunicationTime;

  /**
   * @brief [Profiling] Measures the average policy evaluation time per episode taken by Workers
   */
  double _sessionPolicyEvaluationTime;

  /**
   * @brief [Profiling] Measures the time taken to update the policy in the current generation
   */
  double _sessionPolicyUpdateTime;

  /**
   * @brief [Profiling] Measures the time taken to update the attend the agent's state
   */
  double _sessionWorkerAttendingTime;

  /****************************************************************************************************
   * Generation-wise Profiling Timers
   ***************************************************************************************************/

  /**
   * @brief [Profiling] Measures the amount of time taken by the generation
   */
  double _generationRunningTime;

  /**
   * @brief [Profiling] Measures the amount of time taken by ER serialization
   */
  double _generationSerializationTime;

  /**
   * @brief [Profiling] Stores the computation time per episode taken by worker
   */
  double _generationWorkerComputationTime;

  /**
   * @brief [Profiling] Measures the average communication time per episode taken by Workers
   */
  double _generationWorkerCommunicationTime;

  /**
   * @brief [Profiling] Measures the average policy evaluation time per episode taken by Workers
   */
  double _generationPolicyEvaluationTime;

  /**
   * @brief [Profiling] Measures the time taken to update the policy in the current generation
   */
  double _generationPolicyUpdateTime;

  /**
   * @brief [Profiling] Measures the time taken to update the attend the agent's state
   */
  double _generationWorkerAttendingTime;

  /****************************************************************************************************
   * Common functions
   ***************************************************************************************************/

  /**
   * @brief Mini-batch based normalization routine for Neural Networks with state and action inputs (typically critics)
   * @param neuralNetwork Neural Network to normalize
   * @param miniBatchSize Number of entries in the normalization minibatch
   * @param normalizationSteps How many normalization steps to perform (and grab the average)
   */
  void normalizeStateActionNeuralNetwork(NeuralNetwork *neuralNetwork, size_t miniBatchSize, size_t normalizationSteps);

  /**
   * @brief Mini-batch based normalization routine for Neural Networks with state inputs only (typically policy)
   * @param neuralNetwork Neural Network to normalize
   * @param miniBatchSize Number of entries in the normalization minibatch
   * @param normalizationSteps How many normalization steps to perform (and grab the average)
   */
  void normalizeStateNeuralNetwork(NeuralNetwork *neuralNetwork, size_t miniBatchSize, size_t normalizationSteps);

  /**
   * @brief Average rewards across agents per experience in multi agent framework.
   * @param message A json object containing all experiences from all agents.
   */
  void averageRewardsAcrossAgents(knlohmann::json &message);

  /**
   * @brief Additional post-processing of episode after episode terminated.
   * @param episode A vector of experiences pertaining to the episode.
   */
  void processEpisode(knlohmann::json &episode);

  /**
   * @brief Generates an experience mini batch from the replay memory
   * @return A vector of pairs with the indexes to the experiences and agents in the mini batch
   */
  std::vector<std::pair<size_t, size_t>> generateMiniBatch();

  /**
   * @brief Gets a vector of states corresponding of time sequence corresponding to the provided last experience index
   * @param miniBatch Indexes to the latest experiences in a batch of sequences
   * @return The time step vector of states
   */
  std::vector<std::vector<std::vector<float>>> getMiniBatchStateSequence(const std::vector<std::pair<size_t, size_t>> &miniBatch);

  /**
   * @brief Updates the state value, retrace, importance weight and other metadata for a given minibatch of experiences
   * @param miniBatch The mini batch of experience ids to update
   * @param policyData The policy to use to evaluate the experiences
   */
  void updateExperienceMetadata(const std::vector<std::pair<size_t, size_t>> &miniBatch, const std::vector<policy_t> &policyData);

  /**
   * @brief Resets time sequence within the agent, to forget past actions from other episodes
   */
  void resetTimeSequence();

  /**
   * @brief Function to pass a state time series through the NN and calculates the action probabilities, along with any additional information
   * @param stateSequence The batch of state time series (Format: BxTxS, B is batch size, T is the time series lenght, and S is the state size)
   * @param policyIdx The index for the policy for which the state-value is computed
   * @return A JSON object containing the information produced by the policies given the current state series
   */
  virtual float calculateStateValue(const std::vector<std::vector<float>> &stateSequence, size_t policyIdx = 0) = 0;

  /**
   * @brief Function to pass a state time series through the NN and calculates the action probabilities, along with any additional information
   * @param stateSequenceBatch The batch of state time series (Format: BxTxS, B is batch size, T is the time series lenght, and S is the state size)
   * @param policy Vector with policy objects that is filled after forwarding the policy
   * @param policyIdx The index for the policy for which the state-value is computed
   */
  virtual void runPolicy(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &policy, size_t policyIdx = 0) = 0;

  /**
   * @brief Calculates the starting experience index of the time sequence for the selected experience
   * @param expId The index of the latest experience in the sequence
   * @return The starting time sequence index
   */
  size_t getTimeSequenceStartExpId(size_t expId);

  /**
   * @brief Gets a vector of states corresponding of time sequence corresponding to the provided second-to-last experience index for which a truncated state exists
   * @param expId The index of the second-to-latest experience in the sequence
   * @param agentId The index of the agent
   * @return The time step vector of states, including the truncated state
   */
  std::vector<std::vector<float>> getTruncatedStateSequence(size_t expId, size_t agentId);

  /**
   * @brief Calculates importance weight of current action from old and new policies
   * @param action The action taken
   * @param curPolicy The current policy
   * @param oldPolicy The old policy, the one used for take the action in the first place
   * @return The importance weight
   */
  virtual float calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy) = 0;

  /**
   * @brief Listens to incoming experience from the given agent, sends back policy or terminates the episode depending on what's needed
   * @param workerId The worker's ID
   */
  void attendWorker(const size_t workerId);

  /**
   * @brief Serializes the experience replay into a JSON compatible format
   */
  void serializeExperienceReplay();

  /**
   * @brief Deserializes a JSON object into the experience replay
   */
  void deserializeExperienceReplay();

  /**
   * @brief Runs a generation when running in training mode
   */
  void trainingGeneration();

  /**
   * @brief Runs a generation when running in testing mode
   */
  void testingGeneration();

  /**
   * @brief Rescales states to have a zero mean and unit variance
   */
  void rescaleStates();

  /**
   * @brief Rescales a given reward by the square root of the sum of squarred rewards
   * @param reward the input reward to rescale
   * @return The normalized reward
   */
  inline float getScaledReward(const float reward)
  {
    float rescaledReward = reward / _rewardRescalingSigma;

    if (std::isfinite(rescaledReward) == false)
      KORALI_LOG_ERROR("Scaled reward is non finite: %f  (Sigma: %f)\n", rescaledReward, _rewardRescalingSigma);

    return rescaledReward;
  }

  /****************************************************************************************************
   * Virtual functions (responsibilities) for learning algorithms to fulfill
   ***************************************************************************************************/

  /**
   * @brief Trains the policy, based on the new experiences
   */
  virtual void trainPolicy() = 0;

  /**
   * @brief Obtains the policy hyperaparamters from the learner for the agent to generate new actions
   * @return The current policy hyperparameters
   */
  virtual knlohmann::json getPolicy() = 0;

  /**
   * @brief Updates the agent's hyperparameters
   * @param hyperparameters The hyperparameters to update the agent.
   */
  virtual void setPolicy(const knlohmann::json &hyperparameters) = 0;

  /**
   * @brief Initializes the internal state of the policy
   */
  virtual void initializeAgent() = 0;

  /**
   * @brief Prints information about the training policy
   */
  virtual void printInformation() = 0;

  /**
   * @brief Gathers the next action either from the policy or randomly
   * @param sample Sample on which the action and metadata will be stored
   */
  virtual void getAction(korali::Sample &sample) = 0;

  void runGeneration() override;
  void printGenerationAfter() override;
  void initialize() override;
  void finalize() override;
};

} //solver
} //korali
;
