/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: NeuralNetwork.
*/

/** \dir neuralNetwork
* @brief Contains code, documentation, and scripts for module: NeuralNetwork.
*/

#pragma once

#include "config.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/module.hpp"
#include "modules/neuralNetwork/layer/layer.hpp"
#include "modules/solver/solver.hpp"

#ifdef _KORALI_USE_ONEDNN
  #include "dnnl.hpp"
#endif

#ifdef _KORALI_USE_CUDNN
  #include <cuda.h>
  #include <cudnn.h>
#endif

namespace korali
{
;

/**
 * @brief Structure containing the information of a layer pipeline. There is one pipeline per threadCount x batchSize combination.
 */
struct layerPipeline_t
{
  /**
   * @brief Internal container for the NN layer forward/backward pipelines.
   */
  std::vector<korali::neuralNetwork::Layer *> _layerVector;

  /**
   * @brief Raw data for the NN input values. Format: TxNxIC (T: Time steps, N: Mini-batch size, IC: Input channels).
   */
  std::vector<float> _rawInputValues;

  /**
   * @brief Raw data for the NN input gradients. Format: TxNxIC (T: Time steps, N: Mini-batch size, IC: Input channels).
   */
  std::vector<float> _rawInputGradients;

  /**
   * @brief Formatted data for the NN input gradients. Format: NxIC (N: Mini-batch size, IC: Input channels).
   */
  std::vector<std::vector<float>> _inputGradients;

  /**
   * @brief Raw data for the NN output values. Format: NxOC (N: Mini-batch size, OC: Output channels).
   */
  std::vector<float> _rawOutputValues;

  /**
   * @brief Raw data for the NN output gradients. Format: NxOC (N: Mini-batch size, OC: Output channels).
   */
  std::vector<float> _rawOutputGradients;

  /**
   * @brief Formatted data for the NN output values. Format: NxOC (N: Mini-batch size, OC: Output channels).
   */
  std::vector<std::vector<float>> _outputValues;

  /**
   * @brief F data for the NN hyperparameter gradients. Format: H (H: Hyperparameter count).
   */
  std::vector<float> _hyperparameterGradients;

  /**
   * @brief Remembers the position of the last timestep provided as input
   */
  std::vector<size_t> _inputBatchLastStep;
};

/**
* @brief Class declaration for module: NeuralNetwork.
*/
class NeuralNetwork : public Module
{
  public: 
  /**
  * @brief Specifies which Neural Network backend engine to use.
  */
   std::string _engine;
  /**
  * @brief Specifies the execution mode of the Neural Network.
  */
   std::string _mode;
  /**
  * @brief Complete description of the NN's layers.
  */
   knlohmann::json _layers;
  /**
  * @brief Provides the sequence length for the input/output data.
  */
   size_t _timestepCount;
  /**
  * @brief Specifies the batch sizes.
  */
   std::vector<size_t> _batchSizes;
  /**
  * @brief [Internal Use] Current value of the training loss.
  */
   float _currentTrainingLoss;
  /**
  * @brief [Internal Use] Uniform random number generator for setting the initial value of the weights and biases.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  
 
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
  

#ifdef _KORALI_USE_ONEDNN

  /**
   * @brief oneDNN Stream to run operations
   */
  dnnl::stream _dnnlStream;

  /**
   * @brief oneDNN Engine for the current NN
   */
  dnnl::engine _dnnlEngine;

#endif

#ifdef _KORALI_USE_CUDNN

  /**
   * @brief CUDNN handle for the execution of operations on the GPU
   */
  cudnnHandle_t _cuDNNHandle;

#endif

  /**
   * @brief Layer pipelines, one per threadCount * BatchSize combination. These are all replicas of the user-defined layers that
   *        share the same hyperparameter space.
   */
  std::vector<std::vector<layerPipeline_t>> _pipelines;

  /**
   * @brief Flag to make sure the NN is initialized before creating
   */
  bool _isInitialized;

  /**
   * @brief Number of NN hyperparameters (weights/bias)
   */
  size_t _hyperparameterCount;

  /**
   * @brief Generates the initial values for the hyperparameters
   * @return The generated hyperparameters
   */
  std::vector<float> generateInitialHyperparameters();

  /**
   * @brief Updates the values of weights, biases configuration to the NN
   * @param hyperparameters The input hyperparameters
   */
  void setHyperparameters(const std::vector<float> &hyperparameters);

  /**
   * @brief Gets the values of weights and biases configuration to the NN
   * @return The hyperparameters of the NN
   */
  std::vector<float> getHyperparameters();

  /**
   * @brief Forward-propagates the input values through the network.
   * @param inputValues The input values.  Format: TxNxIC (T: Time steps, N: Mini-batch, IC: Input channels).
   */
  void forward(const std::vector<std::vector<std::vector<float>>> &inputValues);

  /**
   * @brief Backward-propagates the gradients through the network.
   * @param outputGradients Output gradients. Format: NxOC (N: Mini-batch size, OC: Output channels).
   */
  void backward(const std::vector<std::vector<float>> &outputGradients);

  /**
   * @brief Returns the pipeline index corresponding to the batch size requested
   * @param batchSize Size of the batch to request
   * @return Pipeline Id correspoding to the batch size
   */
  size_t getBatchSizeIdx(const size_t batchSize);

  /**
   * @brief Returns a reference to the output values corresponding to the batch size's pipeline
   * @param batchSize Size of the batch to request
   * @return Reference to the output values
   */
  std::vector<std::vector<float>> &getOutputValues(const size_t batchSize);

  /**
   * @brief Returns a reference to the input gradients corresponding to the batch size's pipeline
   * @param batchSize Size of the batch to request
   * @return Reference to the input gradients
   */
  std::vector<std::vector<float>> &getInputGradients(const size_t batchSize);

  /**
   * @brief Returns a reference to the hyperparameter gradients corresponding to the batch size's pipeline
   * @param batchSize Size of the batch to request
   * @return Reference hyperparameter gradients
   */
  std::vector<float> &getHyperparameterGradients(const size_t batchSize);

  /**
   * @brief Creator that sets initialized flag to false
   */
  NeuralNetwork();

  void initialize() override;
};

} //korali
;
