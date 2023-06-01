/** \namespace layer
* @brief Namespace declaration for modules of type: layer.
*/

/** \file
* @brief Header file for module: Recurrent.
*/

/** \dir neuralNetwork/layer/recurrent
* @brief Contains code, documentation, and scripts for module: Recurrent.
*/

#pragma once

#include "modules/neuralNetwork/layer/layer.hpp"

namespace korali
{
namespace neuralNetwork
{
namespace layer
{
;

/**
* @brief Class declaration for module: Recurrent.
*/
class Recurrent : public Layer
{
  public: 
  /**
  * @brief The number of copies of this layer. This has a better performance than just defining many of these layers manually since it is optimized by the underlying engine.
  */
   size_t _depth;
  
 
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
   * @brief Indicates the number of recurrent gates (depends on the architecture)
   */
  size_t _gateCount;

#ifdef _KORALI_USE_ONEDNN

  /**
   * @brief oneDNN Memory object descriptor to contain the hidden state of the recurrent layer -- vector, one per timestep
   */
  std::vector<dnnl::memory> _hiddenStateMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the gradients of the hidden state of the recurrent layer -- vector, one per timestep
   */
  std::vector<dnnl::memory> _hiddenStateGradientMem;

  /**
   * @brief oneDNN Memory object descriptor to contain a null source input hidden state
   */
  dnnl::memory _nullStateInputMem;

  /**
   * @brief oneDNN Memory object descriptor to contain a null source output hidden state
   */
  dnnl::memory _nullStateOutputMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the weights of to apply to the input
   */
  dnnl::memory _weightsLayerMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the gradients of the weights of to apply to the input
   */
  dnnl::memory _weightsLayerGradientMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the weights of to apply to the recurrent gate
   */
  dnnl::memory _weightsRecurrentMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the gradients of the weights of to apply to the recurrent gate
   */
  dnnl::memory _weightsRecurrentGradientMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the bias to apply to the input
   */
  dnnl::memory _biasMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the gradients of the bias to apply to the input
   */
  dnnl::memory _biasGradientMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the workspace
   */
  std::vector<dnnl::memory> _workspaceMem;

#endif

#ifdef _KORALI_USE_CUDNN

  /**
   * @brief cuDNN Specifies what type of RNN operation to perform
   */
  cudnnRNNMode_t _rnnMode;

  /**
   * @brief cuDNN Descriptor for the hidden state tensor memory
   */
  cudnnTensorDescriptor_t _hTensorDesc;

  /**
   * @brief cuDNN Device memory pointers for the internal layer's hidden state input
   */
  std::vector<void *> _hStateTensor;

  /**
   * @brief cuDNN Device memory pointers for the internal layer's hidden state input gradients
   */
  std::vector<void *> _hGradientTensor;

  /**
   * @brief cuDNN descriptor for the dropout operator
   */
  cudnnDropoutDescriptor_t _dropoutDesc;

  /**
   * @brief cuDNN descriptor for the RNN operator
   */
  cudnnRNNDescriptor_t _rnnDesc;

  /**
   * @brief cuDNN Device memory pointers for layer's weights
   */
  void *_weightsTensor;

  /**
   * @brief cuDNN Device memory pointers for layer's weight gradients
   */
  void *_weightsGradientTensor;

  /**
   * @brief cuDNN Device RNN data descriptor for the input
   */
  cudnnRNNDataDescriptor_t _inputRNNDataDesc;

  /**
   * @brief cuDNN Device RNN data descriptor for the output
   */
  cudnnRNNDataDescriptor_t _outputRNNDataDesc;

  /**
   * @brief cuDNN Device memory pointer to of all gate weights
   */
  std::vector<void *> _gateWeightTensors;

  /**
   * @brief cuDNN Device memory pointer to of all gate biases
   */
  std::vector<void *> _gateBiasTensors;

  /**
   * @brief cuDNN Weights size
   */
  size_t _weightsSize;

  /**
   * @brief cuDNN Workspace size
   */
  size_t _workSpaceSize;

  /**
   * @brief cuDNN reserve space size
   */
  size_t _reserveSpaceSize;

  /**
   * @brief cuDNN Device memory pointers for workspace tensor
   */
  std::vector<void *> _workSpaceTensor;

  /**
   * @brief cuDNN Device memory pointers for reserve space tensor
   */
  std::vector<void *> _reserveSpaceTensor;

  /**
   * @brief cuDNN Device memory for sequence lenght array
   */
  int *_devSequenceLengths;

#endif

  void initialize() override;
  void createHyperparameterMemory() override;
  void backwardHyperparameters(const size_t t) override;
  void createBackwardPipeline() override;
  void copyHyperparameterPointers(Layer *dstLayer) override;
  std::vector<float> generateInitialHyperparameters() override;
  void setHyperparameters(const float *hyperparameters) override;
  void getHyperparameters(float *hyperparameters) override;
  void getHyperparameterGradients(float *gradient) override;
};

} //layer
} //neuralNetwork
} //korali
;
