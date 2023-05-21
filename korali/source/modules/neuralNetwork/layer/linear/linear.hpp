/** \namespace layer
* @brief Namespace declaration for modules of type: layer.
*/

/** \file
* @brief Header file for module: Linear.
*/

/** \dir neuralNetwork/layer/linear
* @brief Contains code, documentation, and scripts for module: Linear.
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
* @brief Class declaration for module: Linear.
*/
class Linear : public Layer
{
  public: 
  
 
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
  

  /********************************************************
   * Engine specific members
   *******************************************************/

  /**
   * @brief Contains the values of the weights
   */
  float *_weightValues;

  /**
   * @brief Contains the gradients of the weights
   */
  float *_weightGradient;

  /**
   * @brief Contains the values of the bias
   */
  float *_biasValues;

  /**
   * @brief Contains the gradients of the bias
   */
  float *_biasGradient;

#ifdef _KORALI_USE_ONEDNN
  /**
   * @brief oneDNN Memory object descriptor to contain the weights of inner product with incoming channels
   */
  dnnl::memory _weightsMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the weights of inner product with incoming channels
   */
  dnnl::memory _weightsReorderMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the bias to add to incoming channels
   */
  dnnl::memory _biasMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the gradient of the weights
   */
  dnnl::memory _weightsGradientMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the gradient of the biases
   */
  dnnl::memory _biasGradientMem;

  /**
   * @brief oneDNN primitive attributes that describe the full forward propagation primitive
   */
  dnnl::inner_product_forward::primitive_desc _forwardInnerProductPrimitiveDesc;

  /**
   * @brief oneDNN primitive to run the inner product + bias addition operation
   */
  dnnl::primitive _forwardInnerProductPrimitive;

  /**
   * @brief oneDNN Arguments for the backward propagation of the gradient wrt Data
   */
  std::unordered_map<int, dnnl::memory> _backwardDataArgs;

  /**
   * @brief oneDNN primitive for the backward propagation of the gradient wrt Data
   */
  dnnl::primitive _backwardDataPrimitive;

  /**
   * @brief oneDNN primitive for the backward propagation of the gradient wrt Weights and Biases
   */
  dnnl::primitive _backwardWeightsPrimitive;

#endif

#ifdef _KORALI_USE_CUDNN

  /**
   * @brief cuDNN Descriptor for the filter weights
   */
  cudnnFilterDescriptor_t _weightsFilterDesc;

  /**
   * @brief cuDNN Device memory pointer for the filter weights
   */
  void *_weightsFilter;

  /**
   * @brief cuDNN Device memory pointer for the filter weights gradients
   */
  void *_weightsGradientFilter;

  /**
   * @brief cuDNN Descriptor for the bias memory
   */
  cudnnTensorDescriptor_t _biasTensorDesc;

  /**
   * @brief cuDNN Device memory pointer for the bias tensor
   */
  void *_biasTensor;

  /**
   * @brief cuDNN Device memory pointer for the bias gradients
   */
  void *_biasGradientTensor;

  /**
   * @brief cuDNN Descriptor for the convolution operation
   */
  cudnnConvolutionDescriptor_t _convolutionDesc;

  /**
   * @brief cuDNN Placeholder for the convolution workspace size (bytes)
   */
  size_t _convolutionWorkspaceSize;

  /**
   * @brief cuDNN Device memory pointer for the convolution workspace
   */
  std::vector<void *> _convolutionWorkspace;

#endif

  void copyHyperparameterPointers(Layer *dstLayer) override;
  void initialize() override;
  std::vector<float> generateInitialHyperparameters() override;
  void createHyperparameterMemory() override;
  void createForwardPipeline() override;
  void createBackwardPipeline() override;
  void forwardData(const size_t t) override;

  void setHyperparameters(const float *hyperparameters) override;
  void getHyperparameters(float *hyperparameters) override;
  void getHyperparameterGradients(float *gradient) override;
  void backwardData(const size_t t) override;
  void backwardHyperparameters(const size_t t) override;
};

} //layer
} //neuralNetwork
} //korali
;
