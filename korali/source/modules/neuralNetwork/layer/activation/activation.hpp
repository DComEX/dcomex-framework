/** \namespace layer
* @brief Namespace declaration for modules of type: layer.
*/

/** \file
* @brief Header file for module: Activation.
*/

/** \dir neuralNetwork/layer/activation
* @brief Contains code, documentation, and scripts for module: Activation.
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
* @brief Class declaration for module: Activation.
*/
class Activation : public Layer
{
  public: 
  /**
  * @brief Indicates the activation function for the weighted inputs to the current layer.
  */
   std::string _function;
  /**
  * @brief First (alpha) argument to the activation function, as detailed in https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html
  */
   float _alpha;
  /**
  * @brief Second (beta) argument to the activation function, as detailed in https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html
  */
   float _beta;
  
 
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
   * @brief oneDNN Algorithm chosen for activation function
   */
  dnnl::algorithm _activationAlgorithm;

  /**
   * @brief oneDNN Arguments to the activation function
   */
  std::unordered_map<int, dnnl::memory> _forwardActivationArgs;

  /**
   * @brief oneDNN primitive attributes that describe a softmax activation function
   */
  dnnl::softmax_forward::primitive_desc _forwardSoftmaxActivationPrimitiveDesc;

  /**
   * @brief oneDNN primitive attributes that describe an element-wise activation function
   */
  dnnl::eltwise_forward::primitive_desc _forwardEltwiseActivationPrimitiveDesc;

  /**
   * @brief oneDNN primitive to run the activation function operation
   */
  dnnl::primitive _forwardActivationPrimitive;

  /**
   * @brief oneDNN Arguments for the backward propagation of the gradient wrt activation functions
   */
  std::unordered_map<int, dnnl::memory> _backwardActivationArgs;

  /**
   * @brief oneDNN primitive for the backward propagation of the gradient wrt activation functions
   */
  dnnl::primitive _backwardActivationPrimitive;

  /**
   * @brief oneDNN Arguments for the backward propagation of the gradient wrt Data
   */
  std::unordered_map<int, dnnl::memory> _backwardDataArgs;

  /**
   * @brief oneDNN primitive for the backward propagation of the gradient wrt Data
   */
  dnnl::primitive _backwardDataPrimitive;

  /**
   * @brief oneDNN Arguments for the backward propagation of the gradient wrt Weights and Biases
   */
  std::unordered_map<int, dnnl::memory> _backwardWeightsArgs;

  /**
   * @brief oneDNN primitive for the backward propagation of the gradient wrt Weights and Biases
   */
  dnnl::primitive _backwardWeightsPrimitive;

#endif

#ifdef _KORALI_USE_CUDNN

  /**
   * @brief cuDNN Descriptor for the activation function
   */
  cudnnActivationDescriptor_t _activationDesc;

#endif

  void initialize() override;
  void createForwardPipeline() override;
  void createBackwardPipeline() override;
  void forwardData(const size_t t) override;
  void backwardData(const size_t t) override;
};

} //layer
} //neuralNetwork
} //korali
;
