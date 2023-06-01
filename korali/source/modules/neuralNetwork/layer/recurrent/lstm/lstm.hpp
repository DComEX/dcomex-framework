/** \namespace recurrent
* @brief Namespace declaration for modules of type: recurrent.
*/

/** \file
* @brief Header file for module: LSTM.
*/

/** \dir neuralNetwork/layer/recurrent/lstm
* @brief Contains code, documentation, and scripts for module: LSTM.
*/

#pragma once

#include "modules/neuralNetwork/layer/recurrent/recurrent.hpp"

namespace korali
{
namespace neuralNetwork
{
namespace layer
{
namespace recurrent
{
;

/**
* @brief Class declaration for module: LSTM.
*/
class LSTM : public Recurrent
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
  

#ifdef _KORALI_USE_ONEDNN

  /**
   * @brief oneDNN Memory object descriptor to contain the cell (LSTM) state of the recurrent layer -- vector, one per timestep
   */
  std::vector<dnnl::memory> _cellStateMem;

  /**
   * @brief oneDNN Memory object descriptor to contain the gradients of the cell (LSTM) state of the recurrent layer -- vector, one per timestep
   */
  std::vector<dnnl::memory> _cellStateGradientMem;

  /**
   * @brief oneDNN primitive attributes that describe the forward LSTM primitive
   */
  dnnl::lstm_forward::primitive_desc _forwardLSTMPrimitiveDesc;

  /**
   * @brief oneDNN primitive to run the forward LSTM operation
   */
  dnnl::primitive _forwardLSTMPrimitive;

  /**
   * @brief oneDNN primitive attributes that describe the backward LSTM primitive
   */
  dnnl::lstm_backward::primitive_desc _backwardLSTMPrimitiveDesc;

  /**
   * @brief oneDNN primitive to run the backward LSTM operation
   */
  dnnl::primitive _backwardLSTMPrimitive;

#endif

#ifdef _KORALI_USE_CUDNN

  /**
   * @brief cuDNN Descriptor for the cell state tensor memory
   */
  cudnnTensorDescriptor_t _cTensorDesc;

  /**
   * @brief cuDNN Device memory pointers for the internal layer's cell (LSTM) state input
   */
  std::vector<void *> _cStateTensor;

  /**
   * @brief cuDNN Device memory pointers for the internal layer's cell (LSTM) state input gradients
   */
  std::vector<void *> _cGradientTensor;

#endif

  void initialize() override;
  void createForwardPipeline() override;
  void createBackwardPipeline() override;
  void forwardData(const size_t t) override;
  void backwardData(const size_t t) override;
};

} //recurrent
} //layer
} //neuralNetwork
} //korali
;
