/** \namespace recurrent
* @brief Namespace declaration for modules of type: recurrent.
*/

/** \file
* @brief Header file for module: GRU.
*/

/** \dir neuralNetwork/layer/recurrent/gru
* @brief Contains code, documentation, and scripts for module: GRU.
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
* @brief Class declaration for module: GRU.
*/
class GRU : public Recurrent
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
   * @brief oneDNN primitive attributes that describe the forward GRU primitive
   */
  dnnl::gru_forward::primitive_desc _forwardGRUPrimitiveDesc;

  /**
   * @brief oneDNN primitive to run the forward GRU operation
   */
  dnnl::primitive _forwardGRUPrimitive;

  /**
   * @brief oneDNN primitive attributes that describe the backward GRU primitive
   */
  dnnl::gru_backward::primitive_desc _backwardGRUPrimitiveDesc;

  /**
   * @brief oneDNN primitive to run the backward GRU operation
   */
  dnnl::primitive _backwardGRUPrimitive;

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
