/** \namespace layer
* @brief Namespace declaration for modules of type: layer.
*/

/** \file
* @brief Header file for module: Output.
*/

/** \dir neuralNetwork/layer/output
* @brief Contains code, documentation, and scripts for module: Output.
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
 * @brief This enumerator details all possible transformations. It is used in lieu of string comparison to accelerate the application of this layer
 */
enum transformation_t
{
  /**
   * @brief No transformation
   */
  t_identity = 0,

  /**
   * @brief Apply absolute mask
   */
  t_absolute = 1,

  /**
   * @brief Apply softplus mask
   */
  t_softplus = 2,

  /**
   * @brief Apply tanh mask
   */
  t_tanh = 3,

  /**
   * @brief Apply sigmoid mask
   */
  t_sigmoid = 4
};

/**
* @brief Class declaration for module: Output.
*/
class Output : public Layer
{
  public: 
  /**
  * @brief Indicates a transformation to be performed to the output at the last layer of the neural network. [Order of application on forward propagation: 1/3]
  */
   std::vector<std::string> _transformationMask;
  /**
  * @brief Gives a scaling factor for each of the output values of the NN. [Order of application on forward propagation 2/3]
  */
   std::vector<float> _scale;
  /**
  * @brief Shifts the output of the NN by the values given. [Order of application on forward propagation 3/3]
  */
   std::vector<float> _shift;
  
 
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
   * @brief Contains the original output, before preprocessing
   */
  float *_srcOutputValues;

  /**
   * @brief Contains the postprocessed gradients
   */
  float *_dstOutputGradients;

  /**
   * @brief Contains the description of the transformation to apply to each output element
   */
  std::vector<transformation_t> _transformationVector;

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
