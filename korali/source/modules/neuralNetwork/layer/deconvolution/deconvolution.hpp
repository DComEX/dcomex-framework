/** \namespace layer
* @brief Namespace declaration for modules of type: layer.
*/

/** \file
* @brief Header file for module: Deconvolution.
*/

/** \dir neuralNetwork/layer/deconvolution
* @brief Contains code, documentation, and scripts for module: Deconvolution.
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
* @brief Class declaration for module: Deconvolution.
*/
class Deconvolution : public Layer
{
  public: 
  /**
  * @brief Height of the incoming 2D image.
  */
   ssize_t _imageHeight;
  /**
  * @brief Width of the incoming 2D image.
  */
   ssize_t _imageWidth;
  /**
  * @brief Height of the incoming 2D image.
  */
   ssize_t _kernelHeight;
  /**
  * @brief Width of the incoming 2D image.
  */
   ssize_t _kernelWidth;
  /**
  * @brief Strides for the image on the vertical dimension.
  */
   ssize_t _verticalStride;
  /**
  * @brief Strides for the image on the horizontal dimension.
  */
   ssize_t _horizontalStride;
  /**
  * @brief Paddings for the image left side.
  */
   ssize_t _paddingLeft;
  /**
  * @brief Paddings for the image right side.
  */
   ssize_t _paddingRight;
  /**
  * @brief Paddings for the image top side.
  */
   ssize_t _paddingTop;
  /**
  * @brief Paddings for the image Bottom side.
  */
   ssize_t _paddingBottom;
  
 
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
   * @brief Pre-calculated value for Mini-Batch Size
   */
  ssize_t N;

  /**
   * @brief Pre-calculated value for Input Channels
   */
  ssize_t IC;

  /**
   * @brief Pre-calculated value for Input Image Height
   */
  ssize_t IH;

  /**
   * @brief Pre-calculated value for Input Image Width
   */
  ssize_t IW;

  /**
   * @brief Pre-calculated value for Output Channels
   */
  ssize_t OC;

  /**
   * @brief Pre-calculated value for Output Image Height
   */
  ssize_t OH;

  /**
   * @brief Pre-calculated value for Output Image Width
   */
  ssize_t OW;

  /**
   * @brief Pre-calculated value for Kernel Image Height
   */
  ssize_t KH;

  /**
   * @brief Pre-calculated value for Kernel Image Width
   */
  ssize_t KW;

  /**
   * @brief Pre-calculated values for padding left
   */
  ssize_t PL;

  /**
   * @brief Pre-calculated values for padding right
   */
  ssize_t PR;

  /**
   * @brief Pre-calculated values for padding top
   */
  ssize_t PT;

  /**
   * @brief Pre-calculated values for padding bottom
   */
  ssize_t PB;

  /**
   * @brief Pre-calculated values for horizontal stride
   */
  ssize_t SH;

  /**
   * @brief Pre-calculated values for vertical stride
   */
  ssize_t SV;

#ifdef _KORALI_USE_ONEDNN

  /**
   * @brief Memory descriptor for the 2D mapping of the scalar input channels
   */
  dnnl::memory::desc _srcMemDesc;

  /**
   * @brief Memory descriptor for the 2D mapping of the scalar output channels
   */
  dnnl::memory::desc _dstMemDesc;

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
   * @brief oneDNN primitive attributes that describe the forward deconvolution primitive
   */
  dnnl::deconvolution_forward::primitive_desc _forwardDeconvolutionPrimitiveDesc;

  /**
   * @brief oneDNN primitive to run the inner product + bias addition operation
   */
  dnnl::primitive _forwardDeconvolutionPrimitive;

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
