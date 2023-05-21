/** \namespace neuralNetwork
* @brief Namespace declaration for modules of type: neuralNetwork.
*/

/** \file
* @brief Header file for module: Layer.
*/

/** \dir neuralNetwork/layer
* @brief Contains code, documentation, and scripts for module: Layer.
*/

#pragma once

#include "config.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/module.hpp"

#ifdef _KORALI_USE_ONEDNN
  #include "dnnl.hpp"
#endif

#ifdef _KORALI_USE_CUDNN
  #include <cuda.h>
  #include <cudnn.h>
#endif

namespace korali
{
/**
* @brief Class declaration for module: Layer.
*/
class NeuralNetwork;
struct layerPipeline_t;

} // namespace korali

namespace korali
{
namespace neuralNetwork
{
;

/**
* @brief Class declaration for module: Layer.
*/
class Layer : public Module
{
  public: 
  /**
  * @brief Indicates the size of the output vector produced by the layer.
  */
   size_t _outputChannels;
  /**
  * @brief Factor that is mutliplied by the layers' weights.
  */
   float _weightScaling;
  
 
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
   * @brief Index of the current layer within the NN
   */
  size_t _index;

  /**
   * @brief Pointer to the parent neural network
   */
  NeuralNetwork *_nn;

  /**
   * @brief A pointer to the layer's containing pipeline
   */
  layerPipeline_t *_pipeline;

  /**
   * @brief Pointer to previous layer, NULL if this is the first layer
   */
  Layer *_prevLayer;

  /**
   * @brief Pointer to next layer, NULL if this is the last layer
   */
  Layer *_nextLayer;

  /**
   * @brief Number of layer hyperparameters
   */
  size_t _hyperparameterCount;

  /**
   * @brief Starting index of hyperparameters
   */
  size_t _hyperparameterIndex;

  /**
   * @brief Contains the batch size corresponding to the pipeline
   */
  size_t _batchSize;

  // Input/output memory elements

  /**
   * @brief Contains the output values of the layer
   */
  float *_outputValues;

  /**
   * @brief Contains the gradients of the outputs of the layer
   */
  float *_outputGradient;

#ifdef _KORALI_USE_ONEDNN
  /**
   * @brief oneDNN Stores the propagation kind (inference/training)
   */
  dnnl::prop_kind _propKind;

  /**
   * @brief oneDNN Memory object descriptor to contain the output result of the layer
   */
  std::vector<dnnl::memory> _outputMem;

  /*
   * @brief oneDNN Gradients of the operation wrt to activation function
   */
  std::vector<dnnl::memory> _outputGradientMem;

#endif

#ifdef _KORALI_USE_CUDNN

  /**
   * @brief oneDNN Stores the forward propagation mode
   */
  cudnnForwardMode_t _forwardMode;

  /**
   * @brief cuDNN Descriptor for the output tensor memory
   */
  cudnnTensorDescriptor_t _outputTensorDesc;

  /**
   * @brief cuDNN Device memory pointer for the output tensor
   */
  std::vector<void *> _outputTensor;

  /**
   * @brief cuDNN Device memory pointer for the output gradients tensor
   */
  std::vector<void *> _outputGradientTensor;

#endif

  /**
   * @brief Default constructor
   */
  Layer() = default;

  /**
   * @brief Default destructor
   */
  virtual ~Layer() = default;

  /**
   * @brief Returns the output values for the current layer
   * @return The output values
   */
  std::vector<std::vector<float>> getOutput();

  /**
   * @brief Generates the initial weight/bias hyperparameters for the layer
   * @return The initial hyperparameters
   */
  virtual std::vector<float> generateInitialHyperparameters();

  /**
   * @brief Initializes the layer's internal memory structures for hyperparameter storage
   */
  virtual void createHyperparameterMemory();

  /**
   * @brief Replicates the pointers for the current layer onto a destination layer
   * @param dstLayer The destination layer onto which to copy the pointers
   */
  virtual void copyHyperparameterPointers(Layer *dstLayer){};

  /**
   * @brief Initializes the layer's internal memory structures for the forward pipeline
   */
  virtual void createForwardPipeline();

  /**
   * @brief Initializes the internal memory structures for the backward pipeline
   */
  virtual void createBackwardPipeline();

  /**
   * @brief Performs the forward propagation of the Wx+b operations
   * @param t Indicates the current timestep
   */
  virtual void forwardData(const size_t t) = 0;

  /**
   * @brief Updates layer's hyperparameters (e.g., weights and biases)
   * @param hyperparameters (Input) Pointer to read the hyperparameters from.
   */
  virtual void setHyperparameters(const float *hyperparameters){};

  /**
   * @brief Gets layer's hyperparameters (e.g., weights and biases)
   * @param hyperparameters (Output) Pointer to write the hyperparameters to.
   */
  virtual void getHyperparameters(float *hyperparameters){};

  /**
   * @brief Gets the gradients of the layer's output wrt to is hyperparameters (e.g., weights and biases)
   * @param gradient (Output) Pointer to write the hyperparameter gradients to.
   */
  virtual void getHyperparameterGradients(float *gradient){};

  /**
   * @brief Performs the backward propagation of the data
   * @param t Indicates the current timestep
   */
  virtual void backwardData(const size_t t) = 0;

  /**
   * @brief Calculates the gradients of layer hyperparameters
   * @param t Indicates the current timestep
   */
  virtual void backwardHyperparameters(const size_t t);
};

} //neuralNetwork
} //korali
;
