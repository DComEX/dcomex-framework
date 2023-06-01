#include "modules/neuralNetwork/layer/activation/activation.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"

#ifdef _KORALI_USE_CUDNN
  #include "auxiliar/cudaUtils.hpp"
#endif

#ifdef _KORALI_USE_ONEDNN
  #include "auxiliar/dnnUtils.hpp"
using namespace dnnl;
#endif

#include <Eigen/Dense>
using namespace Eigen;

namespace korali
{
namespace neuralNetwork
{
namespace layer
{
;

void Activation::initialize()
{
  // The node count for this layer should be the same as the previous layer
  _outputChannels = _prevLayer->_outputChannels;

  // Checking Layer size
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("Activation layers cannot be the starting layer of the NN\n");
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("Activation layers cannot be the last layer of the NN\n");
}

void Activation::createForwardPipeline()
{
  // Calling base layer function
  Layer::createForwardPipeline();

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // If it is an element-wise operation, create an element-wise primitive
    if (_function.rfind("Elementwise", 0) == 0)
    {
      if (_function == "Elementwise/Clip") _activationAlgorithm = algorithm::eltwise_clip;
      if (_function == "Elementwise/Linear") _activationAlgorithm = algorithm::eltwise_linear;
      if (_function == "Elementwise/Log") _activationAlgorithm = algorithm::eltwise_log;
      if (_function == "Elementwise/Logistic") _activationAlgorithm = algorithm::eltwise_logistic;
      if (_function == "Elementwise/ReLU") _activationAlgorithm = algorithm::eltwise_relu;
      if (_function == "Elementwise/SoftReLU") _activationAlgorithm = algorithm::eltwise_soft_relu;
      if (_function == "Elementwise/SoftSign") KORALI_LOG_ERROR("ONEDNN does not support activation functions of type 'Elementwise/SoftSign'.");
      if (_function == "Elementwise/Tanh") _activationAlgorithm = algorithm::eltwise_tanh;

      // Creating descriptor
      auto activationDesc = eltwise_forward::desc(
        _propKind,
        _activationAlgorithm,
        _prevLayer->_outputMem[0].get_desc(),
        _alpha,
        _beta);

      // Create primitive descriptor.
      _forwardEltwiseActivationPrimitiveDesc = eltwise_forward::primitive_desc(activationDesc, _nn->_dnnlEngine);

      // Create the primitive.
      _forwardActivationPrimitive = eltwise_forward(_forwardEltwiseActivationPrimitiveDesc);
    }

    // Check other possible types of activation functions
    if (_function == "Softmax")
    {
      // Creating descriptor
      const int axis = 1;
      auto activationDesc = softmax_forward::desc(_propKind, _prevLayer->_outputMem[0].get_desc(), axis);

      // Create primitive descriptor.
      _forwardSoftmaxActivationPrimitiveDesc = softmax_forward::primitive_desc(activationDesc, _nn->_dnnlEngine);

      // Create the primitive.
      _forwardActivationPrimitive = softmax_forward(_forwardSoftmaxActivationPrimitiveDesc);
    }
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudnnErrCheck(cudnnCreateActivationDescriptor(&_activationDesc));
    cudnnActivationMode_t activationMode;

    if (_function == "Elementwise/Clip") activationMode = CUDNN_ACTIVATION_CLIPPED_RELU;
    if (_function == "Elementwise/Linear") activationMode = CUDNN_ACTIVATION_IDENTITY;
    if (_function == "Elementwise/Log") KORALI_LOG_ERROR("CUDNN does not support activation functions of type 'Elementwise/Log'.");
    if (_function == "Elementwise/Logistic") activationMode = CUDNN_ACTIVATION_SIGMOID;
    if (_function == "Elementwise/ReLU") activationMode = CUDNN_ACTIVATION_RELU;
    if (_function == "Elementwise/SoftReLU") KORALI_LOG_ERROR("CUDNN does not support activation functions of type 'Elementwise/SoftReLU'.");
    if (_function == "Elementwise/SoftSign") KORALI_LOG_ERROR("CUDNN does not support activation functions of type 'Elementwise/SoftSign'.");
    if (_function == "Elementwise/Tanh") activationMode = CUDNN_ACTIVATION_TANH;
    if (_function == "Softmax") activationMode = CUDNN_ACTIVATION_IDENTITY;

    if (cudnnSetActivationDescriptor(_activationDesc, activationMode, CUDNN_PROPAGATE_NAN, _alpha) != CUDNN_STATUS_SUCCESS) KORALI_LOG_ERROR("Error creating activation algorithm\n");
  }
#endif
}

void Activation::createBackwardPipeline()
{
  // Calling base layer function
  Layer::createBackwardPipeline();

// Creating backward propagation primitives for activation functions
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // If it is an element-wise operation, create an element-wise backward primitive
    if (_function.rfind("Elementwise", 0) == 0)
    {
      // Creating descriptor
      auto activationDesc = eltwise_backward::desc(_activationAlgorithm, _prevLayer->_outputMem[0].get_desc(), _outputMem[0].get_desc(), _alpha, _beta);

      // Create primitive descriptor.
      auto backwardActivationPrimitiveDesc = eltwise_backward::primitive_desc(activationDesc, _nn->_dnnlEngine, _forwardEltwiseActivationPrimitiveDesc);

      // Create the primitive.
      _backwardActivationPrimitive = eltwise_backward(backwardActivationPrimitiveDesc);
    }

    // Check other possible types of activation functions
    if (_function == "Softmax")
    {
      // Creating descriptor
      const int axis = 1;
      auto activationDesc = softmax_backward::desc(_prevLayer->_outputMem[0].get_desc(), _outputMem[0].get_desc(), axis);

      // Create primitive descriptor.
      auto backwardActivationPrimitiveDesc = softmax_backward::primitive_desc(activationDesc, _nn->_dnnlEngine, _forwardSoftmaxActivationPrimitiveDesc);

      // Create the primitive.
      _backwardActivationPrimitive = softmax_backward(backwardActivationPrimitiveDesc);
    }
  }
#endif
}

void Activation::forwardData(const size_t t)
{
  size_t N = _batchSize;
  size_t OC = _outputChannels;

  if (_nn->_engine == "Korali")
  {
    if (_function == "Elementwise/Clip")
    {
      for (size_t i = 0; i < N * OC; i++)
      {
        if (_prevLayer->_outputValues[i] < _alpha)
          _outputValues[i] = _alpha;
        else if (_prevLayer->_outputValues[i] > _beta)
          _outputValues[i] = _beta;
        else
          _outputValues[i] = _prevLayer->_outputValues[i];
      }
    }
    if (_function == "Elementwise/Linear")
    {
      for (size_t i = 0; i < N * OC; i++)
        _outputValues[i] = _prevLayer->_outputValues[i] * _alpha + _beta;
    }
    if (_function == "Elementwise/Log")
    {
      for (size_t i = 0; i < N * OC; i++)
        _outputValues[i] = std::log(_prevLayer->_outputValues[i]);
    }
    if (_function == "Elementwise/ReLU")
    {
      for (size_t i = 0; i < N * OC; i++)
        if (_prevLayer->_outputValues[i] > 0.0f)
          _outputValues[i] = _prevLayer->_outputValues[i];
        else
          _outputValues[i] = _prevLayer->_outputValues[i] * _alpha;
    }
    if (_function == "Elementwise/SoftReLU")
    {
      for (size_t i = 0; i < N * OC; i++)
        _outputValues[i] = std::log(1.0f + std::exp(_prevLayer->_outputValues[i]));
    }

    if (_function == "Elementwise/Tanh")
    {
      for (size_t i = 0; i < N * OC; i++)
        _outputValues[i] = std::tanh(_prevLayer->_outputValues[i]);
    }
    if (_function == "Elementwise/Logistic")
    {
      for (size_t i = 0; i < N * OC; i++)
        _outputValues[i] = 1.0f / (1.0f + std::exp(-_prevLayer->_outputValues[i]));
    }
    if (_function == "Elementwise/SoftSign")
    {
      for (size_t i = 0; i < N * OC; i++)
        _outputValues[i] = _prevLayer->_outputValues[i] / (1.0f + std::abs(_prevLayer->_outputValues[i]));
    }
    if (_function == "Softmax")
    {
      for (size_t i = 0; i < N; i++)
      {
        float LSE = logSumExp(&_prevLayer->_outputValues[i * OC], OC);
        for (size_t j = 0; j < OC; j++)
          _outputValues[i * OC + j] = std::exp(_prevLayer->_outputValues[i * OC + j] - LSE);
      }
    }
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Primitive arguments.
    _forwardActivationArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];
    _forwardActivationArgs[DNNL_ARG_DST] = _outputMem[t];

    _forwardActivationPrimitive.execute(_nn->_dnnlStream, _forwardActivationArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    if (_function == "Elementwise/Linear")
    {
      cudaErrCheck(cudaMemcpy(
        _outputTensor[t],
        _prevLayer->_outputTensor[t],
        N * OC * sizeof(float),
        cudaMemcpyDeviceToDevice));
    }
    else if (_function == "Softmax")
    {
      cudnnErrCheck(cudnnSoftmaxForward(
        _nn->_cuDNNHandle,
        CUDNN_SOFTMAX_LOG,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &_alpha,
        _prevLayer->_outputTensorDesc,
        _prevLayer->_outputTensor[t],
        &_beta,
        _outputTensorDesc,
        _outputTensor[t]));
    }
    else
    {
      cudnnErrCheck(cudnnActivationForward(
        _nn->_cuDNNHandle,
        _activationDesc,
        &_alpha,
        _prevLayer->_outputTensorDesc,
        _prevLayer->_outputTensor[t],
        &_beta,
        _outputTensorDesc,
        _outputTensor[t]));
    }
  }
#endif
}

void Activation::backwardData(const size_t t)
{
  size_t N = _batchSize;
  size_t OC = _outputChannels;

  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

  if (_nn->_engine == "Korali")
  {
    if (_function == "Elementwise/Linear")
      for (size_t i = 0; i < N * OC; i++)
        _prevLayer->_outputGradient[i] = _outputGradient[i] * _alpha;

    if (_function == "Elementwise/Log")
      for (size_t i = 0; i < N * OC; i++)
        _prevLayer->_outputGradient[i] = _outputGradient[i] / _prevLayer->_outputValues[i];

    if (_function == "Elementwise/ReLU")
    {
      for (size_t i = 0; i < N * OC; i++)
        if (_prevLayer->_outputValues[i] > 0.0f)
        {
          _prevLayer->_outputGradient[i] = _outputGradient[i];
        }
        else
        {
          _prevLayer->_outputGradient[i] = _outputGradient[i] * _alpha;
        }
    }
    if (_function == "Elementwise/SoftReLU")
      for (size_t i = 0; i < N * OC; i++)
      {
        const float expOutVal = std::exp(_outputValues[i]);
        _prevLayer->_outputGradient[i] = _outputGradient[i] * (expOutVal - 1.0f) / expOutVal;
      }

    if (_function == "Elementwise/Tanh")
      for (size_t i = 0; i < N * OC; i++)
        _prevLayer->_outputGradient[i] = _outputGradient[i] * (1.0f - _outputValues[i] * _outputValues[i]);

    if (_function == "Elementwise/Logistic")
      for (size_t i = 0; i < N * OC; i++)
        _prevLayer->_outputGradient[i] = _outputGradient[i] * _outputValues[i] * (1.0f - _outputValues[i]);

    if (_function == "Elementwise/SoftSign")
      for (size_t i = 0; i < N * OC; i++)
        _prevLayer->_outputGradient[i] = _outputGradient[i] / ((1.0f + std::abs(_outputValues[i])) * (1.0f + std::abs(_outputValues[i])));

    if (_function == "Softmax")
      for (size_t i = 0; i < N * OC; i++)
        _prevLayer->_outputGradient[i] = _outputGradient[i] * _outputValues[i] * (1.0f - _outputValues[i]);
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Primitive arguments.
    _backwardActivationArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];             // Input
    _backwardActivationArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];              // Input
    _backwardActivationArgs[DNNL_ARG_DIFF_SRC] = _prevLayer->_outputGradientMem[t]; // Output
    if (_function == "Softmax") _backwardActivationArgs[DNNL_ARG_DST] = _outputMem[t];

    _backwardActivationPrimitive.execute(_nn->_dnnlStream, _backwardActivationArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    if (_function == "Elementwise/Linear")
    {
      cudaErrCheck(cudaMemcpy(
        _prevLayer->_outputGradientTensor[t],
        _outputGradientTensor[t],
        N * OC * sizeof(float),
        cudaMemcpyDeviceToDevice));
    }
    else if (_function == "Softmax")
    {
      cudnnErrCheck(cudnnSoftmaxBackward(
        _nn->_cuDNNHandle,
        CUDNN_SOFTMAX_LOG,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &_alpha,
        _outputTensorDesc,
        _outputTensor[t],
        _outputTensorDesc,
        _outputGradientTensor[t],
        &_beta,
        _prevLayer->_outputTensorDesc,
        _prevLayer->_outputGradientTensor[t]));
    }
    else
    {
      cudnnErrCheck(cudnnActivationBackward(
        _nn->_cuDNNHandle,
        _activationDesc,
        &_alpha,
        _outputTensorDesc,
        _outputTensor[t],
        _outputTensorDesc,
        _outputGradientTensor[t],
        _prevLayer->_outputTensorDesc,
        _prevLayer->_outputTensor[t],
        &_beta,
        _prevLayer->_outputTensorDesc,
        _prevLayer->_outputGradientTensor[t]));
    }
  }
#endif
}

void Activation::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Function"))
 {
 try { _function = js["Function"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ activation ] \n + Key:    ['Function']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_function == "Elementwise/Clip") validOption = true; 
 if (_function == "Elementwise/Linear") validOption = true; 
 if (_function == "Elementwise/Log") validOption = true; 
 if (_function == "Elementwise/Logistic") validOption = true; 
 if (_function == "Elementwise/ReLU") validOption = true; 
 if (_function == "Elementwise/SoftReLU") validOption = true; 
 if (_function == "Elementwise/SoftSign") validOption = true; 
 if (_function == "Elementwise/Tanh") validOption = true; 
 if (_function == "Softmax") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Function'] required by activation.\n", _function.c_str()); 
}
   eraseValue(js, "Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Function'] required by activation.\n"); 

 if (isDefined(js, "Alpha"))
 {
 try { _alpha = js["Alpha"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ activation ] \n + Key:    ['Alpha']\n%s", e.what()); } 
   eraseValue(js, "Alpha");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Alpha'] required by activation.\n"); 

 if (isDefined(js, "Beta"))
 {
 try { _beta = js["Beta"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ activation ] \n + Key:    ['Beta']\n%s", e.what()); } 
   eraseValue(js, "Beta");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Beta'] required by activation.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/activation";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: activation: \n%s\n", js.dump(2).c_str());
} 

void Activation::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Function"] = _function;
   js["Alpha"] = _alpha;
   js["Beta"] = _beta;
 Layer::getConfiguration(js);
} 

void Activation::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Alpha\": 1.0, \"Beta\": 0.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Activation::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
