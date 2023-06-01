#include "modules/neuralNetwork/layer/linear/linear.hpp"
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

void Linear::initialize()
{
  // Checking Layer size
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("Feed Forward layers cannot be the starting layer of the NN\n");
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("Feed Forward layers cannot be the last layer of the NN\n");
}

std::vector<float> Linear::generateInitialHyperparameters()
{
  std::vector<float> hyperparameters;

  // If this is not the initial layer, calculate hyperparameters for weight and bias operation
  if (_prevLayer != nullptr)
  {
    // Setting value for this layer's xavier constant
    float xavierConstant = std::sqrt(6.0f) / std::sqrt(_outputChannels + _prevLayer->_outputChannels);

    // Adding layer's weights hyperparameter values
    for (size_t i = 0; i < _outputChannels; i++)
      for (size_t j = 0; j < _prevLayer->_outputChannels; j++)
        hyperparameters.push_back(_weightScaling * xavierConstant * _nn->_uniformGenerator->getRandomNumber());

    // Adding layer's bias hyperparameter values
    for (size_t i = 0; i < _outputChannels; i++)
      hyperparameters.push_back(0.0f);
  }

  return hyperparameters;
}

void Linear::createHyperparameterMemory()
{
  // Checking Layer sizes
  ssize_t OC = _outputChannels;
  ssize_t IC = _prevLayer->_outputChannels;

  // Setting hyperparameter count
  _hyperparameterCount = IC * OC + OC;

  if (_nn->_engine == "Korali")
  {
    _weightValues = (float *)malloc(IC * OC * sizeof(float));
    _biasValues = (float *)malloc(OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    memory::dims weightDims = {OC, IC};
    auto weightMemDesc = memory::desc(weightDims, memory::data_type::f32, memory::format_tag::ab);
    _weightsMem = memory(weightMemDesc, _nn->_dnnlEngine);

    auto biasMemDesc = memory::desc({OC}, memory::data_type::f32, memory::format_tag::a);
    _biasMem = memory(biasMemDesc, _nn->_dnnlEngine);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudnnErrCheck(cudnnCreateFilterDescriptor(&_weightsFilterDesc));
    cudnnErrCheck(cudnnSetFilter4dDescriptor(_weightsFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OC, IC, 1, 1));
    cudaErrCheck(cudaMalloc((void **)&_weightsFilter, IC * OC * sizeof(float)));

    cudnnErrCheck(cudnnCreateTensorDescriptor(&_biasTensorDesc));
    cudnnErrCheck(cudnnSetTensor4dDescriptor(_biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, OC, 1, 1));
    cudaErrCheck(cudaMalloc((void **)&_biasTensor, OC * sizeof(float)));
  }
#endif
}

void Linear::copyHyperparameterPointers(Layer *dstLayer)
{
  Linear *dstPtr = dynamic_cast<Linear *>(dstLayer);
  dstPtr->_hyperparameterCount = _hyperparameterCount;

  if (_nn->_engine == "Korali")
  {
    dstPtr->_weightValues = _weightValues;
    dstPtr->_biasValues = _biasValues;
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    dstPtr->_weightsMem = _weightsMem;
    dstPtr->_biasMem = _biasMem;
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    dstPtr->_weightsFilterDesc = _weightsFilterDesc;
    dstPtr->_weightsFilter = _weightsFilter;
    dstPtr->_biasTensorDesc = _biasTensorDesc;
    dstPtr->_biasTensor = _biasTensor;
  }
#endif
}

void Linear::createForwardPipeline()
{
  // Calling base layer function
  Layer::createForwardPipeline();

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // We create the inner product (Wx + b) operation
    auto inner_product_d = inner_product_forward::desc(_propKind, _prevLayer->_outputMem[0].get_desc(), _weightsMem.get_desc(), _biasMem.get_desc(), _outputMem[0].get_desc());

    // Create inner product primitive descriptor.
    dnnl::primitive_attr forwardPrimitiveAttributes;
    _forwardInnerProductPrimitiveDesc = inner_product_forward::primitive_desc(inner_product_d, forwardPrimitiveAttributes, _nn->_dnnlEngine);

    // Create the weights+bias primitive.
    _forwardInnerProductPrimitive = inner_product_forward(_forwardInnerProductPrimitiveDesc);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // Creating convolution operator
    cudnnErrCheck(cudnnCreateConvolutionDescriptor(&_convolutionDesc));
    cudnnErrCheck(cudnnSetConvolution2dDescriptor(_convolutionDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    cudnnErrCheck(cudnnGetConvolutionForwardWorkspaceSize(_nn->_cuDNNHandle, _prevLayer->_outputTensorDesc, _weightsFilterDesc, _convolutionDesc, _outputTensorDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &_convolutionWorkspaceSize));

    _convolutionWorkspace.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++)
      cudaErrCheck(cudaMalloc((void **)&_convolutionWorkspace[t], _convolutionWorkspaceSize * sizeof(float)));
  }
#endif
}

void Linear::createBackwardPipeline()
{
  /*********************************************************************************
   *  Initializing memory objects and primitives for BACKWARD propagation
   *********************************************************************************/

  // Checking Layer sizes
  ssize_t OC = _outputChannels;
  ssize_t IC = _prevLayer->_outputChannels;

  // Calling base layer function
  Layer::createBackwardPipeline();

  if (_nn->_engine == "Korali")
  {
    _weightGradient = (float *)malloc(IC * OC * sizeof(float));
    _biasGradient = (float *)malloc(OC * sizeof(float));
  }

// Creating backward propagation primitives
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    _weightsGradientMem = memory(_weightsMem.get_desc(), _nn->_dnnlEngine);
    _biasGradientMem = memory(_biasMem.get_desc(), _nn->_dnnlEngine);

    auto backwardDataDesc = inner_product_backward_data::desc(
      _prevLayer->_outputGradientMem[0].get_desc(),
      _weightsMem.get_desc(),
      _outputGradientMem[0].get_desc());

    // Create the primitive.
    auto backwardDataPrimitiveDesc = inner_product_backward_data::primitive_desc(backwardDataDesc, _nn->_dnnlEngine, _forwardInnerProductPrimitiveDesc);
    _backwardDataPrimitive = inner_product_backward_data(backwardDataPrimitiveDesc);

    auto backwardWeightsDesc = inner_product_backward_weights::desc(
      _prevLayer->_outputMem[0].get_desc(),
      _weightsMem.get_desc(),
      _biasMem.get_desc(),
      _outputGradientMem[0].get_desc());

    // Create the primitive.
    auto backwardWeightsPrimitiveDesc = inner_product_backward_weights::primitive_desc(backwardWeightsDesc, _nn->_dnnlEngine, _forwardInnerProductPrimitiveDesc);
    _backwardWeightsPrimitive = inner_product_backward_weights(backwardWeightsPrimitiveDesc);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMalloc((void **)&_weightsGradientFilter, IC * OC * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&_biasGradientTensor, OC * sizeof(float)));
  }
#endif
}

void Linear::forwardData(const size_t t)
{
  size_t N = _batchSize;
  size_t IC = _prevLayer->_outputChannels;
  size_t OC = _outputChannels;

  if (_nn->_engine == "Korali")
  {
    // Performing Wx computation
    Map<MatrixXf> matA(_weightValues, IC, OC);
    Map<MatrixXf> matB(_prevLayer->_outputValues, IC, N);
    Map<MatrixXf> matC(_outputValues, OC, N);

    matC = matA.transpose() * matB;

    // Adding Bias
    for (size_t i = 0; i < N; i++)
      for (size_t j = 0; j < OC; j++)
        _outputValues[i * OC + j] += _biasValues[j];
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Arguments to the inner product operation
    std::unordered_map<int, dnnl::memory> forwardInnerProductArgs;
    forwardInnerProductArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];
    forwardInnerProductArgs[DNNL_ARG_WEIGHTS] = _weightsMem;
    forwardInnerProductArgs[DNNL_ARG_BIAS] = _biasMem;
    forwardInnerProductArgs[DNNL_ARG_DST] = _outputMem[t];

    _forwardInnerProductPrimitive.execute(_nn->_dnnlStream, forwardInnerProductArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;
    cudnnErrCheck(cudnnConvolutionForward(
      _nn->_cuDNNHandle,
      &alpha1,
      _prevLayer->_outputTensorDesc,
      _prevLayer->_outputTensor[t],
      _weightsFilterDesc,
      _weightsFilter,
      _convolutionDesc,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      _convolutionWorkspace[t],
      _convolutionWorkspaceSize,
      &alpha2,
      _outputTensorDesc,
      _outputTensor[t]));

    float alpha = 1.0f;
    float beta = 1.0f;
    cudnnAddTensor(_nn->_cuDNNHandle, &alpha, _biasTensorDesc, _biasTensor, &beta, _outputTensorDesc, _outputTensor[t]);
  }
#endif
}

void Linear::backwardData(const size_t t)
{
  int N = _batchSize;
  int IC = _prevLayer->_outputChannels;
  int OC = _outputChannels;

  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

  if (_nn->_engine == "Korali")
  {
    // Backward propagating Wx+b operation
    Map<MatrixXf> matA(_weightValues, IC, OC);
    Map<MatrixXf> matB(_outputGradient, OC, N);
    Map<MatrixXf> matC(_prevLayer->_outputGradient, IC, N);

    matC = matA * matB;
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    _backwardDataArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];             // Input
    _backwardDataArgs[DNNL_ARG_WEIGHTS] = _weightsMem;                        // Input
    _backwardDataArgs[DNNL_ARG_DIFF_SRC] = _prevLayer->_outputGradientMem[t]; // Output

    _backwardDataPrimitive.execute(_nn->_dnnlStream, _backwardDataArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnErrCheck(cudnnConvolutionBackwardData(
      _nn->_cuDNNHandle,
      &alpha,
      _weightsFilterDesc,
      _weightsFilter,
      _outputTensorDesc,
      _outputGradientTensor[t],
      _convolutionDesc,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
      _convolutionWorkspace[t],
      _convolutionWorkspaceSize,
      &beta,
      _prevLayer->_outputTensorDesc,
      _prevLayer->_outputGradientTensor[t]));
  }
#endif
}

void Linear::backwardHyperparameters(size_t t)
{
  const size_t N = _batchSize;
  const size_t IC = _prevLayer->_outputChannels;
  const size_t OC = _outputChannels;

  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer hyperparameter gradient propagation but NN was configured for inference only.\n");

  if (_nn->_engine == "Korali")
  {
    // Performing Weight gradient calculation
    Map<MatrixXf> matA(_prevLayer->_outputValues, IC, N);
    Map<MatrixXf> matB(_outputGradient, OC, N);
    Map<MatrixXf> matC(_weightGradient, IC, OC);

    matC = matA * matB.transpose();

    // Setting the bias values to all minibatch inputs
    for (size_t j = 0; j < OC; j++) _biasGradient[j] = _outputGradient[0 * OC + j];
    for (size_t i = 1; i < N; i++)
      for (size_t j = 0; j < OC; j++) _biasGradient[j] += _outputGradient[i * OC + j];
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Arguments for the backward propagation of the gradient wrt Weights and Biases
    std::unordered_map<int, dnnl::memory> backwardWeightsArgs;
    backwardWeightsArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];    // Input
    backwardWeightsArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];   // Input
    backwardWeightsArgs[DNNL_ARG_DIFF_WEIGHTS] = _weightsGradientMem; // Output
    backwardWeightsArgs[DNNL_ARG_DIFF_BIAS] = _biasGradientMem;       // Output

    _backwardWeightsPrimitive.execute(_nn->_dnnlStream, backwardWeightsArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnErrCheck(cudnnConvolutionBackwardBias(
      _nn->_cuDNNHandle,
      &alpha,
      _outputTensorDesc,
      _outputGradientTensor[t],
      &beta,
      _biasTensorDesc,
      _biasGradientTensor));

    cudnnErrCheck(cudnnConvolutionBackwardFilter(
      _nn->_cuDNNHandle,
      &alpha,
      _prevLayer->_outputTensorDesc,
      _prevLayer->_outputTensor[t],
      _outputTensorDesc,
      _outputGradientTensor[t],
      _convolutionDesc,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
      _convolutionWorkspace[t],
      _convolutionWorkspaceSize,
      &beta,
      _weightsFilterDesc,
      _weightsGradientFilter));
  }
#endif
}

void Linear::setHyperparameters(const float *hyperparameters)
{
  size_t IC = _prevLayer->_outputChannels;
  size_t OC = _outputChannels;

  if (_nn->_engine == "Korali")
  {
    memcpy(_weightValues, &hyperparameters[0], IC * OC * sizeof(float));
    memcpy(_biasValues, &hyperparameters[IC * OC], OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    write_to_dnnl_memory(&hyperparameters[0], _weightsMem);
    write_to_dnnl_memory(&hyperparameters[IC * OC], _biasMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMemcpy(_weightsFilter, &hyperparameters[0], IC * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(_biasTensor, &hyperparameters[IC * OC], OC * sizeof(float), cudaMemcpyHostToDevice));
  }
#endif
}

void Linear::getHyperparameters(float *hyperparameters)
{
  size_t IC = _prevLayer->_outputChannels;
  size_t OC = _outputChannels;

  if (_nn->_engine == "Korali")
  {
    memcpy(&hyperparameters[0], _weightValues, IC * OC * sizeof(float));
    memcpy(&hyperparameters[IC * OC], _biasValues, OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&hyperparameters[0], _weightsMem);
    read_from_dnnl_memory(&hyperparameters[IC * OC], _biasMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMemcpy(&hyperparameters[0], _weightsFilter, IC * OC * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(&hyperparameters[IC * OC], _biasTensor, OC * sizeof(float), cudaMemcpyDeviceToHost));
  }
#endif
}

void Linear::getHyperparameterGradients(float *gradient)
{
  size_t IC = _prevLayer->_outputChannels;
  size_t OC = _outputChannels;

  if (_nn->_engine == "Korali")
  {
    memcpy(&gradient[0], _weightGradient, IC * OC * sizeof(float));
    memcpy(&gradient[IC * OC], _biasGradient, OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&gradient[0], _weightsGradientMem);
    read_from_dnnl_memory(&gradient[IC * OC], _biasGradientMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMemcpy(&gradient[0], _weightsGradientFilter, IC * OC * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(&gradient[IC * OC], _biasGradientTensor, OC * sizeof(float), cudaMemcpyDeviceToHost));
  }
#endif
}

void Linear::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Layer::setConfiguration(js);
 _type = "layer/linear";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: linear: \n%s\n", js.dump(2).c_str());
} 

void Linear::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Layer::getConfiguration(js);
} 

void Linear::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Linear::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
