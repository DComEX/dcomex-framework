#include "modules/neuralNetwork/layer/recurrent/recurrent.hpp"
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

void Recurrent::initialize()
{
  // Checking Layer size
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer %lu should be larger than zero.\n", _index);

  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("Recurrent layers cannot be the starting layer of the NN\n");
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("Recurrent layers cannot be the last layer of the NN\n");

  // If using depth > 1, the input layer channels must be consistent
  const size_t IC = _prevLayer->_outputChannels;
  const size_t OC = _outputChannels;
  if (IC != OC) KORALI_LOG_ERROR("Channel count (%lu) for LSTM layer %lu should be the same as that of the previous layer (%lu).\n", OC, _index, IC);

  if (_nn->_engine == "Korali")
    KORALI_LOG_ERROR("Recurrent layers are not yet supported by the Korali NN backend, use OneDNN or CuDNN.\n");
}

void Recurrent::createHyperparameterMemory()
{
  // Getting dimensions
  const size_t L = _depth;     // Physical Layers
  const size_t D = 1;          // Directions
  const size_t G = _gateCount; // Gates
  const size_t IC = _prevLayer->_outputChannels;
  const size_t OC = _outputChannels;

  // Setting hyperparameter count
  const size_t weightsInputCount = L * D * G * IC * OC;
  const size_t weightsRecurrentCount = L * D * G * OC * OC;
  const size_t biasCount = L * D * G * OC;
  _hyperparameterCount = weightsInputCount + weightsRecurrentCount + biasCount;

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Checking Layer sizes
    const memory::dim IC = _prevLayer->_outputChannels; // channels
    const memory::dim OC = _outputChannels;             // channels
    const memory::dim G = _gateCount;                   // gates
    const memory::dim L = _depth;                       // layers
    const memory::dim D = 1;                            // directions

    memory::dims weightInputDims = {L, D, IC, G, OC};
    auto weightInputMemDesc = memory::desc(weightInputDims, memory::data_type::f32, memory::format_tag::ldigo);

    _weightsLayerMem = memory(weightInputMemDesc, _nn->_dnnlEngine);

    memory::dims weightRecurrentDims = {L, D, OC, G, OC};
    auto weightRecurrentMemDesc = memory::desc(weightRecurrentDims, memory::data_type::f32, memory::format_tag::ldigo);

    _weightsRecurrentMem = memory(weightRecurrentMemDesc, _nn->_dnnlEngine);

    memory::dims bias_dims = {L, D, G, OC};
    auto biasMemDesc = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::ldgo);

    _biasMem = memory(biasMemDesc, _nn->_dnnlEngine);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // Creating dropout operator and its memory
    size_t seed = _nn->_k->_randomSeed++; // Pick a seed.
    cudnnErrCheck(cudnnCreateDropoutDescriptor(&_dropoutDesc));
    cudnnErrCheck(cudnnSetDropoutDescriptor(_dropoutDesc,
                                            _nn->_cuDNNHandle,
                                            0.0f,
                                            NULL,
                                            0,
                                            seed));

    // Creating RNN operator
    cudnnErrCheck(cudnnCreateRNNDescriptor(&_rnnDesc));
    cudnnErrCheck(cudnnSetRNNDescriptor_v8(_rnnDesc,
                                           CUDNN_RNN_ALGO_STANDARD,
                                           _rnnMode,
                                           CUDNN_RNN_SINGLE_REC_BIAS,
                                           CUDNN_UNIDIRECTIONAL,
                                           CUDNN_LINEAR_INPUT,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_DEFAULT_MATH,
                                           _prevLayer->_outputChannels,
                                           _outputChannels,
                                           _outputChannels,
                                           L, // Pseudo Layer Count
                                           _dropoutDesc,
                                           CUDNN_RNN_PADDED_IO_DISABLED));

    // Allocating memory for hyperparameters
    cudnnErrCheck(cudnnGetRNNWeightSpaceSize(_nn->_cuDNNHandle, _rnnDesc, &_weightsSize));

    // The number of hyperparameters is then the workspace size divided by the size of a float
    if (_hyperparameterCount != _weightsSize / sizeof(float))
      KORALI_LOG_ERROR("CuDNN - Non-consistent weights size with expected hyperparameters. Given: %lu, Expected: %lu\n", _hyperparameterCount, _weightsSize / sizeof(float));

    // Creating memory for hyperparameters and their gradients
    cudaErrCheck(cudaMalloc((void **)&_weightsTensor, _weightsSize));

    // Allocating space to store pointers to hyperparameters and their sizes
    _gateWeightTensors.resize(_gateCount);
    _gateBiasTensors.resize(_gateCount);

    // Getting pointers and length of all of the RNN hyperparameters
    for (size_t gateId = 0; gateId < _gateCount; gateId++)
    {
      cudnnTensorDescriptor_t gateWeightDesc;
      cudnnTensorDescriptor_t gateBiasDesc;
      cudnnErrCheck(cudnnCreateTensorDescriptor(&gateWeightDesc));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&gateBiasDesc));

      cudnnErrCheck(cudnnGetRNNWeightParams(_nn->_cuDNNHandle,
                                            _rnnDesc,
                                            0,
                                            _weightsSize,
                                            _weightsTensor,
                                            gateId,
                                            gateWeightDesc,
                                            &_gateWeightTensors[gateId],
                                            gateBiasDesc,
                                            &_gateBiasTensors[gateId]));

      cudnnErrCheck(cudnnDestroyTensorDescriptor(gateWeightDesc));
      cudnnErrCheck(cudnnDestroyTensorDescriptor(gateBiasDesc));
    }
  }
#endif
}

void Recurrent::copyHyperparameterPointers(Layer *dstLayer)
{
  Recurrent *dstPtr = dynamic_cast<Recurrent *>(dstLayer);
  dstPtr->_hyperparameterCount = _hyperparameterCount;

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    dstPtr->_weightsLayerMem = _weightsLayerMem;
    dstPtr->_weightsRecurrentMem = _weightsRecurrentMem;
    dstPtr->_biasMem = _biasMem;
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    dstPtr->_rnnDesc = _rnnDesc;
    dstPtr->_dropoutDesc = _dropoutDesc;
    dstPtr->_weightsSize = _weightsSize;
    dstPtr->_weightsTensor = _weightsTensor;
    dstPtr->_gateWeightTensors = _gateWeightTensors;
    dstPtr->_gateBiasTensors = _gateBiasTensors;
  }
#endif
}

std::vector<float> Recurrent::generateInitialHyperparameters()
{
  std::vector<float> hyperparameters;

  // Getting dimensions
  const size_t L = _depth;
  const size_t G = _gateCount;
  const size_t IC = _prevLayer->_outputChannels;
  const size_t OC = _outputChannels;

  // Calculate hyperparameters for weight and bias of all linear layers
  // Setting value for this layer's xavier constant
  float xavierConstant = (_weightScaling * sqrtf(6.0f)) / sqrt(_outputChannels + _prevLayer->_outputChannels);

  // Weights applied to the input layer(s)
  for (size_t layerId = 0; layerId < L; layerId++)
    for (size_t gateId = 0; gateId < G; gateId++)
      for (size_t i = 0; i < IC * OC; i++)
        hyperparameters.push_back(xavierConstant * _nn->_uniformGenerator->getRandomNumber());

  // Weights applied to the recurrent layer
  for (size_t layerId = 0; layerId < L; layerId++)
    for (size_t gateId = 0; gateId < G; gateId++)
      for (size_t i = 0; i < OC * OC; i++)
        hyperparameters.push_back(xavierConstant * _nn->_uniformGenerator->getRandomNumber());

  // Bias for the recurrent layer
  for (size_t layerId = 0; layerId < L; layerId++)
    for (size_t gateId = 0; gateId < G; gateId++)
      for (size_t i = 0; i < OC; i++)
        hyperparameters.push_back(0.0f);

  return hyperparameters;
}

void Recurrent::createBackwardPipeline()
{
  /*********************************************************************************
   *  Initializing memory objects and primitives for BACKWARD propagation
   *********************************************************************************/

  // Calling base layer function
  Layer::createBackwardPipeline();

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    _weightsLayerGradientMem = memory(_weightsLayerMem.get_desc(), _nn->_dnnlEngine);
    _weightsRecurrentGradientMem = memory(_weightsRecurrentMem.get_desc(), _nn->_dnnlEngine);
    _biasGradientMem = memory(_biasMem.get_desc(), _nn->_dnnlEngine);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMalloc((void **)&_weightsGradientTensor, _weightsSize));
  }
#endif
}

void Recurrent::backwardHyperparameters(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer hyperparameter gradient propagation but NN was configured for inference only.\n");

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Nothing to do here, weights and bias gradients have been generated already by backwardData.
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // We need to clear the gradients because they are additive in CUDNN
    cudaErrCheck(cudaMemset(_weightsGradientTensor, 0, _hyperparameterCount));

    cudnnErrCheck(cudnnRNNBackwardWeights_v8(
      _nn->_cuDNNHandle,                    // handle
      _rnnDesc,                             // rnnDesc
      CUDNN_WGRAD_MODE_ADD,                 // addGrad
      _devSequenceLengths,                  // devSeqLengths
      _inputRNNDataDesc,                    // xDesc
      _prevLayer->_outputTensor[t],         // x
      _hTensorDesc,                         // hDesc
      t == 0 ? NULL : _hStateTensor[t - 1], // hx
      _outputRNNDataDesc,                   // yDesc
      _outputTensor[t],                     // y
      _weightsSize,
      _weightsGradientTensor,
      _workSpaceSize,
      _workSpaceTensor[t],
      _reserveSpaceSize,
      _reserveSpaceTensor[t]));
  }
#endif
}

void Recurrent::setHyperparameters(const float *hyperparameters)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Getting dimensions
    const size_t L = _depth;
    const size_t G = _gateCount;
    const size_t IC = _prevLayer->_outputChannels;
    const size_t OC = _outputChannels;
    const size_t D = 1; // directions

    write_to_dnnl_memory(&hyperparameters[0], _weightsLayerMem);
    write_to_dnnl_memory(&hyperparameters[L * D * G * IC * OC], _weightsRecurrentMem);
    write_to_dnnl_memory(&hyperparameters[L * D * G * OC * OC + L * D * G * IC * OC], _biasMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN") cudaErrCheck(cudaMemcpy(_weightsTensor, hyperparameters, _weightsSize, cudaMemcpyHostToDevice));
#endif
}

void Recurrent::getHyperparameters(float *hyperparameters)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Getting dimensions
    const size_t L = _depth;
    const size_t G = _gateCount;
    const size_t IC = _prevLayer->_outputChannels;
    const size_t OC = _outputChannels;
    const size_t D = 1; // directions

    read_from_dnnl_memory(&hyperparameters[0], _weightsLayerMem);
    read_from_dnnl_memory(&hyperparameters[L * D * G * IC * OC], _weightsRecurrentMem);
    read_from_dnnl_memory(&hyperparameters[L * D * G * OC * OC + L * D * G * IC * OC], _biasMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN") cudaErrCheck(cudaMemcpy(hyperparameters, _weightsTensor, _weightsSize, cudaMemcpyDeviceToHost));
#endif
}

void Recurrent::getHyperparameterGradients(float *gradient)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Getting dimensions
    const size_t L = _depth;
    const size_t G = _gateCount;
    const size_t IC = _prevLayer->_outputChannels;
    const size_t OC = _outputChannels;
    const size_t D = 1; // directions

    read_from_dnnl_memory(&gradient[0], _weightsLayerGradientMem);
    read_from_dnnl_memory(&gradient[L * D * G * IC * OC], _weightsRecurrentGradientMem);
    read_from_dnnl_memory(&gradient[L * D * G * OC * OC + L * D * G * IC * OC], _biasGradientMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN") cudaErrCheck(cudaMemcpy(gradient, _weightsGradientTensor, _weightsSize, cudaMemcpyDeviceToHost));
#endif
}

void Recurrent::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Depth"))
 {
 try { _depth = js["Depth"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ recurrent ] \n + Key:    ['Depth']\n%s", e.what()); } 
   eraseValue(js, "Depth");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Depth'] required by recurrent.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/recurrent";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: recurrent: \n%s\n", js.dump(2).c_str());
} 

void Recurrent::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Depth"] = _depth;
 Layer::getConfiguration(js);
} 

void Recurrent::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Depth\": 1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Recurrent::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
