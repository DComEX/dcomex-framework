#include "modules/neuralNetwork/layer/layer.hpp"
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
;

void Layer::createForwardPipeline()
{
  // Obtaining batch size
  ssize_t N = _batchSize;

  // Checking Layer sizes
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);
  ssize_t OC = _outputChannels;

  if (_nn->_engine == "Korali")
  {
    _outputValues = (float *)malloc(N * OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Setting propagation kind
    _propKind = _nn->_mode == "Training" ? prop_kind::forward_training : prop_kind::forward_inference;

    // Creating layer's data memory storage
    const memory::dims layerDims = {N, OC};
    auto dataMemDesc = memory::desc(layerDims, memory::data_type::f32, memory::format_tag::nc);

    // Creating activation layer memory
    _outputMem.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++)
      _outputMem[t] = memory(dataMemDesc, _nn->_dnnlEngine);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // Setting propagation mode
    _forwardMode = _nn->_mode == "Training" ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;

    cudnnErrCheck(cudnnCreateTensorDescriptor(&_outputTensorDesc));
    cudnnErrCheck(cudnnSetTensor4dDescriptor(_outputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OC, 1, 1));

    _outputTensor.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++)
      cudaErrCheck(cudaMalloc((void **)&_outputTensor[t], N * OC * sizeof(float)));
  }
#endif
}

std::vector<std::vector<float>> Layer::getOutput()
{
  size_t N = _batchSize;
  size_t OC = _outputChannels;

  std::vector<float> outputVals(N * OC);

  // Copying previous layer's output to this layer's output
  if (_nn->_engine == "Korali")
  {
    memcpy(outputVals.data(), _outputValues, N * OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    int t = _nn->_timestepCount - 1;
    read_from_dnnl_memory(outputVals.data(), _outputMem[t]);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    int t = _nn->_timestepCount - 1;
    cudaErrCheck(cudaMemcpy(outputVals.data(), _outputTensor[t], N * OC * sizeof(float), cudaMemcpyDeviceToHost));
  }
#endif

  std::vector<std::vector<float>> outputVector(N);
  for (size_t i = 0; i < N; i++) outputVector[i].resize(OC);
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < OC; j++)
      outputVector[i][j] = outputVals[i * OC + j];

  return outputVector;
}

void Layer::createBackwardPipeline()
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting layer's backward pipeline creation but NN was configured for inference only.\n");

  /*********************************************************************************
   *  Initializing memory objects and primitives for BACKWARD propagation
   *********************************************************************************/

  size_t N = _batchSize;
  size_t OC = _outputChannels;

  if (_nn->_engine == "Korali")
  {
    _outputGradient = (float *)malloc(N * OC * sizeof(float));
  }

// Creating backward propagation primitives for activation functions
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Creating data-related gradient memory
    _outputGradientMem.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++)
      _outputGradientMem[t] = memory(_outputMem[t].get_desc(), _nn->_dnnlEngine);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    _outputGradientTensor.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++)
      cudaErrCheck(cudaMalloc((void **)&_outputGradientTensor[i], N * OC * sizeof(float)));
  }
#endif
}

std::vector<float> Layer::generateInitialHyperparameters()
{
  std::vector<float> hyperparameters;
  return hyperparameters;
}

void Layer::createHyperparameterMemory()
{
  _hyperparameterCount = 0;
}

void Layer::backwardHyperparameters(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer hyperparameter gradient propagation but NN was configured for inference only.\n");
};

void Layer::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Output Channels"))
 {
 try { _outputChannels = js["Output Channels"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ layer ] \n + Key:    ['Output Channels']\n%s", e.what()); } 
   eraseValue(js, "Output Channels");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Output Channels'] required by layer.\n"); 

 if (isDefined(js, "Weight Scaling"))
 {
 try { _weightScaling = js["Weight Scaling"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ layer ] \n + Key:    ['Weight Scaling']\n%s", e.what()); } 
   eraseValue(js, "Weight Scaling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Weight Scaling'] required by layer.\n"); 

 Module::setConfiguration(js);
 _type = "layer";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: layer: \n%s\n", js.dump(2).c_str());
} 

void Layer::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Output Channels"] = _outputChannels;
   js["Weight Scaling"] = _weightScaling;
 Module::getConfiguration(js);
} 

void Layer::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Output Channels\": 0, \"Weight Scaling\": 1.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Module::applyModuleDefaults(js);
} 

void Layer::applyVariableDefaults() 
{

 Module::applyVariableDefaults();
} 

;

} //neuralNetwork
} //korali
;
