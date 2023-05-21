#include "modules/neuralNetwork/layer/input/input.hpp"
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

void Input::initialize()
{
  // Checking Layer size
  if (_outputChannels == 0) KORALI_LOG_ERROR("Output Channels for layer (%lu) should be larger than zero.\n", _index);

  // Checking position
  if (_index != 0) KORALI_LOG_ERROR("Input layers can only be placed at the start of the NN\n");
}

void Input::forwardData(const size_t t)
{
  size_t N = _batchSize;
  size_t OC = _outputChannels;

  if (_nn->_engine == "Korali")
  {
    memcpy(_outputValues, &_pipeline->_rawInputValues[t * N * OC], N * OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    write_to_dnnl_memory(&_pipeline->_rawInputValues[t * N * OC], _outputMem[t]);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMemcpy(_outputTensor[t], &_pipeline->_rawInputValues[t * N * OC], N * OC * sizeof(float), cudaMemcpyHostToDevice));
  }
#endif
}

void Input::backwardData(const size_t t)
{
  int N = _batchSize;
  int OC = _outputChannels;

  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

  if (_nn->_engine == "Korali")
  {
    memcpy(&_pipeline->_rawInputGradients[t * N * OC], _outputGradient, N * OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&_pipeline->_rawInputGradients[t * N * OC], _outputGradientMem[t]);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMemcpy(&_pipeline->_rawInputGradients[t * N * OC], _outputGradientTensor[t], N * OC * sizeof(float), cudaMemcpyDeviceToHost));
  }
#endif
}

void Input::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Layer::setConfiguration(js);
 _type = "layer/input";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: input: \n%s\n", js.dump(2).c_str());
} 

void Input::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Layer::getConfiguration(js);
} 

void Input::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Input::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
