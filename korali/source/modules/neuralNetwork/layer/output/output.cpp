#include "modules/neuralNetwork/layer/output/output.hpp"
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

void Output::initialize()
{
  if (_index != _nn->_layers.size() - 1)
    KORALI_LOG_ERROR("Output layers can only be placed at the last position in the NN\n");

  // The node count for this layer should be the same as the previous layer
  _outputChannels = _prevLayer->_outputChannels;

  // Checking Layer size
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

  // Check output scaling configuration
  if (_scale.empty() == false)
    if (_scale.size() != _outputChannels)
      KORALI_LOG_ERROR("Wrong number of output scaling factors passed to the neural network. Expected: %lu, provided: %lu.\n", _outputChannels, _scale.size());

  // Check output shifting configuration
  if (_shift.empty() == false)
    if (_shift.size() != _outputChannels)
      KORALI_LOG_ERROR("Wrong number of output shift factors passed to the neural network. Expected: %lu, provided: %lu.\n", _outputChannels, _shift.size());

  // Initializing transformation vector to know what operations to apply, without the need of constant string comparison
  _transformationVector.resize(_outputChannels);

  // If no transformation mask is specified, use identity exclusively
  if (_transformationMask.size() == 0)
    for (size_t i = 0; i < _outputChannels; i++)
      _transformationVector[i] = t_identity;

  // If incorrect size of mask is specified, produce error
  if (_transformationMask.size() != 0 && _transformationMask.size() != _outputChannels)
    KORALI_LOG_ERROR("Wrong size of transformation mask specified: %lu, expected: %lu.\n", _transformationMask.size(), _outputChannels);

  // If transformation mask is specified, storing the correct transformations
  if (_transformationMask.size() == _outputChannels)
    for (size_t i = 0; i < _outputChannels; i++)
    {
      bool isRecognized = false;
      if (_transformationMask[i] == "Identity")
      {
        _transformationVector[i] = t_identity;
        isRecognized = true;
      }
      if (_transformationMask[i] == "Absolute")
      {
        _transformationVector[i] = t_absolute;
        isRecognized = true;
      }
      if (_transformationMask[i] == "Softplus")
      {
        _transformationVector[i] = t_softplus;
        isRecognized = true;
      }
      if (_transformationMask[i] == "Tanh")
      {
        _transformationVector[i] = t_tanh;
        isRecognized = true;
      }
      if (_transformationMask[i] == "Sigmoid")
      {
        _transformationVector[i] = t_sigmoid;
        isRecognized = true;
      }
      if (isRecognized == false) KORALI_LOG_ERROR("Wrong transformation mask specified: %s for output variable %lu.\n", _transformationMask[i].c_str(), i);
    }
}

void Output::createForwardPipeline()
{
  size_t N = _batchSize;
  size_t IC = _prevLayer->_outputChannels;
  size_t OC = _outputChannels;

  if (IC != OC) KORALI_LOG_ERROR("Output layers node count (%lu) is different than its previous layer's (%lu)\n", OC, IC);

  // Calling base layer function
  Layer::createForwardPipeline();

  // Allocating storage for pre-processing data
  _srcOutputValues = (float *)malloc(N * OC * sizeof(float));
}

void Output::createBackwardPipeline()
{
  size_t N = _batchSize;
  size_t IC = _prevLayer->_outputChannels;
  size_t OC = _outputChannels;

  if (IC != OC) KORALI_LOG_ERROR("Output layers node count (%lu) is different than its previous layer's (%lu)\n", OC, IC);

  // Calling base layer function
  Layer::createBackwardPipeline();

  // Allocating storage for pre-processing data
  _dstOutputGradients = (float *)malloc(N * OC * sizeof(float));
}

void Output::forwardData(const size_t t)
{
  size_t N = _batchSize;
  size_t OC = _outputChannels;

  // Copying previous layer's output to this layer's output
  if (_nn->_engine == "Korali")
  {
    memcpy(_srcOutputValues, _prevLayer->_outputValues, N * OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(_srcOutputValues, _prevLayer->_outputMem[t]);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMemcpy(_srcOutputValues, _prevLayer->_outputTensor[t], N * OC * sizeof(float), cudaMemcpyDeviceToHost));
  }
#endif

  // Performing postprocessing
#pragma omp parallel for
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < OC; j++)
    {
      auto x = _srcOutputValues[i * OC + j];

      // Apply selected transformation now
      if (_transformationVector[j] == t_absolute) x = std::fabs(x);
      if (_transformationVector[j] == t_sigmoid) x = 1. / (1. + std::exp(-x));
      if (_transformationVector[j] == t_softplus) x = 0.5 * (x + std::sqrt(1. + x * x));
      if (_transformationVector[j] == t_tanh) x = std::tanh(x);

      // If we  use scaling, then apply the scaling factors now
      if (_scale.size() > 0) x *= _scale[j];

      // If we  use shifting, then apply the shifting now
      if (_shift.size() > 0) x += _shift[j];

      // Saving result to NN's output directly
      size_t pos = (t * N * OC) + (i * OC) + j;
      _pipeline->_rawOutputValues[pos] = x;
    }
}

void Output::backwardData(const size_t t)
{
  const size_t N = _batchSize;
  const size_t OC = _outputChannels;

  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

// Performing gradient pre-processing
#pragma omp parallel for
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < OC; j++)
    {
      // Getting forward propagation output and passed gradients
      size_t pos = (t * N * OC) + (i * OC) + j;
      float x = _pipeline->_rawOutputValues[pos];
      float g = _pipeline->_rawOutputGradients[pos];

      // If we use shift, then apply the inverse shift to the forward output
      if (_shift.size() > 0)
        x = x - _shift[j];

      // If we use scaling, then apply the inverse scaling factors gradient now
      if (_scale.size() > 0)
      {
        x = x / _scale[j];
        g = g * _scale[j];
      }

      // Apply backward operation on the selected transformations now
      if (_transformationVector[j] == t_absolute)
        if (std::signbit(x) != std::signbit(_srcOutputValues[i * OC + j]))
          g = -g;

      if (_transformationVector[j] == t_tanh)
        g = g * (1.0f - x * x);

      if (_transformationVector[j] == t_softplus)
      {
        float nnx = x - 0.25 / x;
        g = g * 0.5 * (1. + nnx / std::sqrt(nnx * nnx + 1));
      }

      if (_transformationVector[j] == t_sigmoid)
        g = g * x * (1. - x);

      _dstOutputGradients[i * OC + j] = g;
    }

  if (_nn->_engine == "Korali")
  {
    memcpy(_prevLayer->_outputGradient, _dstOutputGradients, N * OC * sizeof(float));
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    write_to_dnnl_memory(_dstOutputGradients, _prevLayer->_outputGradientMem[t]);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMemcpy(_prevLayer->_outputGradientTensor[t], _dstOutputGradients, N * OC * sizeof(float), cudaMemcpyHostToDevice));
  }
#endif
}

void Output::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Transformation Mask"))
 {
 try { _transformationMask = js["Transformation Mask"].get<std::vector<std::string>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ output ] \n + Key:    ['Transformation Mask']\n%s", e.what()); } 
   eraseValue(js, "Transformation Mask");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Transformation Mask'] required by output.\n"); 

 if (isDefined(js, "Scale"))
 {
 try { _scale = js["Scale"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ output ] \n + Key:    ['Scale']\n%s", e.what()); } 
   eraseValue(js, "Scale");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Scale'] required by output.\n"); 

 if (isDefined(js, "Shift"))
 {
 try { _shift = js["Shift"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ output ] \n + Key:    ['Shift']\n%s", e.what()); } 
   eraseValue(js, "Shift");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Shift'] required by output.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/output";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: output: \n%s\n", js.dump(2).c_str());
} 

void Output::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Transformation Mask"] = _transformationMask;
   js["Scale"] = _scale;
   js["Shift"] = _shift;
 Layer::getConfiguration(js);
} 

void Output::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Scale\": [], \"Shift\": [], \"Transformation Mask\": []}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Output::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
