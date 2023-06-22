#include "modules/neuralNetwork/layer/deconvolution/deconvolution.hpp"
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

void Deconvolution::initialize()
{
  // Checking Layer size
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("Deconvolutional layers cannot be the starting layer of the NN\n");
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("Deconvolutional layers cannot be the last layer of the NN\n");

  // Precalculating values for the deconvolution operation
  N = _batchSize;
  OH = _imageHeight;
  OW = _imageWidth;
  KH = _kernelHeight;
  KW = _kernelWidth;

  SV = _verticalStride;
  SH = _horizontalStride;
  PT = _paddingTop;
  PL = _paddingLeft;
  PB = _paddingBottom;
  PR = _paddingRight;

  // Check for non zeros
  if (OH <= 0) KORALI_LOG_ERROR("Image height must be larger than zero for deconvolutional layer.\n");
  if (OW <= 0) KORALI_LOG_ERROR("Image width must be larger than zero for deconvolutional layer.\n");
  if (KH <= 0) KORALI_LOG_ERROR("Kernel height must be larger than zero for deconvolutional layer.\n");
  if (KW <= 0) KORALI_LOG_ERROR("Kernel width must be larger than zero for deconvolutional layer.\n");
  if (SV <= 0) KORALI_LOG_ERROR("Vertical stride must be larger than zero for deconvolutional layer.\n");
  if (SH <= 0) KORALI_LOG_ERROR("Horizontal stride must be larger than zero for deconvolutional layer.\n");

  // Several sanity checks
  if (KH > OH) KORALI_LOG_ERROR("Kernel height cannot be larger than output image height.\n");
  if (KW > OW) KORALI_LOG_ERROR("Kernel height cannot be larger than output image height.\n");
  if (PR + PL > OW) KORALI_LOG_ERROR("L+R Paddings cannot exceed the width of the output image.\n");
  if (PT + PB > OH) KORALI_LOG_ERROR("T+B Paddings cannot exceed the height of the output image.\n");

  // Check whether the output channels of the previous layer is divided by the height and width
  if (_outputChannels % (OH * OW) > 0) KORALI_LOG_ERROR("Deconvolutional layer contains a number of channels (%lu) not divisible by the 2D HxW setup (%lux%lu).\n", _outputChannels, OH, OW);
  OC = _outputChannels / (OH * OW);

  // Deriving output height and width
  IH = (OH - KH + PT + PB) / SV + 1;
  IW = (OW - KW + PR + PL) / SH + 1;

  // Check whether the output channels of the previous layer is divided by the height and width
  if (_prevLayer->_outputChannels % (IH * IW) > 0) KORALI_LOG_ERROR("Previous layer to the convolutional layer contains a number of output channels (%lu) not divisible by the image size (%lux%lu) given kernel (%lux%lu) size and padding/stride configuration.\n", _prevLayer->_outputChannels, IH, IW, KH, KW);
  IC = _prevLayer->_outputChannels / (IH * IW);
}

std::vector<float> Deconvolution::generateInitialHyperparameters()
{
  std::vector<float> hyperparameters;
  size_t weightCount = OC * IC * KH * KW;
  size_t biasCount = OC;

  // If this is not the initial layer, calculate hyperparameters for weight and bias operation
  if (_prevLayer != nullptr)
  {
    // Setting value for this layer's xavier constant
    float xavierConstant = std::sqrt(6.0f) / std::sqrt(_outputChannels + _prevLayer->_outputChannels);

    // Adding layer's weights hyperparameter values
    for (size_t i = 0; i < weightCount; i++)
      hyperparameters.push_back(_weightScaling * xavierConstant * _nn->_uniformGenerator->getRandomNumber());

    // Adding layer's bias hyperparameter values
    for (size_t i = 0; i < biasCount; i++)
      hyperparameters.push_back(0.0f);
  }

  return hyperparameters;
}

void Deconvolution::createHyperparameterMemory()
{
  // Setting hyperparameter count
  size_t weightCount = OC * IC * KH * KW;
  size_t biasCount = OC;

  _hyperparameterCount = weightCount + biasCount;

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    memory::dims weightDims = {OC, IC, KH, KW};
    auto weightMemDesc = memory::desc(weightDims, memory::data_type::f32, memory::format_tag::oihw);
    _weightsMem = memory(weightMemDesc, _nn->_dnnlEngine);

    auto biasMemDesc = memory::desc({OC}, memory::data_type::f32, memory::format_tag::a);
    _biasMem = memory(biasMemDesc, _nn->_dnnlEngine);
  }
#endif
}

void Deconvolution::copyHyperparameterPointers(Layer *dstLayer)
{
  Deconvolution *dstPtr = dynamic_cast<Deconvolution *>(dstLayer);
  dstPtr->_hyperparameterCount = _hyperparameterCount;

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    dstPtr->_weightsMem = _weightsMem;
    dstPtr->_biasMem = _biasMem;
  }
#endif
}

void Deconvolution::createForwardPipeline()
{
  // Calling base layer function
  Layer::createForwardPipeline();

  if (_nn->_engine == "Korali") KORALI_LOG_ERROR("Deconvolutional Layers still not supported in Korali's NN backend. Use OneDNN.\n");
  if (_nn->_engine == "CuDNN") KORALI_LOG_ERROR("Deconvolutional Layers still not supported in CuDNNbackend. Use OneDNN.\n");

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Creating memory descriptor mappings for input memory
    _srcMemDesc = memory::desc({N, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nchw);
    _dstMemDesc = memory::desc({N, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nchw);

    // Creating padding dims
    memory::dims ST = {SV, SH};  // Horizontal Vertical
    memory::dims PTL = {PT, PL}; // Top Left
    memory::dims PBR = {PB, PR}; // Bottom Right

    // We create the deconvolution operation
    auto deconvolution_d = deconvolution_forward::desc(_propKind, algorithm::deconvolution_direct, _srcMemDesc, _weightsMem.get_desc(), _biasMem.get_desc(), _dstMemDesc, ST, PTL, PBR);

    // Create inner product primitive descriptor.
    dnnl::primitive_attr deconvolutionPrimitiveAttributes;
    _forwardDeconvolutionPrimitiveDesc = deconvolution_forward::primitive_desc(deconvolution_d, deconvolutionPrimitiveAttributes, _nn->_dnnlEngine);

    // Create the weights+bias primitive.
    _forwardDeconvolutionPrimitive = deconvolution_forward(_forwardDeconvolutionPrimitiveDesc);
  }
#endif
}

void Deconvolution::createBackwardPipeline()
{
  //  Initializing memory objects and primitives for BACKWARD propagation

  // Calling base layer function
  Layer::createBackwardPipeline();

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Creating memory descriptor mappings for input memory
    _srcMemDesc = memory::desc({N, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nchw);
    _dstMemDesc = memory::desc({N, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nchw);

    // Creating padding dims
    memory::dims ST = {SV, SH};  // Horizontal Vertical
    memory::dims PTL = {PT, PL}; // Top Left
    memory::dims PBR = {PB, PR}; // Bottom Right

    // Setting strides and padding configuration
    _weightsGradientMem = memory(_weightsMem.get_desc(), _nn->_dnnlEngine);
    _biasGradientMem = memory(_biasMem.get_desc(), _nn->_dnnlEngine);

    auto backwardDataDesc = deconvolution_backward_data::desc(
      algorithm::deconvolution_direct,
      _srcMemDesc,
      _weightsMem.get_desc(),
      _dstMemDesc,
      ST,
      PTL,
      PBR);

    // Create the primitive.
    auto backwardDataPrimitiveDesc = deconvolution_backward_data::primitive_desc(backwardDataDesc, _nn->_dnnlEngine, _forwardDeconvolutionPrimitiveDesc);
    _backwardDataPrimitive = deconvolution_backward_data(backwardDataPrimitiveDesc);

    auto backwardWeightsDesc = deconvolution_backward_weights::desc(
      algorithm::deconvolution_direct,
      _srcMemDesc,
      _weightsMem.get_desc(),
      _biasMem.get_desc(),
      _dstMemDesc,
      ST,
      PTL,
      PBR);

    // Create the primitive.
    auto backwardWeightsPrimitiveDesc = deconvolution_backward_weights::primitive_desc(backwardWeightsDesc, _nn->_dnnlEngine, _forwardDeconvolutionPrimitiveDesc);
    _backwardWeightsPrimitive = deconvolution_backward_weights(backwardWeightsPrimitiveDesc);
  }
#endif
}

void Deconvolution::forwardData(const size_t t)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Arguments to the inner product operation
    std::unordered_map<int, dnnl::memory> forwardDeconvolutionArgs;
    forwardDeconvolutionArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];
    forwardDeconvolutionArgs[DNNL_ARG_WEIGHTS] = _weightsMem;
    forwardDeconvolutionArgs[DNNL_ARG_BIAS] = _biasMem;
    forwardDeconvolutionArgs[DNNL_ARG_DST] = _outputMem[t];

    _forwardDeconvolutionPrimitive.execute(_nn->_dnnlStream, forwardDeconvolutionArgs);
  }
#endif
}

void Deconvolution::backwardData(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    _backwardDataArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];             // Input
    _backwardDataArgs[DNNL_ARG_WEIGHTS] = _weightsMem;                        // Input
    _backwardDataArgs[DNNL_ARG_DIFF_SRC] = _prevLayer->_outputGradientMem[t]; // Output

    _backwardDataPrimitive.execute(_nn->_dnnlStream, _backwardDataArgs);
  }
#endif
}

void Deconvolution::backwardHyperparameters(size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer hyperparameter gradient propagation but NN was configured for inference only.\n");

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
}

void Deconvolution::setHyperparameters(const float *hyperparameters)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    write_to_dnnl_memory(&hyperparameters[0], _weightsMem);
    write_to_dnnl_memory(&hyperparameters[OC * IC * KH * KW], _biasMem);
  }
#endif
}

void Deconvolution::getHyperparameters(float *hyperparameters)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&hyperparameters[0], _weightsMem);
    read_from_dnnl_memory(&hyperparameters[OC * IC * KH * KW], _biasMem);
  }
#endif
}

void Deconvolution::getHyperparameterGradients(float *gradient)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&gradient[0], _weightsGradientMem);
    read_from_dnnl_memory(&gradient[OC * IC * KH * KW], _biasGradientMem);
  }
#endif
}

void Deconvolution::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Image Height"))
 {
 try { _imageHeight = js["Image Height"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Image Height']\n%s", e.what()); } 
   eraseValue(js, "Image Height");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Height'] required by deconvolution.\n"); 

 if (isDefined(js, "Image Width"))
 {
 try { _imageWidth = js["Image Width"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Image Width']\n%s", e.what()); } 
   eraseValue(js, "Image Width");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Width'] required by deconvolution.\n"); 

 if (isDefined(js, "Kernel Height"))
 {
 try { _kernelHeight = js["Kernel Height"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Kernel Height']\n%s", e.what()); } 
   eraseValue(js, "Kernel Height");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Height'] required by deconvolution.\n"); 

 if (isDefined(js, "Kernel Width"))
 {
 try { _kernelWidth = js["Kernel Width"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Kernel Width']\n%s", e.what()); } 
   eraseValue(js, "Kernel Width");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Width'] required by deconvolution.\n"); 

 if (isDefined(js, "Vertical Stride"))
 {
 try { _verticalStride = js["Vertical Stride"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Vertical Stride']\n%s", e.what()); } 
   eraseValue(js, "Vertical Stride");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Vertical Stride'] required by deconvolution.\n"); 

 if (isDefined(js, "Horizontal Stride"))
 {
 try { _horizontalStride = js["Horizontal Stride"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Horizontal Stride']\n%s", e.what()); } 
   eraseValue(js, "Horizontal Stride");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Horizontal Stride'] required by deconvolution.\n"); 

 if (isDefined(js, "Padding Left"))
 {
 try { _paddingLeft = js["Padding Left"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Left']\n%s", e.what()); } 
   eraseValue(js, "Padding Left");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Left'] required by deconvolution.\n"); 

 if (isDefined(js, "Padding Right"))
 {
 try { _paddingRight = js["Padding Right"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Right']\n%s", e.what()); } 
   eraseValue(js, "Padding Right");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Right'] required by deconvolution.\n"); 

 if (isDefined(js, "Padding Top"))
 {
 try { _paddingTop = js["Padding Top"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Top']\n%s", e.what()); } 
   eraseValue(js, "Padding Top");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Top'] required by deconvolution.\n"); 

 if (isDefined(js, "Padding Bottom"))
 {
 try { _paddingBottom = js["Padding Bottom"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Bottom']\n%s", e.what()); } 
   eraseValue(js, "Padding Bottom");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Bottom'] required by deconvolution.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/deconvolution";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: deconvolution: \n%s\n", js.dump(2).c_str());
} 

void Deconvolution::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Image Height"] = _imageHeight;
   js["Image Width"] = _imageWidth;
   js["Kernel Height"] = _kernelHeight;
   js["Kernel Width"] = _kernelWidth;
   js["Vertical Stride"] = _verticalStride;
   js["Horizontal Stride"] = _horizontalStride;
   js["Padding Left"] = _paddingLeft;
   js["Padding Right"] = _paddingRight;
   js["Padding Top"] = _paddingTop;
   js["Padding Bottom"] = _paddingBottom;
 Layer::getConfiguration(js);
} 

void Deconvolution::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Deconvolution::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
