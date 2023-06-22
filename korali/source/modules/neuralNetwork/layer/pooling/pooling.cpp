#include "modules/neuralNetwork/layer/pooling/pooling.hpp"
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

void Pooling::initialize()
{
  // Checking Layer size
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("Pooling layers cannot be the starting layer of the NN\n");
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("Pooling layers cannot be the last layer of the NN\n");

  // Precalculating values for the pooling operation
  N = _batchSize;
  IH = _imageHeight;
  IW = _imageWidth;
  KH = _kernelHeight;
  KW = _kernelWidth;

  SV = _verticalStride;
  SH = _horizontalStride;
  PT = _paddingTop;
  PL = _paddingLeft;
  PB = _paddingBottom;
  PR = _paddingRight;

  // Check for non zeros
  if (IH <= 0) KORALI_LOG_ERROR("Image height must be larger than zero for pooling layer.\n");
  if (IW <= 0) KORALI_LOG_ERROR("Image width must be larger than zero for pooling layer.\n");
  if (KH <= 0) KORALI_LOG_ERROR("Kernel height must be larger than zero for pooling layer.\n");
  if (KW <= 0) KORALI_LOG_ERROR("Kernel width must be larger than zero for pooling layer.\n");
  if (SV <= 0) KORALI_LOG_ERROR("Vertical stride must be larger than zero for pooling layer.\n");
  if (SH <= 0) KORALI_LOG_ERROR("Horizontal stride must be larger than zero for pooling layer.\n");

  // Several sanity checks
  if (KH > IH) KORALI_LOG_ERROR("Kernel height cannot be larger than input image height.\n");
  if (KW > IW) KORALI_LOG_ERROR("Kernel height cannot be larger than input image height.\n");
  if (PR + PL > IW) KORALI_LOG_ERROR("L+R Paddings cannot exceed the width of the input image.\n");
  if (PT + PB > IH) KORALI_LOG_ERROR("T+B Paddings cannot exceed the height of the input image.\n");

  // Check whether the output channels of the previous layer is divided by the height and width
  if (_prevLayer->_outputChannels % (IH * IW) > 0) KORALI_LOG_ERROR("Previous layer contains a number of channels (%lu) not divisible by the pooling 2D HxW setup (%lux%lu).\n", _prevLayer->_outputChannels, IH, IW);
  IC = _prevLayer->_outputChannels / (IH * IW);

  // Deriving output height and width
  OH = std::floor((IH - (KH - (PR + PL))) / SH) + 1;
  OW = std::floor((IW - (KW - (PT + PB))) / SV) + 1;

  // Check whether the output channels of the previous layer is divided by the height and width
  if (_outputChannels % (OH * OW) > 0) KORALI_LOG_ERROR("Pooling layer contains a number of output channels (%lu) not divisible by the output image size (%lux%lu) given kernel (%lux%lu) size and padding/stride configuration.\n", _outputChannels, OH, OW, KH, KW);
  OC = _outputChannels / (OH * OW);
}

void Pooling::createForwardPipeline()
{
  // Calling base layer function
  Layer::createForwardPipeline();

  if (_nn->_engine == "Korali") KORALI_LOG_ERROR("Pooling Layers still not supported in Korali's NN backend. Use OneDNN.\n");
  if (_nn->_engine == "CuDNN") KORALI_LOG_ERROR("Pooling Layers still not supported in CuDNNbackend. Use OneDNN.\n");

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

    // Creating work memory
    memory::dims kernelDims = {KH, KW};

    // Determining algorithm
    dnnl::algorithm algorithmType;
    if (_function == "Max") algorithmType = dnnl::algorithm::pooling_max;
    if (_function == "Inclusive Average") algorithmType = dnnl::algorithm::pooling_avg_include_padding;
    if (_function == "Exclusive Average") algorithmType = dnnl::algorithm::pooling_avg_exclude_padding;

    // We create the pooling operation
    auto pooling_d = pooling_forward::desc(_propKind, algorithmType, _srcMemDesc, _dstMemDesc, ST, kernelDims, PTL, PBR);

    // Create inner product primitive descriptor.
    dnnl::primitive_attr poolingPrimitiveAttributes;
    _forwardPoolingPrimitiveDesc = pooling_forward::primitive_desc(pooling_d, poolingPrimitiveAttributes, _nn->_dnnlEngine);

    // Create pooling workspace memory
    _workspaceMem.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++)
      _workspaceMem[t] = memory(_forwardPoolingPrimitiveDesc.workspace_desc(), _nn->_dnnlEngine);

    // Create the weights+bias primitive.
    _forwardPoolingPrimitive = pooling_forward(_forwardPoolingPrimitiveDesc);
  }
#endif
}

void Pooling::createBackwardPipeline()
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

    // Creating work memory
    memory::dims kernelDims = {KH, KW};

    // Determining algorithm
    dnnl::algorithm algorithmType;
    if (_function == "Max") algorithmType = dnnl::algorithm::pooling_max;
    if (_function == "Inclusive Average") algorithmType = dnnl::algorithm::pooling_avg_include_padding;
    if (_function == "Exclusive Average") algorithmType = dnnl::algorithm::pooling_avg_exclude_padding;

    auto backwardDataDesc = pooling_backward::desc(
      algorithmType,
      _srcMemDesc,
      _dstMemDesc,
      ST,
      kernelDims,
      PTL,
      PBR);

    // Create the primitive.
    auto backwardDataPrimitiveDesc = pooling_backward::primitive_desc(backwardDataDesc, _nn->_dnnlEngine, _forwardPoolingPrimitiveDesc);
    _backwardDataPrimitive = pooling_backward(backwardDataPrimitiveDesc);
  }
#endif
}

void Pooling::forwardData(const size_t t)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Arguments to the inner product operation
    std::unordered_map<int, dnnl::memory> forwardPoolingArgs;
    forwardPoolingArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];
    forwardPoolingArgs[DNNL_ARG_DST] = _outputMem[t];
    forwardPoolingArgs[DNNL_ARG_WORKSPACE] = _workspaceMem[t];
    _forwardPoolingPrimitive.execute(_nn->_dnnlStream, forwardPoolingArgs);
  }
#endif
}

void Pooling::backwardData(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    _backwardDataArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];             // Input
    _backwardDataArgs[DNNL_ARG_DIFF_SRC] = _prevLayer->_outputGradientMem[t]; // Output
    _backwardDataArgs[DNNL_ARG_WORKSPACE] = _workspaceMem[t];
    _backwardDataPrimitive.execute(_nn->_dnnlStream, _backwardDataArgs);
  }
#endif
}

void Pooling::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Function"))
 {
 try { _function = js["Function"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Function']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_function == "Max") validOption = true; 
 if (_function == "Inclusive Average") validOption = true; 
 if (_function == "Exclusive Average") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Function'] required by pooling.\n", _function.c_str()); 
}
   eraseValue(js, "Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Function'] required by pooling.\n"); 

 if (isDefined(js, "Image Height"))
 {
 try { _imageHeight = js["Image Height"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Image Height']\n%s", e.what()); } 
   eraseValue(js, "Image Height");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Height'] required by pooling.\n"); 

 if (isDefined(js, "Image Width"))
 {
 try { _imageWidth = js["Image Width"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Image Width']\n%s", e.what()); } 
   eraseValue(js, "Image Width");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Width'] required by pooling.\n"); 

 if (isDefined(js, "Kernel Height"))
 {
 try { _kernelHeight = js["Kernel Height"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Kernel Height']\n%s", e.what()); } 
   eraseValue(js, "Kernel Height");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Height'] required by pooling.\n"); 

 if (isDefined(js, "Kernel Width"))
 {
 try { _kernelWidth = js["Kernel Width"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Kernel Width']\n%s", e.what()); } 
   eraseValue(js, "Kernel Width");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Width'] required by pooling.\n"); 

 if (isDefined(js, "Vertical Stride"))
 {
 try { _verticalStride = js["Vertical Stride"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Vertical Stride']\n%s", e.what()); } 
   eraseValue(js, "Vertical Stride");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Vertical Stride'] required by pooling.\n"); 

 if (isDefined(js, "Horizontal Stride"))
 {
 try { _horizontalStride = js["Horizontal Stride"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Horizontal Stride']\n%s", e.what()); } 
   eraseValue(js, "Horizontal Stride");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Horizontal Stride'] required by pooling.\n"); 

 if (isDefined(js, "Padding Left"))
 {
 try { _paddingLeft = js["Padding Left"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Left']\n%s", e.what()); } 
   eraseValue(js, "Padding Left");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Left'] required by pooling.\n"); 

 if (isDefined(js, "Padding Right"))
 {
 try { _paddingRight = js["Padding Right"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Right']\n%s", e.what()); } 
   eraseValue(js, "Padding Right");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Right'] required by pooling.\n"); 

 if (isDefined(js, "Padding Top"))
 {
 try { _paddingTop = js["Padding Top"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Top']\n%s", e.what()); } 
   eraseValue(js, "Padding Top");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Top'] required by pooling.\n"); 

 if (isDefined(js, "Padding Bottom"))
 {
 try { _paddingBottom = js["Padding Bottom"].get<ssize_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Bottom']\n%s", e.what()); } 
   eraseValue(js, "Padding Bottom");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Bottom'] required by pooling.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/pooling";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: pooling: \n%s\n", js.dump(2).c_str());
} 

void Pooling::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Function"] = _function;
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

void Pooling::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Pooling::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
