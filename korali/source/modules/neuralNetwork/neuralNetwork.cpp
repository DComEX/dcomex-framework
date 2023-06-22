#include "modules/experiment/experiment.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace korali
{
;

NeuralNetwork::NeuralNetwork()
{
  _isInitialized = false;
  _timestepCount = 0;
}

void NeuralNetwork::initialize()
{
  // Initializing corresponding NN Engine

  if (_engine == "OneDNN")
  {
#ifdef _KORALI_USE_ONEDNN
    _dnnlEngine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    _dnnlStream = dnnl::stream(_dnnlEngine);
#else

    fprintf(stderr, "[Korali] Warning: Neural Network's engine set to OneDNN, but Korali was installed without support for OneDNN. Using Korali's default NN Engine\n");
    _engine = "Korali";

#endif
  }

  if (_engine == "CuDNN")
  {
#ifdef _KORALI_USE_CUDNN

    if (cudnnCreate(&_cuDNNHandle) != CUDNN_STATUS_SUCCESS) KORALI_LOG_ERROR("Error initializing CUDNN Handle\n");

#else

    fprintf(stderr, "[Korali] Warning: Neural Network's engine set to OneDNN, but Korali was installed without support for OneDNN. Using Korali's default NN Engine\n");
    _engine = "Korali";
#endif
  }

  if (_isInitialized) KORALI_LOG_ERROR("Neural Network has already been initialized!.\n");

  // Checking correct batch sizes provided
  if (_batchSizes.size() == 0)
    KORALI_LOG_ERROR("No batch sizes specified for the Neural Network.\n");

  for (size_t i = 0; i < _batchSizes.size(); i++)
    if (_batchSizes[i] == 0)
      KORALI_LOG_ERROR("Batch size %lu is zero.\n", i, _batchSizes[i]);

  // Creating layer pipelines of format ThreadCount x Batch Sizes x Layer Count
  size_t layerCount = _layers.size();
#ifdef _OPENMP
  int maxThreads = omp_get_max_threads();
#else
  int maxThreads = 1;
#endif
  _pipelines.resize(maxThreads);
  for (int curThread = 0; curThread < maxThreads; curThread++)
  {
    _pipelines[curThread].resize(_batchSizes.size());
    for (size_t batchSizeIdx = 0; batchSizeIdx < _batchSizes.size(); batchSizeIdx++)
      _pipelines[curThread][batchSizeIdx]._layerVector.resize(layerCount);
  }

  // Creating layer objects
  for (int curThread = 0; curThread < maxThreads; curThread++)
    for (size_t batchSizeIdx = 0; batchSizeIdx < _batchSizes.size(); batchSizeIdx++)
      for (size_t i = 0; i < layerCount; i++)
      {
        auto layerJson = _layers[i];
        layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];
        p->_layerVector[i] = dynamic_cast<neuralNetwork::Layer *>(getModule(layerJson, _k));
        p->_layerVector[i]->applyModuleDefaults(layerJson);
        p->_layerVector[i]->setConfiguration(layerJson);
      }

  // Assigning relevant metadata to all the layers
  for (int curThread = 0; curThread < maxThreads; curThread++)
    for (size_t batchSizeIdx = 0; batchSizeIdx < _batchSizes.size(); batchSizeIdx++)
      for (size_t i = 0; i < layerCount; i++)
      {
        layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];
        p->_layerVector[i]->_prevLayer = i > 0 ? p->_layerVector[i - 1] : nullptr;
        p->_layerVector[i]->_nextLayer = i < layerCount - 1 ? p->_layerVector[i + 1] : nullptr;
        p->_layerVector[i]->_index = i;
        p->_layerVector[i]->_nn = this;
        p->_layerVector[i]->_batchSize = _batchSizes[batchSizeIdx];
        p->_layerVector[i]->_pipeline = p;
      }

  // Initialize layers
  for (int curThread = 0; curThread < maxThreads; curThread++)
    for (size_t batchSizeIdx = 0; batchSizeIdx < _batchSizes.size(); batchSizeIdx++)
      for (size_t i = 0; i < layerCount; i++)
      {
        layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];
        p->_layerVector[i]->initialize();
      }

  // Creating a single set of hyperparameter memory
  for (size_t i = 0; i < layerCount; i++)
    _pipelines[0][0]._layerVector[i]->createHyperparameterMemory();

  // Propagating hyperparamter memory to all other instances
  for (int curThread = 0; curThread < maxThreads; curThread++)
    for (size_t batchSizeIdx = 0; batchSizeIdx < _batchSizes.size(); batchSizeIdx++)
      for (size_t i = 0; i < layerCount; i++)
      {
        layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];
        _pipelines[0][0]._layerVector[i]->copyHyperparameterPointers(p->_layerVector[i]);
      }

  // Getting layer parameter counts and indexes
  _hyperparameterCount = 0;
  for (size_t i = 0; i < layerCount; i++)
  {
    for (int curThread = 0; curThread < maxThreads; curThread++)
      for (size_t batchSizeIdx = 0; batchSizeIdx < _batchSizes.size(); batchSizeIdx++)
      {
        layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];
        p->_layerVector[i]->_hyperparameterIndex = _hyperparameterCount;
      }

    _hyperparameterCount += _pipelines[0][0]._layerVector[i]->_hyperparameterCount;
  }

  // Create forward and backward (only for training) pipelines
  for (int curThread = 0; curThread < maxThreads; curThread++)
    for (size_t batchSizeIdx = 0; batchSizeIdx < _batchSizes.size(); batchSizeIdx++)
    {
      // Getting corresponding layer pipeline pointer
      layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];

      // Getting batch dimensions
      const size_t T = _timestepCount;
      const size_t N = _batchSizes[batchSizeIdx];
      const size_t IC = p->_layerVector[0]->_outputChannels;
      const size_t OC = p->_layerVector[layerCount - 1]->_outputChannels;
      const size_t H = _hyperparameterCount;

      for (size_t i = 0; i < layerCount; i++)
        p->_layerVector[i]->createForwardPipeline();

      if (_mode == "Training")
        for (size_t i = 0; i < layerCount; i++)
          p->_layerVector[i]->createBackwardPipeline();

      // Allocating NN Forward storage
      p->_rawInputValues.resize(T * N * IC);
      p->_rawOutputValues.resize(T * N * OC);
      p->_inputBatchLastStep.resize(N);

      // Allocating NN Backward storage (only for training)
      if (_mode == "Training")
      {
        p->_rawInputGradients.resize(T * N * IC);
        p->_rawOutputGradients.resize(T * N * OC);
        p->_hyperparameterGradients.resize(H);
      }

      // Allocating storage for formatted output values
      p->_outputValues.resize(N);
      for (size_t b = 0; b < N; b++)
        p->_outputValues[b].resize(OC);

      // Allocating storage for formatted input gradients (only for training)
      if (_mode == "Training")
      {
        p->_inputGradients.resize(N);
        for (size_t b = 0; b < N; b++)
          p->_inputGradients[b].resize(IC);
      }
    }

  // Making sure we do not re-initialize
  _isInitialized = true;
}

std::vector<float> NeuralNetwork::generateInitialHyperparameters()
{
  // Empty storage for hyperparameters
  std::vector<float> initialHyperparameters;

  // Initialize hyperparameters layer by layer
  for (size_t i = 0; i < _pipelines[0][0]._layerVector.size(); i++)
  {
    auto layerParameters = _pipelines[0][0]._layerVector[i]->generateInitialHyperparameters();
    initialHyperparameters.insert(initialHyperparameters.end(), layerParameters.begin(), layerParameters.end());
  }

  // Set hyperparameters in neural network
  setHyperparameters(initialHyperparameters);

  return initialHyperparameters;
}

void NeuralNetwork::forward(const std::vector<std::vector<std::vector<float>>> &inputValues)
{
  // Finding out current thread
#ifdef _OPENMP
  size_t curThread = omp_get_thread_num();
#else
  size_t curThread = 0;
#endif

  // Finding out pipeline corresponding to the input batch size id
  size_t N = inputValues.size();
  size_t batchSizeIdx = getBatchSizeIdx(N);

  // Getting corresponding layer pipeline pointer
  layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];

  // Gathering parameters
  size_t T = _timestepCount;
  size_t IC = p->_layerVector[0]->_outputChannels;
  size_t layerCount = p->_layerVector.size();
  size_t lastLayer = layerCount - 1;
  size_t OC = p->_layerVector[lastLayer]->_outputChannels;

  // Safety checks for timestep information
  for (size_t b = 0; b < N; b++)
  {
    if (inputValues[b].size() > T)
      KORALI_LOG_ERROR("Timestep of input batch (%lu) is larger (%lu) than max configured (%lu).\n", b, inputValues[b].size(), T);

    for (size_t t = 0; t < inputValues[b].size(); t++)
      if (inputValues[b][t].size() != IC)
        KORALI_LOG_ERROR("Input size of input batch (%lu), timestep (%lu) is different (%lu) than input channels configured (%lu).\n", b, t, inputValues[b][t].size(), IC);
  }

  // First, re-setting all input values to zero
  std::fill(p->_rawInputValues.begin(), p->_rawInputValues.end(), 0.0f);

  // Storing timestep count per batch input, for later use on backward propagation
  for (size_t b = 0; b < N; b++) p->_inputBatchLastStep[b] = inputValues[b].size() - 1;

// Now replacing values provided by the user in N*T*IC format
#pragma omp parallel for
  for (size_t b = 0; b < N; b++)
    for (size_t t = 0; t < inputValues[b].size(); t++)
      for (size_t i = 0; i < IC; i++)
        p->_rawInputValues[t * N * IC + b * IC + i] = inputValues[b][t][i];

  // Forward propagate layers, once per timestep
  for (size_t t = 0; t < T; t++)
    for (size_t i = 0; i < layerCount; i++)
      p->_layerVector[i]->forwardData(t);

#pragma omp parallel for
  for (size_t b = 0; b < N; b++)
    for (size_t i = 0; i < OC; i++)
      p->_outputValues[b][i] = p->_rawOutputValues[p->_inputBatchLastStep[b] * N * OC + b * OC + i];
}

void NeuralNetwork::backward(const std::vector<std::vector<float>> &outputGradients)
{
  // Finding out current thread
#ifdef _OPENMP
  size_t curThread = omp_get_thread_num();
#else
  size_t curThread = 0;
#endif

  // Finding out pipeline corresponding to the input batch size id
  size_t N = outputGradients.size();
  size_t batchSizeIdx = getBatchSizeIdx(N);

  // Getting corresponding layer pipeline pointer
  layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];

  // Getting batch dimensions
  size_t T = _timestepCount;
  size_t layerCount = p->_layerVector.size();
  size_t lastLayer = layerCount - 1;
  size_t OC = p->_layerVector[lastLayer]->_outputChannels;
  size_t IC = p->_layerVector[0]->_outputChannels;

  // Safety checks
  if (_mode == "Inference")
    KORALI_LOG_ERROR("Requesting backward propagation but NN was configured for inference only.\n");

  for (size_t b = 0; b < N; b++)
    if (outputGradients[b].size() > OC)
      KORALI_LOG_ERROR("Size (%lu) of output gradients batch %lu, is different than expected (%lu).\n", outputGradients[b].size(), b, OC);

  // First, re-setting all output gradients to zero
  std::fill(p->_rawOutputGradients.begin(), p->_rawOutputGradients.end(), 0.0f);
// To store the gradients in the NN we place them on the last input timestep
#pragma omp parallel for
  for (size_t b = 0; b < N; b++)
    for (size_t i = 0; i < OC; i++)
    {
      p->_rawOutputGradients[p->_inputBatchLastStep[b] * N * OC + b * OC + i] = outputGradients[b][i];
    }

  // Resetting cumulative hyperparameter gradients
  std::fill(p->_hyperparameterGradients.begin(), p->_hyperparameterGradients.end(), 0.0f);

  // Backward propagating in time, process the corresponding mini-batch
  for (size_t t = 0; t < T; t++)
  {
    // Storage for the hyperparameter gradients for the current timestep, if needed (T > 1)
    std::vector<float> batchHyperparameterGradients(_hyperparameterCount);

    // Starting from the last timestep, and going backwards
    size_t currentTimestep = T - t - 1;

    // Backward propagating in layer space
    for (size_t i = 0; i < layerCount; i++)
    {
      // Starting from the last layer, and going backwards
      size_t curLayer = lastLayer - i;

      // Running backward data propagation
      p->_layerVector[curLayer]->backwardData(currentTimestep);
    }

    for (size_t i = 0; i < layerCount; i++)
    {
      p->_layerVector[i]->backwardHyperparameters(currentTimestep);

      // If we are passing only one timestep, copy the hyperparameters directly on the NN output
      auto index = p->_layerVector[i]->_hyperparameterIndex;
      if (T == 1) p->_layerVector[i]->getHyperparameterGradients(&p->_hyperparameterGradients[index]);
      if (T > 1) p->_layerVector[i]->getHyperparameterGradients(&batchHyperparameterGradients[index]);
    }

    // Adding current hyperparameters to the cumulative vector, only if more than one timestep used
    if (T > 1)
    {
#pragma omp parallel for simd
      for (size_t i = 0; i < _hyperparameterCount; i++)
        p->_hyperparameterGradients[i] += batchHyperparameterGradients[i];
    }
  }

  // Copying input gradients -- only for the last timestep provided in the input
#pragma omp parallel for
  for (size_t b = 0; b < N; b++)
    for (size_t i = 0; i < IC; i++)
      p->_inputGradients[b][i] = p->_rawInputGradients[p->_inputBatchLastStep[b] * N * IC + b * IC + i];
}

size_t NeuralNetwork::getBatchSizeIdx(const size_t batchSize)
{
  bool foundBatchIdx = false;
  size_t batchSizeIdx = 0;

  for (size_t idx = 0; idx < _batchSizes.size(); idx++)
    if (_batchSizes[idx] == batchSize)
    {
      batchSizeIdx = idx;
      foundBatchIdx = true;
    }

  if (foundBatchIdx == false)
    KORALI_LOG_ERROR("Batch size (%lu) of input is not within the configured batch sizes of the neural network..\n", batchSize);

  return batchSizeIdx;
}

std::vector<std::vector<float>> &NeuralNetwork::getOutputValues(const size_t batchSize)
{
#ifdef _OPENMP
  size_t curThread = omp_get_thread_num();
#else
  size_t curThread = 0;
#endif
  size_t batchSizeIdx = getBatchSizeIdx(batchSize);
  layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];
  return p->_outputValues;
}

std::vector<std::vector<float>> &NeuralNetwork::getInputGradients(const size_t batchSize)
{
#ifdef _OPENMP
  size_t curThread = omp_get_thread_num();
#else
  size_t curThread = 0;
#endif
  size_t batchSizeIdx = getBatchSizeIdx(batchSize);
  layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];
  return p->_inputGradients;
}

std::vector<float> &NeuralNetwork::getHyperparameterGradients(const size_t batchSize)
{
#ifdef _OPENMP
  size_t curThread = omp_get_thread_num();
#else
  size_t curThread = 0;
#endif
  size_t batchSizeIdx = getBatchSizeIdx(batchSize);
  layerPipeline_t *p = &_pipelines[curThread][batchSizeIdx];
  return p->_hyperparameterGradients;
}

std::vector<float> NeuralNetwork::getHyperparameters()
{
  auto hyperparameters = std::vector<float>(_hyperparameterCount);

  size_t layerCount = _pipelines[0][0]._layerVector.size();

  for (size_t i = 0; i < layerCount; i++)
  {
    auto index = _pipelines[0][0]._layerVector[i]->_hyperparameterIndex;
    _pipelines[0][0]._layerVector[i]->getHyperparameters(&hyperparameters[index]);
  }

  return hyperparameters;
}

void NeuralNetwork::setHyperparameters(const std::vector<float> &hyperparameters)
{
  if (hyperparameters.size() != _hyperparameterCount)
    KORALI_LOG_ERROR("Wrong number of hyperparameters passed to the neural network. Expected: %lu, provided: %lu.\n", _hyperparameterCount, hyperparameters.size());

  size_t layerCount = _pipelines[0][0]._layerVector.size();

  for (size_t i = 0; i < layerCount; i++)
  {
    auto index = _pipelines[0][0]._layerVector[i]->_hyperparameterIndex;
    _pipelines[0][0]._layerVector[i]->setHyperparameters(&hyperparameters[index]);
  }
}

void NeuralNetwork::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Current Training Loss"))
 {
 try { _currentTrainingLoss = js["Current Training Loss"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ neuralNetwork ] \n + Key:    ['Current Training Loss']\n%s", e.what()); } 
   eraseValue(js, "Current Training Loss");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Engine"))
 {
 try { _engine = js["Engine"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ neuralNetwork ] \n + Key:    ['Engine']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_engine == "Korali") validOption = true; 
 if (_engine == "OneDNN") validOption = true; 
 if (_engine == "CuDNN") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Engine'] required by neuralNetwork.\n", _engine.c_str()); 
}
   eraseValue(js, "Engine");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Engine'] required by neuralNetwork.\n"); 

 if (isDefined(js, "Mode"))
 {
 try { _mode = js["Mode"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ neuralNetwork ] \n + Key:    ['Mode']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_mode == "Training") validOption = true; 
 if (_mode == "Inference") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mode'] required by neuralNetwork.\n", _mode.c_str()); 
}
   eraseValue(js, "Mode");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mode'] required by neuralNetwork.\n"); 

 if (isDefined(js, "Layers"))
 {
 _layers = js["Layers"].get<knlohmann::json>();

   eraseValue(js, "Layers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Layers'] required by neuralNetwork.\n"); 

 if (isDefined(js, "Timestep Count"))
 {
 try { _timestepCount = js["Timestep Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ neuralNetwork ] \n + Key:    ['Timestep Count']\n%s", e.what()); } 
   eraseValue(js, "Timestep Count");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Timestep Count'] required by neuralNetwork.\n"); 

 if (isDefined(js, "Batch Sizes"))
 {
 try { _batchSizes = js["Batch Sizes"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ neuralNetwork ] \n + Key:    ['Batch Sizes']\n%s", e.what()); } 
   eraseValue(js, "Batch Sizes");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Batch Sizes'] required by neuralNetwork.\n"); 

 Module::setConfiguration(js);
 _type = ".";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: neuralNetwork: \n%s\n", js.dump(2).c_str());
} 

void NeuralNetwork::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Engine"] = _engine;
   js["Mode"] = _mode;
   js["Layers"] = _layers;
   js["Timestep Count"] = _timestepCount;
   js["Batch Sizes"] = _batchSizes;
   js["Current Training Loss"] = _currentTrainingLoss;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
 Module::getConfiguration(js);
} 

void NeuralNetwork::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Engine\": \"Korali\", \"Input Values\": [], \"Batch Sizes\": [], \"Uniform Generator\": {\"Name\": \"Neural Network / Uniform Generator\", \"Type\": \"Univariate/Uniform\", \"Minimum\": -1.0, \"Maximum\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Module::applyModuleDefaults(js);
} 

void NeuralNetwork::applyVariableDefaults() 
{

 Module::applyVariableDefaults();
} 

;

} //korali
;
