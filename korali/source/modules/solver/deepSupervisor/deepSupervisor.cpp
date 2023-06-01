#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/solver/deepSupervisor/deepSupervisor.hpp"
#include "sample/sample.hpp"
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace korali
{
namespace solver
{
;

void DeepSupervisor::initialize()
{
  // Getting problem pointer
  _problem = dynamic_cast<problem::SupervisedLearning *>(_k->_problem);

  // Fixing termination criteria for testing mode
  if (_mode == "Testing") _maxGenerations = _k->_currentGeneration + 1;

  // Don't reinitialize neural network if experiment was already initialized [if running several minibatches]
  if (_k->_isInitialized == true)
    return;

  // Check whether the minibatch size (N) can be divided by the requested concurrency
  if (_problem->_trainingBatchSize % _batchConcurrency > 0) KORALI_LOG_ERROR("The training concurrency requested (%lu) does not divide the training mini batch size (%lu) perfectly.", _batchConcurrency, _problem->_trainingBatchSize);

  // Check whether the minibatch size (N) can be divided by the requested concurrency
  if (_problem->_testingBatchSize % _batchConcurrency > 0) KORALI_LOG_ERROR("The Testing concurrency requested (%lu) does not divide the training mini batch size (%lu) perfectly.", _batchConcurrency, _problem->_testingBatchSize);

  // Determining batch sizes
  std::vector<size_t> batchSizes = {_problem->_trainingBatchSize, _problem->_testingBatchSize};

  // If parallelizing training, we need to support the split batch size
  if (_batchConcurrency > 1) batchSizes.push_back(_problem->_trainingBatchSize / _batchConcurrency);
  if (_batchConcurrency > 1) batchSizes.push_back(_problem->_testingBatchSize / _batchConcurrency);

  // Configuring neural network
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = _neuralNetworkEngine;
  neuralNetworkConfig["Timestep Count"] = _problem->_maxTimesteps;

  // Iterator for the current layer id
  size_t curLayer = 0;

  // Setting the number of input layer nodes as number of input vector size
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][curLayer]["Output Channels"] = _problem->_inputSize;
  curLayer++;

  // Adding user-defined hidden layers
  for (size_t i = 0; i < _neuralNetworkHiddenLayers.size(); i++)
  {
    neuralNetworkConfig["Layers"][curLayer]["Weight Scaling"] = _outputWeightsScaling;
    neuralNetworkConfig["Layers"][curLayer] = _neuralNetworkHiddenLayers[i];
    curLayer++;
  }

  // Adding linear transformation layer to convert hidden state to match output channels
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Linear";
  neuralNetworkConfig["Layers"][curLayer]["Output Channels"] = _problem->_solutionSize;
  neuralNetworkConfig["Layers"][curLayer]["Weight Scaling"] = _outputWeightsScaling;
  curLayer++;

  // Applying a user-defined pre-activation function
  if (_neuralNetworkOutputActivation != "Identity")
  {
    neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Activation";
    neuralNetworkConfig["Layers"][curLayer]["Function"] = _neuralNetworkOutputActivation;
    curLayer++;
  }

  // Applying output layer configuration
  neuralNetworkConfig["Layers"][curLayer] = _neuralNetworkOutputLayer;
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Output";

  // Instancing training neural network
  auto trainingNeuralNetworkConfig = neuralNetworkConfig;
  trainingNeuralNetworkConfig["Batch Sizes"] = batchSizes;
  trainingNeuralNetworkConfig["Mode"] = "Training";
  _neuralNetwork = dynamic_cast<NeuralNetwork *>(getModule(trainingNeuralNetworkConfig, _k));
  _neuralNetwork->applyModuleDefaults(trainingNeuralNetworkConfig);
  _neuralNetwork->setConfiguration(trainingNeuralNetworkConfig);
  _neuralNetwork->initialize();

  // Configure optimizer
  if (_k->_currentGeneration == 0)
  {
    // Configure Optimizer
    knlohmann::json optimizerConfig;
    optimizerConfig["Type"] = "deepSupervisor/optimizers/f" + _neuralNetworkOptimizer;
    optimizerConfig["N Vars"] = _neuralNetwork->_hyperparameterCount;

    _optimizer = dynamic_cast<korali::fGradientBasedOptimizer *>(korali::Module::getModule(optimizerConfig, _k));
    _optimizer->applyModuleDefaults(optimizerConfig);
    _optimizer->setConfiguration(optimizerConfig);
    _optimizer->initialize();
    _optimizer->_currentValue = _neuralNetwork->generateInitialHyperparameters();
  }
  else
  {
    // Load saved hyperparameters
    _neuralNetwork->setHyperparameters(_optimizer->_currentValue);
  }

  // Setting current loss
  _currentLoss = 0.0f;
}

void DeepSupervisor::runGeneration()
{
  if (_mode == "Training") runTrainingGeneration();
  if (_mode == "Testing") runTestingGeneration();
}

void DeepSupervisor::runTestingGeneration()
{
  // Check whether training concurrency exceeds the number of workers
  if (_batchConcurrency > _k->_engine->_conduit->getWorkerCount()) KORALI_LOG_ERROR("The batch concurrency requested (%lu) exceeds the number of Korali workers defined in the conduit type/configuration (%lu).", _batchConcurrency, _k->_engine->_conduit->getWorkerCount());

  // Checking that incoming data has a correct format
  if (_problem->_testingBatchSize != _problem->_inputData.size())
    KORALI_LOG_ERROR("Testing Batch size %lu different than that of input data (%lu).\n", _problem->_inputData.size(), _problem->_testingBatchSize);

  // In case we run Mean Squared Error with concurrency, distribute the work among samples
  if (_batchConcurrency > 1)
  {
    // Calculating per worker dimensions
    const size_t NW = _problem->_testingBatchSize / _batchConcurrency;
    const size_t T = _problem->_inputData[0].size();
    const size_t IC = _problem->_inputData[0][0].size();

    // Getting current NN hyperparameters
    const auto nnHyperparameters = _neuralNetwork->getHyperparameters();

    // Sending input to workers for parallel processing
    std::vector<Sample> samples(_batchConcurrency);
#pragma omp parallel for
    for (size_t sId = 0; sId < _batchConcurrency; sId++)
    {
      // Carving part of the batch data that corresponds to this sample
      auto workerInputDataFlat = std::vector<float>(NW * IC * T);
      for (size_t i = 0; i < NW; i++)
        for (size_t j = 0; j < T; j++)
          for (size_t k = 0; k < IC; k++)
            workerInputDataFlat[i * T * IC + j * IC + k] = _problem->_inputData[sId * NW + i][j][k];

      // Setting up sample
      samples[sId]["Sample Id"] = sId;
      samples[sId]["Module"] = "Solver";
      samples[sId]["Operation"] = "Run Evaluation On Worker";
      samples[sId]["Input Data"] = workerInputDataFlat;
      samples[sId]["Input Dims"] = std::vector<size_t>({NW, T, IC});
      samples[sId]["Hyperparameters"] = nnHyperparameters;
    }

    // Launching samples
    for (size_t i = 0; i < _batchConcurrency; i++) KORALI_START(samples[i]);

    // Waiting for samples to finish
    KORALI_WAITALL(samples);

    // Assembling hyperparameters and the total mean squared loss
    _evaluation.clear();
    for (size_t i = 0; i < _batchConcurrency; i++)
    {
      const auto workerEvaluations = KORALI_GET(std::vector<std::vector<float>>, samples[i], "Evaluation");
      _evaluation.insert(_evaluation.end(), workerEvaluations.begin(), workerEvaluations.end());
    }
  }

  // If we use an MSE loss function, we need to update the gradient vector with its difference with each of batch's last timestep of the NN output
  if (_batchConcurrency == 1)
  {
    // Getting a reference to the neural network output
    _evaluation = getEvaluation(_problem->_inputData);
  }
}

void DeepSupervisor::runTrainingGeneration()
{
  // Grabbing batch size
  const size_t N = _problem->_trainingBatchSize;

  // Check whether training concurrency exceeds the number of workers
  if (_batchConcurrency > _k->_engine->_conduit->getWorkerCount()) KORALI_LOG_ERROR("The batch concurrency requested (%lu) exceeds the number of Korali workers defined in the conduit type/configuration (%lu).", _batchConcurrency, _k->_engine->_conduit->getWorkerCount());

  // Updating solver's learning rate, if changed
  _optimizer->_eta = _learningRate;

  // Checking that incoming data has a correct format
  _problem->verifyData();

  // Hyperparameter gradient storage
  std::vector<float> nnHyperparameterGradients;

  // In case we run Mean Squared Error with concurrency, distribute the work among samples
  if (_lossFunction == "Mean Squared Error" && _batchConcurrency > 1)
  {
    // Calculating per worker dimensions
    const size_t NW = _problem->_trainingBatchSize / _batchConcurrency;
    const size_t T = _problem->_inputData[0].size();
    const size_t IC = _problem->_inputData[0][0].size();
    const size_t OC = _problem->_solutionData[0].size();

    // Getting current NN hyperparameters
    const auto nnHyperparameters = _neuralNetwork->getHyperparameters();

    // Sending input to workers for parallel processing
    std::vector<Sample> samples(_batchConcurrency);
#pragma omp parallel for
    for (size_t sId = 0; sId < _batchConcurrency; sId++)
    {
      // Carving part of the batch data that corresponds to this sample
      auto workerInputDataFlat = std::vector<float>(NW * IC * T);
      for (size_t i = 0; i < NW; i++)
        for (size_t j = 0; j < T; j++)
          for (size_t k = 0; k < IC; k++)
            workerInputDataFlat[i * T * IC + j * IC + k] = _problem->_inputData[sId * NW + i][j][k];

      auto workerSolutionDataFlat = std::vector<float>(NW * IC * T);
      for (size_t i = 0; i < NW; i++)
        for (size_t j = 0; j < OC; j++)
          workerSolutionDataFlat[i * OC + j] = _problem->_solutionData[sId * NW + i][j];

      // Setting up sample
      samples[sId]["Sample Id"] = sId;
      samples[sId]["Module"] = "Solver";
      samples[sId]["Operation"] = "Run Training On Worker";
      samples[sId]["Input Data"] = workerInputDataFlat;
      samples[sId]["Input Dims"] = std::vector<size_t>({NW, T, IC});
      samples[sId]["Solution Data"] = workerSolutionDataFlat;
      samples[sId]["Solution Dims"] = std::vector<size_t>({NW, OC});
      samples[sId]["Hyperparameters"] = nnHyperparameters;
    }

    // Launching samples
    for (size_t i = 0; i < _batchConcurrency; i++) KORALI_START(samples[i]);

    // Waiting for samples to finish
    KORALI_WAITALL(samples);

    // Assembling hyperparameters and the total mean squared loss
    _currentLoss = 0.0f;
    nnHyperparameterGradients = std::vector<float>(_neuralNetwork->_hyperparameterCount, 0.0f);
    for (size_t i = 0; i < _batchConcurrency; i++)
    {
      _currentLoss += KORALI_GET(float, samples[i], "Squared Loss");
      const auto workerGradients = KORALI_GET(std::vector<float>, samples[i], "Hyperparameter Gradients");
      for (size_t i = 0; i < workerGradients.size(); i++) nnHyperparameterGradients[i] += workerGradients[i];
    }
    _currentLoss = _currentLoss / ((float)N * 2.0f);
  }

  // If we use an MSE loss function, we need to update the gradient vector with its difference with each of batch's last timestep of the NN output
  if (_lossFunction == "Mean Squared Error" && _batchConcurrency == 1)
  {
    // Grabbing constants
    const size_t OC = _problem->_solutionSize;

    // Making a copy of the solution data for MSE calculation
    auto MSEVector = _problem->_solutionData;

    // Getting a reference to the neural network output
    const auto &results = getEvaluation(_problem->_inputData);

// Calculating gradients via the loss function
#pragma omp parallel for simd
    for (size_t b = 0; b < N; b++)
      for (size_t i = 0; i < OC; i++)
        MSEVector[b][i] = MSEVector[b][i] - results[b][i];

#pragma omp parallel for reduction(+ \
                                   : _currentLoss) collapse(2)
    for (size_t b = 0; b < N; b++)
      for (size_t i = 0; i < OC; i++)
        _currentLoss += MSEVector[b][i] * MSEVector[b][i];
    _currentLoss = _currentLoss / ((float)N * 2.0f);

    // Running back propagation on the MSE vector
    nnHyperparameterGradients = backwardGradients(MSEVector);
  }

  // Append Loss to Loss History
  _lossHistory.push_back(_currentLoss);

  // If the solution represents the gradients, just pass them on
  if (_lossFunction == "Direct Gradient") nnHyperparameterGradients = backwardGradients(_problem->_solutionData);

  // Passing hyperparameter gradients through a gradient descent update
  _optimizer->processResult(nnHyperparameterGradients);

  // Getting new set of hyperparameters from the gradient descent algorithm
  _neuralNetwork->setHyperparameters(_optimizer->_currentValue);
}

std::vector<float> DeepSupervisor::getHyperparameters()
{
  return _neuralNetwork->getHyperparameters();
}

std::vector<std::vector<float>> &DeepSupervisor::getEvaluation(const std::vector<std::vector<std::vector<float>>> &input)
{
  // Grabbing constants
  const size_t N = input.size();

  // Running the input values through the neural network
  _neuralNetwork->forward(input);

  // Returning the output values for the last given timestep
  return _neuralNetwork->getOutputValues(N);
}

std::vector<float> DeepSupervisor::backwardGradients(const std::vector<std::vector<float>> &gradients)
{
  // Grabbing constants
  const size_t N = gradients.size();

  // Running the input values through the neural network
  _neuralNetwork->backward(gradients);

  // Getting NN hyperparameter gradients
  auto hyperparameterGradients = _neuralNetwork->getHyperparameterGradients(N);

  // If required, apply L2 Normalization to the network's hyperparameters
  if (_l2RegularizationEnabled)
  {
    const auto nnHyperparameters = _neuralNetwork->getHyperparameters();
#pragma omp parallel for simd
    for (size_t i = 0; i < hyperparameterGradients.size(); i++)
      hyperparameterGradients[i] -= _l2RegularizationImportance * nnHyperparameters[i];
  }

  // Returning the hyperparameter gradients
  return hyperparameterGradients;
}

void DeepSupervisor::runTrainingOnWorker(korali::Sample &sample)
{
  // Updating hyperparameters in the worker's NN
  const auto nnHyperparameters = KORALI_GET(std::vector<float>, sample, "Hyperparameters");
  _neuralNetwork->setHyperparameters(nnHyperparameters);
  sample._js.getJson().erase("Hyperparameters");

  // Getting input from sample
  const auto inputDataFlat = KORALI_GET(std::vector<float>, sample, "Input Data");
  sample._js.getJson().erase("Input Data");

  // Getting solution from sample
  const auto solutionDataFlat = KORALI_GET(std::vector<float>, sample, "Solution Data");
  sample._js.getJson().erase("Solution Data");

  // Getting input dimensions
  const auto inputDims = KORALI_GET(std::vector<size_t>, sample, "Input Dims");
  const size_t N = inputDims[0];
  const size_t T = inputDims[1];
  const size_t IC = inputDims[2];
  sample._js.getJson().erase("Input Dims");

  // De-flattening input
  auto input = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(T, std::vector<float>(IC)));
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < T; j++)
      for (size_t k = 0; k < IC; k++)
        input[i][j][k] = inputDataFlat[i * T * IC + j * IC + k];

  // Getting solution dimensions
  const auto solutionDims = KORALI_GET(std::vector<size_t>, sample, "Solution Dims");
  const size_t OC = solutionDims[1];
  sample._js.getJson().erase("Solution Dims");

  // De-flattening solution
  auto solution = std::vector<std::vector<float>>(N, std::vector<float>(OC));
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < OC; j++)
      solution[i][j] = solutionDataFlat[i * OC + j];

  // Getting a reference to the neural network output
  const auto &results = getEvaluation(input);

  // Running Mean squared error function
  float squaredLoss = 0.0f;

// Calculating gradients via the loss function
#pragma omp parallel for simd
  for (size_t b = 0; b < N; b++)
    for (size_t i = 0; i < OC; i++)
      solution[b][i] = solution[b][i] - results[b][i];

  // Adding square losses
  for (size_t b = 0; b < N; b++)
    for (size_t i = 0; i < OC; i++)
      squaredLoss += solution[b][i] * solution[b][i];

  // Running the input values through the neural network
  backwardGradients(solution);

  // Storing the output values for the last given timestep
  sample["Hyperparameter Gradients"] = _neuralNetwork->getHyperparameterGradients(N);
  sample["Squared Loss"] = squaredLoss;
}

void DeepSupervisor::runEvaluationOnWorker(korali::Sample &sample)
{
  // Updating hyperparameters in the worker's NN
  auto nnHyperparameters = KORALI_GET(std::vector<float>, sample, "Hyperparameters");
  _neuralNetwork->setHyperparameters(nnHyperparameters);
  sample._js.getJson().erase("Hyperparameters");

  // Getting input from sample
  auto inputDataFlat = KORALI_GET(std::vector<float>, sample, "Input Data");
  sample._js.getJson().erase("Input Data");

  // Getting input dimensions
  auto inputDims = KORALI_GET(std::vector<size_t>, sample, "Input Dims");
  size_t N = inputDims[0];
  size_t T = inputDims[1];
  size_t IC = inputDims[2];
  sample._js.getJson().erase("Input Dims");

  // De-flattening input
  auto input = std::vector<std::vector<std::vector<float>>>(N, std::vector<std::vector<float>>(T, std::vector<float>(IC)));
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < T; j++)
      for (size_t k = 0; k < IC; k++)
        input[i][j][k] = inputDataFlat[i * T * IC + j * IC + k];

  // Storing the output values for the last given timestep
  sample["Evaluation"] = getEvaluation(input);
}

void DeepSupervisor::printGenerationAfter()
{
  // Printing results so far
  if (_lossFunction == "Mean Squared Error") _k->_logger->logInfo("Normal", " + Training Loss: %.15f\n", _currentLoss);
  if (_lossFunction == "Direct Gradient") _k->_logger->logInfo("Normal", " + Gradient L2-Norm: %.15f\n", std::sqrt(_currentLoss));
}

void DeepSupervisor::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Evaluation"))
 {
 try { _evaluation = js["Evaluation"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Evaluation']\n%s", e.what()); } 
   eraseValue(js, "Evaluation");
 }

 if (isDefined(js, "Current Loss"))
 {
 try { _currentLoss = js["Current Loss"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Current Loss']\n%s", e.what()); } 
   eraseValue(js, "Current Loss");
 }

 if (isDefined(js, "Loss History"))
 {
 try { _lossHistory = js["Loss History"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Loss History']\n%s", e.what()); } 
   eraseValue(js, "Loss History");
 }

 if (isDefined(js, "Normalization Means"))
 {
 try { _normalizationMeans = js["Normalization Means"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Normalization Means']\n%s", e.what()); } 
   eraseValue(js, "Normalization Means");
 }

 if (isDefined(js, "Normalization Variances"))
 {
 try { _normalizationVariances = js["Normalization Variances"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Normalization Variances']\n%s", e.what()); } 
   eraseValue(js, "Normalization Variances");
 }

 if (isDefined(js, "Optimizer"))
 {
 _optimizer = dynamic_cast<korali::fGradientBasedOptimizer*>(korali::Module::getModule(js["Optimizer"], _k));
 _optimizer->applyVariableDefaults();
 _optimizer->applyModuleDefaults(js["Optimizer"]);
 _optimizer->setConfiguration(js["Optimizer"]);
   eraseValue(js, "Optimizer");
 }

 if (isDefined(js, "Mode"))
 {
 try { _mode = js["Mode"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Mode']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_mode == "Training") validOption = true; 
 if (_mode == "Testing") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mode'] required by deepSupervisor.\n", _mode.c_str()); 
}
   eraseValue(js, "Mode");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mode'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Hidden Layers"))
 {
 _neuralNetworkHiddenLayers = js["Neural Network"]["Hidden Layers"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Hidden Layers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Hidden Layers'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Output Activation"))
 {
 _neuralNetworkOutputActivation = js["Neural Network"]["Output Activation"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Output Activation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Output Activation'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Output Layer"))
 {
 _neuralNetworkOutputLayer = js["Neural Network"]["Output Layer"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Output Layer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Output Layer'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Engine"))
 {
 try { _neuralNetworkEngine = js["Neural Network"]["Engine"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Neural Network']['Engine']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Engine");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Engine'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Optimizer"))
 {
 try { _neuralNetworkOptimizer = js["Neural Network"]["Optimizer"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Neural Network']['Optimizer']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_neuralNetworkOptimizer == "Adam") validOption = true; 
 if (_neuralNetworkOptimizer == "AdaBelief") validOption = true; 
 if (_neuralNetworkOptimizer == "MadGrad") validOption = true; 
 if (_neuralNetworkOptimizer == "AdaGrad") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Neural Network']['Optimizer'] required by deepSupervisor.\n", _neuralNetworkOptimizer.c_str()); 
}
   eraseValue(js, "Neural Network", "Optimizer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Optimizer'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Loss Function"))
 {
 try { _lossFunction = js["Loss Function"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Loss Function']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_lossFunction == "Direct Gradient") validOption = true; 
 if (_lossFunction == "Mean Squared Error") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Loss Function'] required by deepSupervisor.\n", _lossFunction.c_str()); 
}
   eraseValue(js, "Loss Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Loss Function'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Learning Rate"))
 {
 try { _learningRate = js["Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate'] required by deepSupervisor.\n"); 

 if (isDefined(js, "L2 Regularization", "Enabled"))
 {
 try { _l2RegularizationEnabled = js["L2 Regularization"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['L2 Regularization']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Enabled'] required by deepSupervisor.\n"); 

 if (isDefined(js, "L2 Regularization", "Importance"))
 {
 try { _l2RegularizationImportance = js["L2 Regularization"]["Importance"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['L2 Regularization']['Importance']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Importance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Importance'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Output Weights Scaling"))
 {
 try { _outputWeightsScaling = js["Output Weights Scaling"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Output Weights Scaling']\n%s", e.what()); } 
   eraseValue(js, "Output Weights Scaling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Output Weights Scaling'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Batch Concurrency"))
 {
 try { _batchConcurrency = js["Batch Concurrency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Batch Concurrency']\n%s", e.what()); } 
   eraseValue(js, "Batch Concurrency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Batch Concurrency'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Termination Criteria", "Target Loss"))
 {
 try { _targetLoss = js["Termination Criteria"]["Target Loss"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Termination Criteria']['Target Loss']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Target Loss");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Target Loss'] required by deepSupervisor.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Solver::setConfiguration(js);
 _type = "deepSupervisor";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: deepSupervisor: \n%s\n", js.dump(2).c_str());
} 

void DeepSupervisor::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Mode"] = _mode;
   js["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
   js["Neural Network"]["Output Activation"] = _neuralNetworkOutputActivation;
   js["Neural Network"]["Output Layer"] = _neuralNetworkOutputLayer;
   js["Neural Network"]["Engine"] = _neuralNetworkEngine;
   js["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
   js["Loss Function"] = _lossFunction;
   js["Learning Rate"] = _learningRate;
   js["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
   js["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
   js["Output Weights Scaling"] = _outputWeightsScaling;
   js["Batch Concurrency"] = _batchConcurrency;
   js["Termination Criteria"]["Target Loss"] = _targetLoss;
   js["Evaluation"] = _evaluation;
   js["Current Loss"] = _currentLoss;
   js["Loss History"] = _lossHistory;
   js["Normalization Means"] = _normalizationMeans;
   js["Normalization Variances"] = _normalizationVariances;
 if(_optimizer != NULL) _optimizer->getConfiguration(js["Optimizer"]);
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void DeepSupervisor::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"L2 Regularization\": {\"Enabled\": false, \"Importance\": 0.0001}, \"Neural Network\": {\"Output Activation\": \"Identity\", \"Output Layer\": {}}, \"Termination Criteria\": {\"Target Loss\": -1.0}, \"Hyperparameters\": [], \"Output Weights Scaling\": 1.0, \"Batch Concurrency\": 1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void DeepSupervisor::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool DeepSupervisor::checkTermination()
{
 bool hasFinished = false;

 if ((_k->_currentGeneration > 1) && (_targetLoss > 0.0) && (_currentLoss <= _targetLoss))
 {
  _terminationCriteria.push_back("deepSupervisor['Target Loss'] = " + std::to_string(_targetLoss) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

bool DeepSupervisor::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Run Training On Worker")
 {
  runTrainingOnWorker(sample);
  return true;
 }

 if (operation == "Run Evaluation On Worker")
 {
  runEvaluationOnWorker(sample);
  return true;
 }

 operationDetected = operationDetected || Solver::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem DeepSupervisor.\n", operation.c_str());
 return operationDetected;
}

;

} //solver
} //korali
;
