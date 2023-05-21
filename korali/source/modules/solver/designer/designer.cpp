#include "engine.hpp"
#include "modules/solver/designer/designer.hpp"

namespace korali
{
namespace solver
{
;

void Designer::setInitialConfiguration()
{
  // Getting problem pointer
  _problem = dynamic_cast<problem::Design *>(_k->_problem);

  // Check parameter configuration
  _parameterLowerBounds.resize(_problem->_parameterVectorSize);
  _parameterUpperBounds.resize(_problem->_parameterVectorSize);
  _parameterExtent.resize(_problem->_parameterVectorSize);
  _numberOfParameterSamples.resize(_problem->_parameterVectorSize);
  _parameterGridSpacing.resize(_problem->_parameterVectorSize);
  _parameterDistributionIndex.resize(_problem->_parameterVectorSize);
  _parameterHelperIndices.resize(_problem->_parameterVectorSize);
  for (size_t i = 0; i < _problem->_parameterVectorSize; i++)
  {
    const auto varIdx = _problem->_parameterVectorIndexes[i];

    // Check upper and lower bound
    _parameterLowerBounds[i] = _k->_variables[varIdx]->_lowerBound;
    _parameterUpperBounds[i] = _k->_variables[varIdx]->_upperBound;
    _parameterExtent[i] = _parameterUpperBounds[i] - _parameterLowerBounds[i];

    if (_parameterExtent[i] <= 0.0) KORALI_LOG_ERROR("Upper (%f) and Lower Bound (%f) of parameter variable %lu invalid.\n", _parameterLowerBounds[i], _parameterUpperBounds[i], i);

    // Check number of samples
    _numberOfParameterSamples[i] = _k->_variables[varIdx]->_numberOfSamples;

    if (_numberOfParameterSamples[i] <= 0)
      KORALI_LOG_ERROR("Number Of Samples for variable %lu is not given .\n", varIdx);

    // Set gridspacing
    _parameterGridSpacing[i] = _parameterExtent[i] / (_numberOfParameterSamples[i] - 1);

    // Set helpers to transform linear to cartesian index
    if (i == 0)
      _parameterHelperIndices[i] = 1;
    else
      _parameterHelperIndices[i] = _numberOfParameterSamples[i - 1] * _parameterHelperIndices[i - 1];

    // Check validity of parameter distribution
    const auto priorName = _k->_variables[varIdx]->_distribution;
    bool foundDistribution = false;

    if (priorName == "Grid")
    {
      _parameterDistributionIndex[i] = -1;
      foundDistribution = true;
      if ((isfinite(_parameterLowerBounds[i]) == false) || (isfinite(_parameterUpperBounds[i]) == false))
        KORALI_LOG_ERROR("Provided bounds (%f,%f) for parameter variable %lu is non-finite, but grid evaluation requires bound domain.\n", _parameterLowerBounds[i], _parameterUpperBounds[i], i);
    }

    for (size_t j = 0; (j < _k->_distributions.size()) && (foundDistribution == false); j++)
      if (priorName == _k->_distributions[j]->_name)
      {
        _parameterDistributionIndex[i] = j;
        foundDistribution = true;
      }

    if (foundDistribution == false)
      KORALI_LOG_ERROR("Did not find distribution %s, specified by variable %s\n", _k->_variables[varIdx]->_distribution.c_str(), _k->_variables[i]->_name.c_str());
  }
  // clang-format off
  if (std::any_of(_parameterDistributionIndex.begin(), _parameterDistributionIndex.end(), [](int x) {
        return x == -1;
      }))
  {
    // Check consistency of parameter distribution
    if (std::all_of(_parameterDistributionIndex.begin(), _parameterDistributionIndex.end(), [](int x) {
          return x == -1;
        }) == false)
      KORALI_LOG_ERROR("Parameter distributions are inconsistent. You have to specify a valid distribution or Grid for all of them.\n");

    // Set number of parameter samples
    _numberOfPriorSamples = std::accumulate(_numberOfParameterSamples.begin(), _numberOfParameterSamples.end(), 1, std::multiplies<size_t>());

    // Set integrator
    _parameterIntegrator = "Integrator/Quadrature";

    // Fail here, as this is currently not supported
    KORALI_LOG_ERROR("Currently the quadrature evaluation of the integral is not supported. Choose a prior distribution, not Grid.\n");
  }
  else
  {
    // Check consistency of number of parameter samples
    if (std::all_of(_numberOfParameterSamples.begin(), _numberOfParameterSamples.end(), [&](size_t x) {
          return x == _numberOfParameterSamples[0];
        }) == false)
      KORALI_LOG_ERROR("Parameter distributions are inconsistent. You have to specify the same number of samples for every dimension.\n");

    // Set number of parameter samples
    _numberOfPriorSamples = _numberOfParameterSamples[0];

    // Set integrator
    _parameterIntegrator = "Integrator/MonteCarlo";
  }
  // clang-format on

  // Check design configuration
  _designLowerBounds.resize(_problem->_designVectorSize);
  _designUpperBounds.resize(_problem->_designVectorSize);
  _designExtent.resize(_problem->_designVectorSize);
  _numberOfDesignSamples.resize(_problem->_designVectorSize);
  _designGridSpacing.resize(_problem->_designVectorSize);
  _designHelperIndices.resize(_problem->_designVectorSize);
  for (size_t i = 0; i < _problem->_designVectorSize; i++)
  {
    const auto varIdx = _problem->_designVectorIndexes[i];

    // Check upper and lower bound
    _designLowerBounds[i] = _k->_variables[varIdx]->_lowerBound;
    _designUpperBounds[i] = _k->_variables[varIdx]->_upperBound;
    _designExtent[i] = _designUpperBounds[i] - _designLowerBounds[i];

    if (_designExtent[i] <= 0.0) KORALI_LOG_ERROR("Upper (%f) and Lower Bound (%f) of parameter variable %lu invalid.\n", _designLowerBounds[i], _designUpperBounds[i], i);

    // Check number of samples
    _numberOfDesignSamples[i] = _k->_variables[varIdx]->_numberOfSamples;

    if (_numberOfDesignSamples[i] <= 0)
      KORALI_LOG_ERROR("Number Of Samples for variable %lu is not given .\n", varIdx);

    // Set grid spacing
    _designGridSpacing[i] = _designExtent[i] / (_numberOfDesignSamples[i] - 1);

    // Set helpers to transform linear to cartesian index
    if (i == 0)
      _designHelperIndices[i] = 1;
    else
      _designHelperIndices[i] = _numberOfDesignSamples[i - 1] * _designHelperIndices[i - 1];
  }

  // Compute number of designs
  _numberOfDesigns = std::accumulate(_numberOfDesignSamples.begin(), _numberOfDesignSamples.end(), 1, std::multiplies<size_t>());

  // Compute candidate designs
  for (size_t candidate = 0; candidate < _numberOfDesigns; candidate++)
  {
    std::vector<double> design(_problem->_designVectorSize);
    for (size_t d = 0; d < _problem->_designVectorSize; d++)
    {
      const size_t dimIdx = (size_t)(candidate / _designHelperIndices[d]) % _numberOfDesignSamples[d];
      design[d] = _designLowerBounds[d] + dimIdx * _designGridSpacing[d];
    }
    _designCandidates.push_back(design);
  }

  // Check measurement configuration
  _numberOfMeasurementSamples.resize(_problem->_parameterVectorSize);
  for (size_t i = 0; i < _problem->_measurementVectorSize; i++)
  {
    const auto varIdx = _problem->_measurementVectorIndexes[i];

    // Check number of samples
    _numberOfMeasurementSamples[i] = _k->_variables[varIdx]->_numberOfSamples;

    if (_numberOfMeasurementSamples[i] <= 0)
      KORALI_LOG_ERROR("Number Of Samples for variable %lu is not given .\n", varIdx);
  }

  // clang-format off
  // Check consistency of number of likelihood samples
  if (std::all_of(_numberOfMeasurementSamples.begin(), _numberOfMeasurementSamples.end(), [&](size_t x) {
        return x == _numberOfMeasurementSamples[0];
      }) == false)
    KORALI_LOG_ERROR("Number of measurement samples are inconsistent. You have to specify the same number of samples for every dimension.\n");
  // clang-format on
  // Set number of likelihood samples
  _numberOfLikelihoodSamples = _numberOfMeasurementSamples[0];

  // Resize measurment vector
  _modelEvaluations.resize(_numberOfDesigns);

  // Resize utility vector
  _utility.resize(_numberOfDesigns, 0.0);

  // Set termination criterium
  _maxModelEvaluations = _numberOfPriorSamples + _numberOfDesigns;
}

void Designer::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  /*** Step 1: Sample Prior and Evaluate Model ***/
  _k->_logger->logInfo("Minimal", "Sampling the prior and evaluating...\n");

  // Compute how many samples still have to be evaluated
  size_t numEvaluations = _numberOfPriorSamples;
  if (_executionsPerGeneration > 0)
    numEvaluations = std::min(_executionsPerGeneration, _numberOfPriorSamples - _modelEvaluations[0].size());

  // Create and start samples
  _samples.resize(numEvaluations);
  for (size_t i = 0; i < numEvaluations; i++)
  {
    std::vector<double> params(_problem->_parameterVectorSize);

    // Set parameter values
    if (_parameterIntegrator == "Integrator/Quadrature")
    {
      for (size_t d = 0; d < _problem->_parameterVectorSize; d++)
      {
        const size_t dimIdx = (size_t)(_priorSamples.size() / _parameterHelperIndices[d]) % _numberOfParameterSamples[d];
        params[d] = _parameterLowerBounds[d] + dimIdx * _parameterGridSpacing[d];
      }
    }
    else
    { // Sample parameters
      for (size_t d = 0; d < _problem->_parameterVectorSize; d++)
        params[d] = _k->_distributions[_parameterDistributionIndex[d]]->getRandomNumber();
    }

    // Save prior samples
    _priorSamples.push_back(params);

    // Configure Sample
    _samples[i]["Sample Id"] = i;
    _samples[i]["Module"] = "Problem";
    _samples[i]["Operation"] = "Run Model";
    _samples[i]["Parameters"] = params;
    _samples[i]["Designs"] = _designCandidates;

    // Start Sample
    KORALI_START(_samples[i]);

    // Increase counter
    _modelEvaluationCount++;
  }

  KORALI_WAITALL(_samples);

  for (size_t i = 0; i < numEvaluations; i++)
  {
    const auto evaluation = KORALI_GET(std::vector<std::vector<double>>, _samples[i], "Model Evaluation");

    // Check whether one value per design was returned
    if (evaluation.size() != _numberOfDesigns)
      KORALI_LOG_ERROR("Evaluation returned vector returned with the wrong size: %lu, expected: %lu.\n", evaluation.size(), _numberOfDesigns);

    // Check whether the dimension of the measurement is correct
    for (size_t e = 0; e < _numberOfDesigns; e++)
    {
      if (evaluation[e].size() != _problem->_measurementVectorSize)
        KORALI_LOG_ERROR("Evaluation %ld returned vector returned with the wrong size: %lu, expected: %lu.\n", e, evaluation[e].size(), _problem->_measurementVectorSize);

      // Save evaluation
      _modelEvaluations[e].push_back(evaluation[e]);
    }
  }

  if (_modelEvaluations[0].size() < _numberOfPriorSamples)
    return;

  /*** Step 2: Compute Utility  ***/
  _k->_logger->logInfo("Minimal", "Computing the utility...\n");

  _samples.clear();
  _samples.resize(_numberOfDesigns);
  for (size_t e = 0; e < _numberOfDesigns; e++)
  {
    // Configure Sample
    _samples[e]["Sample Id"] = e;
    _samples[e]["Module"] = "Solver";
    _samples[e]["Operation"] = "Evaluate Design";
    _samples[e]["Evaluations"] = _modelEvaluations[e];

    // Start Sample
    KORALI_START(_samples[e]);

    // Increase counter
    _modelEvaluationCount++;
  }

  KORALI_WAITALL(_samples);

  // Get utility and determine maximum
  double maxUtility = -std::numeric_limits<double>::infinity();
  for (size_t e = 0; e < _numberOfDesigns; e++)
  {
    // Gather result
    _utility[e] = KORALI_GET(double, _samples[e], "Utility");

    // Update maximum
    if (_utility[e] > maxUtility)
    {
      maxUtility = _utility[e];
      _optimalDesignIndex = e;
    }
  }
  (*_k)["Results"]["Utility"] = _utility;
}

void Designer::evaluateDesign(Sample &sample)
{
  const auto evaluations = KORALI_GET(std::vector<std::vector<double>>, sample, "Evaluations");

  double utility = 0.0;
  const double negInvTwoSigmaSq = -1 / (2 * _sigma * _sigma);
#pragma omp parallel for reduction(+ \
                                   : utility)
  for (size_t i = 0; i < _numberOfPriorSamples; i++)
  {
    const auto &mean = evaluations[i];
    for (size_t j = 0; j < _numberOfLikelihoodSamples; j++)
    {
      // Sample likelihood
      std::vector<double> eps(_problem->_measurementVectorSize);
      for (size_t d = 0; d < _problem->_measurementVectorSize; d++)
        eps[d] = _sigma * _normalGenerator->getRandomNumber();

      // Compute first part of utility (note 1/sqrt(2 pi sigma^2) cancels with second term)
      double MSE = 0.0;
      for (size_t d = 0; d < _problem->_measurementVectorSize; d++)
        MSE += eps[d] * eps[d];
      MSE *= negInvTwoSigmaSq;

      /* Compute second part using log-sum-trick (https://en.wikipedia.org/wiki/LogSumExp) */

      // First compute exponents
      std::vector<double> exponents(_numberOfPriorSamples);
      for (size_t k = 0; k < _numberOfPriorSamples; k++)
      {
        const auto &meanInner = evaluations[k];
        double MSEinner = 0.0;
        for (size_t d = 0; d < _problem->_measurementVectorSize; d++)
          MSEinner += (mean[d] + eps[d] - meanInner[d]) * (mean[d] + eps[d] - meanInner[d]);
        MSEinner *= negInvTwoSigmaSq;
        exponents[k] = MSEinner;
      }

      // Finalize computation of second part
      const double marginal = logSumExp(exponents);

      // Sum Utility
      utility += (MSE - marginal);
    }
  }
  utility /= (float)(_numberOfPriorSamples * _numberOfLikelihoodSamples);

  sample["Utility"] = utility;
}

void Designer::printGenerationBefore()
{
}

void Designer::printGenerationAfter()
{
}

void Designer::finalize()
{
  _k->_logger->logInfo("Minimal", "Optimal Design (indx=%ld): [", _optimalDesignIndex);
  for (size_t d = 0; d < _problem->_designVectorSize; d++)
  {
    const size_t dimIdx = (size_t)(_optimalDesignIndex / _designHelperIndices[d]) % _numberOfDesignSamples[d];
    const double optimalDesign = _designLowerBounds[d] + dimIdx * _designGridSpacing[d];
    (*_k)["Results"]["Optimal Design"] = optimalDesign;
    _k->_logger->logData("Minimal", " %f ", optimalDesign);
  }
  _k->_logger->logData("Minimal", "]\n");

  _k->_logger->logInfo("Detailed", "Utility: [");
  for (size_t i = 0; i < _numberOfDesigns - 1; i++)
  {
    _k->_logger->logData("Detailed", " %f, ", _utility[i]);
  }
  _k->_logger->logData("Detailed", " %f ]\n", _utility[_numberOfDesigns - 1]);
}

void Designer::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Prior Samples"))
 {
 try { _priorSamples = js["Prior Samples"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Prior Samples']\n%s", e.what()); } 
   eraseValue(js, "Prior Samples");
 }

 if (isDefined(js, "Model Evaluations"))
 {
 try { _modelEvaluations = js["Model Evaluations"].get<std::vector<std::vector<std::vector<double>>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Model Evaluations']\n%s", e.what()); } 
   eraseValue(js, "Model Evaluations");
 }

 if (isDefined(js, "Normal Generator"))
 {
 _normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(js["Normal Generator"], _k));
 _normalGenerator->applyVariableDefaults();
 _normalGenerator->applyModuleDefaults(js["Normal Generator"]);
 _normalGenerator->setConfiguration(js["Normal Generator"]);
   eraseValue(js, "Normal Generator");
 }

 if (isDefined(js, "Parameter Lower Bounds"))
 {
 try { _parameterLowerBounds = js["Parameter Lower Bounds"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Parameter Lower Bounds']\n%s", e.what()); } 
   eraseValue(js, "Parameter Lower Bounds");
 }

 if (isDefined(js, "Parameter Upper Bounds"))
 {
 try { _parameterUpperBounds = js["Parameter Upper Bounds"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Parameter Upper Bounds']\n%s", e.what()); } 
   eraseValue(js, "Parameter Upper Bounds");
 }

 if (isDefined(js, "Parameter Extent"))
 {
 try { _parameterExtent = js["Parameter Extent"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Parameter Extent']\n%s", e.what()); } 
   eraseValue(js, "Parameter Extent");
 }

 if (isDefined(js, "Number Of Parameter Samples"))
 {
 try { _numberOfParameterSamples = js["Number Of Parameter Samples"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Number Of Parameter Samples']\n%s", e.what()); } 
   eraseValue(js, "Number Of Parameter Samples");
 }

 if (isDefined(js, "Parameter Distribution Index"))
 {
 try { _parameterDistributionIndex = js["Parameter Distribution Index"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Parameter Distribution Index']\n%s", e.what()); } 
   eraseValue(js, "Parameter Distribution Index");
 }

 if (isDefined(js, "Parameter Grid Spacing"))
 {
 try { _parameterGridSpacing = js["Parameter Grid Spacing"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Parameter Grid Spacing']\n%s", e.what()); } 
   eraseValue(js, "Parameter Grid Spacing");
 }

 if (isDefined(js, "Parameter Helper Indices"))
 {
 try { _parameterHelperIndices = js["Parameter Helper Indices"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Parameter Helper Indices']\n%s", e.what()); } 
   eraseValue(js, "Parameter Helper Indices");
 }

 if (isDefined(js, "Parameter Integrator"))
 {
 try { _parameterIntegrator = js["Parameter Integrator"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Parameter Integrator']\n%s", e.what()); } 
   eraseValue(js, "Parameter Integrator");
 }

 if (isDefined(js, "Design Lower Bounds"))
 {
 try { _designLowerBounds = js["Design Lower Bounds"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Design Lower Bounds']\n%s", e.what()); } 
   eraseValue(js, "Design Lower Bounds");
 }

 if (isDefined(js, "Design Upper Bounds"))
 {
 try { _designUpperBounds = js["Design Upper Bounds"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Design Upper Bounds']\n%s", e.what()); } 
   eraseValue(js, "Design Upper Bounds");
 }

 if (isDefined(js, "Design Extent"))
 {
 try { _designExtent = js["Design Extent"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Design Extent']\n%s", e.what()); } 
   eraseValue(js, "Design Extent");
 }

 if (isDefined(js, "Number Of Design Samples"))
 {
 try { _numberOfDesignSamples = js["Number Of Design Samples"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Number Of Design Samples']\n%s", e.what()); } 
   eraseValue(js, "Number Of Design Samples");
 }

 if (isDefined(js, "Design Grid Spacing"))
 {
 try { _designGridSpacing = js["Design Grid Spacing"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Design Grid Spacing']\n%s", e.what()); } 
   eraseValue(js, "Design Grid Spacing");
 }

 if (isDefined(js, "Design Helper Indices"))
 {
 try { _designHelperIndices = js["Design Helper Indices"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Design Helper Indices']\n%s", e.what()); } 
   eraseValue(js, "Design Helper Indices");
 }

 if (isDefined(js, "Design Candidates"))
 {
 try { _designCandidates = js["Design Candidates"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Design Candidates']\n%s", e.what()); } 
   eraseValue(js, "Design Candidates");
 }

 if (isDefined(js, "Number Of Measurement Samples"))
 {
 try { _numberOfMeasurementSamples = js["Number Of Measurement Samples"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Number Of Measurement Samples']\n%s", e.what()); } 
   eraseValue(js, "Number Of Measurement Samples");
 }

 if (isDefined(js, "Number Of Prior Samples"))
 {
 try { _numberOfPriorSamples = js["Number Of Prior Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Number Of Prior Samples']\n%s", e.what()); } 
   eraseValue(js, "Number Of Prior Samples");
 }

 if (isDefined(js, "Number Of Likelihood Samples"))
 {
 try { _numberOfLikelihoodSamples = js["Number Of Likelihood Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Number Of Likelihood Samples']\n%s", e.what()); } 
   eraseValue(js, "Number Of Likelihood Samples");
 }

 if (isDefined(js, "Number Of Designs"))
 {
 try { _numberOfDesigns = js["Number Of Designs"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Number Of Designs']\n%s", e.what()); } 
   eraseValue(js, "Number Of Designs");
 }

 if (isDefined(js, "Optimal Design Index"))
 {
 try { _optimalDesignIndex = js["Optimal Design Index"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Optimal Design Index']\n%s", e.what()); } 
   eraseValue(js, "Optimal Design Index");
 }

 if (isDefined(js, "Utility"))
 {
 try { _utility = js["Utility"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Utility']\n%s", e.what()); } 
   eraseValue(js, "Utility");
 }

 if (isDefined(js, "Executions Per Generation"))
 {
 try { _executionsPerGeneration = js["Executions Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Executions Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Executions Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Executions Per Generation'] required by designer.\n"); 

 if (isDefined(js, "Sigma"))
 {
 try { _sigma = js["Sigma"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ designer ] \n + Key:    ['Sigma']\n%s", e.what()); } 
   eraseValue(js, "Sigma");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Sigma'] required by designer.\n"); 

 Solver::setConfiguration(js);
 _type = "designer";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: designer: \n%s\n", js.dump(2).c_str());
} 

void Designer::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Executions Per Generation"] = _executionsPerGeneration;
   js["Sigma"] = _sigma;
   js["Prior Samples"] = _priorSamples;
   js["Model Evaluations"] = _modelEvaluations;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
   js["Parameter Lower Bounds"] = _parameterLowerBounds;
   js["Parameter Upper Bounds"] = _parameterUpperBounds;
   js["Parameter Extent"] = _parameterExtent;
   js["Number Of Parameter Samples"] = _numberOfParameterSamples;
   js["Parameter Distribution Index"] = _parameterDistributionIndex;
   js["Parameter Grid Spacing"] = _parameterGridSpacing;
   js["Parameter Helper Indices"] = _parameterHelperIndices;
   js["Parameter Integrator"] = _parameterIntegrator;
   js["Design Lower Bounds"] = _designLowerBounds;
   js["Design Upper Bounds"] = _designUpperBounds;
   js["Design Extent"] = _designExtent;
   js["Number Of Design Samples"] = _numberOfDesignSamples;
   js["Design Grid Spacing"] = _designGridSpacing;
   js["Design Helper Indices"] = _designHelperIndices;
   js["Design Candidates"] = _designCandidates;
   js["Number Of Measurement Samples"] = _numberOfMeasurementSamples;
   js["Number Of Prior Samples"] = _numberOfPriorSamples;
   js["Number Of Likelihood Samples"] = _numberOfLikelihoodSamples;
   js["Number Of Designs"] = _numberOfDesigns;
   js["Optimal Design Index"] = _optimalDesignIndex;
   js["Utility"] = _utility;
 Solver::getConfiguration(js);
} 

void Designer::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Executions Per Generation\": 0, \"Sigma\": 0, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Designer::applyVariableDefaults() 
{

 Solver::applyVariableDefaults();
} 

bool Designer::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Evaluate Design")
 {
  evaluateDesign(sample);
  return true;
 }

 operationDetected = operationDetected || Solver::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Designer.\n", operation.c_str());
 return operationDetected;
}

;

} //solver
} //korali
;
