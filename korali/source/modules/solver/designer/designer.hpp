/** \namespace solver
* @brief Namespace declaration for modules of type: solver.
*/

/** \file
* @brief Header file for module: Designer.
*/

/** \dir solver/designer
* @brief Contains code, documentation, and scripts for module: Designer.
*/

#pragma once

#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/problem/design/design.hpp"
#include "modules/solver/solver.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
;

/**
* @brief Class declaration for module: Designer.
*/
class Designer : public Solver
{
  public: 
  /**
  * @brief Specifies the number of model executions per generation. By default this setting is 0, meaning that all executions will be performed in the first generation. For values greater 0, executions will be split into batches and split int generations for intermediate output.
  */
   size_t _executionsPerGeneration;
  /**
  * @brief Standard deviation for measurement.
  */
   double _sigma;
  /**
  * @brief [Internal Use] The samples of the prior distribution.
  */
   std::vector<std::vector<double>> _priorSamples;
  /**
  * @brief [Internal Use] Evaluations of the samples of the prior distribution.
  */
   std::vector<std::vector<std::vector<double>>> _modelEvaluations;
  /**
  * @brief [Internal Use] Gaussian random number generator.
  */
   korali::distribution::univariate::Normal* _normalGenerator;
  /**
  * @brief [Internal Use] The lower bound of the parameters.
  */
   std::vector<double> _parameterLowerBounds;
  /**
  * @brief [Internal Use] The upper bound of the parameters.
  */
   std::vector<double> _parameterUpperBounds;
  /**
  * @brief [Internal Use] The extent of the domain of the parameters (for grid-based evaluation).
  */
   std::vector<double> _parameterExtent;
  /**
  * @brief [Internal Use] The number of samples per direction.
  */
   std::vector<size_t> _numberOfParameterSamples;
  /**
  * @brief [Internal Use] The distribution of parameters (for monte-carlo evaluation).
  */
   std::vector<int> _parameterDistributionIndex;
  /**
  * @brief [Internal Use] The grid spacing of the parameters (for grid-based evaluation).
  */
   std::vector<double> _parameterGridSpacing;
  /**
  * @brief [Internal Use] Holds helper to calculate cartesian indices from linear index (for grid-based evaluation).
  */
   std::vector<size_t> _parameterHelperIndices;
  /**
  * @brief [Internal Use] The integrator that is used for the parameter-integral.
  */
   std::string _parameterIntegrator;
  /**
  * @brief [Internal Use] The lower bound of the designs.
  */
   std::vector<double> _designLowerBounds;
  /**
  * @brief [Internal Use] The upper bound of the designs.
  */
   std::vector<double> _designUpperBounds;
  /**
  * @brief [Internal Use] The extent of the design space.
  */
   std::vector<double> _designExtent;
  /**
  * @brief [Internal Use] The number of samples per direction.
  */
   std::vector<size_t> _numberOfDesignSamples;
  /**
  * @brief [Internal Use] The grid spacing of the designs (for grid-based evaluation).
  */
   std::vector<double> _designGridSpacing;
  /**
  * @brief [Internal Use] Holds helper to calculate cartesian indices from linear index (for grid-based evaluation).
  */
   std::vector<size_t> _designHelperIndices;
  /**
  * @brief [Internal Use] Holds candidate designs.
  */
   std::vector<std::vector<double>> _designCandidates;
  /**
  * @brief [Internal Use] The number of samples per direction.
  */
   std::vector<size_t> _numberOfMeasurementSamples;
  /**
  * @brief [Internal Use] Specifies the number of samples drawn from the prior distribution.
  */
   size_t _numberOfPriorSamples;
  /**
  * @brief [Internal Use] Specifies the number of samples drawn from the likelihood.
  */
   size_t _numberOfLikelihoodSamples;
  /**
  * @brief [Internal Use] Specifies the number of design parameters (for grid-based evaluation).
  */
   size_t _numberOfDesigns;
  /**
  * @brief [Internal Use] Index of the optimal design.
  */
   size_t _optimalDesignIndex;
  /**
  * @brief [Internal Use] Evaluation of utility.
  */
   std::vector<double> _utility;
  
 
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  /**
  * @brief Runs the operation specified on the given sample. It checks recursively whether the function was found by the current module or its parents.
  * @param sample Sample to operate on. Should contain in the 'Operation' field an operation accepted by this module or its parents.
  * @param operation Should specify an operation type accepted by this module or its parents.
  * @return True, if operation found and executed; false, otherwise.
  */
  bool runOperation(std::string operation, korali::Sample& sample) override;
  

  /**
   * @brief Container for samples to be evaluated per generation
   */
  std::vector<Sample> _samples;

  /**
   * @brief Problem pointer
   */
  problem::Design *_problem;

  /**
   * @brief Evaluates the utility function for a given design
   * @param sample A Korali Sample
   */
  void evaluateDesign(Sample &sample);

  virtual void setInitialConfiguration() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;
  void finalize() override;
};

} //solver
} //korali
;
