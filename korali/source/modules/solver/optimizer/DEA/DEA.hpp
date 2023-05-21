/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: DEA.
*/

/** \dir solver/optimizer/DEA
* @brief Contains code, documentation, and scripts for module: DEA.
*/

#pragma once

#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/optimizer/optimizer.hpp"
#include <vector>

namespace korali
{
namespace solver
{
namespace optimizer
{
;

/**
* @brief Class declaration for module: DEA.
*/
class DEA : public Optimizer
{
  private:
  /**
   * @brief Mutate a sample.
   * @param sampleIdx Index of sample to be mutated.
   */
  void mutateSingle(size_t sampleIdx);

  /**
   * @brief Fix sample params that are outside of domain.
   * @param sampleIdx Index of sample that is outside of domain.
   */
  void fixInfeasible(size_t sampleIdx);

  /**
   * @brief Update the state of Differential Evolution
   * @param samples Sample evaluations.
   */
  void updateSolver(std::vector<Sample> &samples);

  /**
   * @brief Create new set of candidates.
   */
  void initSamples();

  /**
   * @brief Mutate samples and distribute them.
   */
  void prepareGeneration();

  public: 
  /**
  * @brief Specifies the number of samples to evaluate per generation (preferably 5-10x the number of variables).
  */
   size_t _populationSize;
  /**
  * @brief Controls the rate at which dimensions of the samples are mixed (must be in [0,1]).
  */
   double _crossoverRate;
  /**
  * @brief Controls the scaling of the vector differentials (must be in [0,2], preferably < 1).
  */
   double _mutationRate;
  /**
  * @brief Controls the Mutation Rate.
  */
   std::string _mutationRule;
  /**
  * @brief Defines the selection rule of the parent vector.
  */
   std::string _parentSelectionRule;
  /**
  * @brief Sets the accept rule after sample mutation and evaluation.
  */
   std::string _acceptRule;
  /**
  * @brief If set true, Korali samples a random sample between Parent and the voiolated boundary. If set false, infeasible samples are mutated again until feasible.
  */
   int _fixInfeasible;
  /**
  * @brief [Internal Use] Normal random number generator.
  */
   korali::distribution::univariate::Normal* _normalGenerator;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] Objective Function Values.
  */
   std::vector<double> _valueVector;
  /**
  * @brief [Internal Use] Objective Function Values from previous evaluations.
  */
   std::vector<double> _previousValueVector;
  /**
  * @brief [Internal Use] Sample variable information.
  */
   std::vector<std::vector<double>> _samplePopulation;
  /**
  * @brief [Internal Use] Sample candidates variable information.
  */
   std::vector<std::vector<double>> _candidatePopulation;
  /**
  * @brief [Internal Use] Index of the best sample in current generation.
  */
   size_t _bestSampleIndex;
  /**
  * @brief [Internal Use] Best ever model evaluation as of previous generation.
  */
   double _previousBestEverValue;
  /**
  * @brief [Internal Use] Current mean of population.
  */
   std::vector<double> _currentMean;
  /**
  * @brief [Internal Use] Previous mean of population.
  */
   std::vector<double> _previousMean;
  /**
  * @brief [Internal Use] Best variables of current generation.
  */
   std::vector<double> _currentBestVariables;
  /**
  * @brief [Internal Use] Max distance between samples per dimension.
  */
   std::vector<double> _maxDistances;
  /**
  * @brief [Internal Use] Minimum step size of any variable in the current generation.
  */
   double _currentMinimumStepSize;
  /**
  * @brief [Termination Criteria] Specifies the target fitness to stop minimization.
  */
   double _minValue;
  /**
  * @brief [Termination Criteria] Specifies the minimal step size of the population mean from one gneration to another.
  */
   double _minStepSize;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
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
   * @brief Configures Differential Evolution/
   */
  void setInitialConfiguration() override;

  /**
   * @brief Executes sampling & evaluation generation.
   */
  void runGeneration() override;

  /**
   * @brief Console Output before generation runs.
   */
  void printGenerationBefore() override;

  /**
   * @brief Console output after generation.
   */
  void printGenerationAfter() override;

  /**
   * @brief Final console output at termination.
   */
  void finalize() override;
};

} //optimizer
} //solver
} //korali
;
