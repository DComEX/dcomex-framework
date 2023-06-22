/** \namespace solver
* @brief Namespace declaration for modules of type: solver.
*/

/** \file
* @brief Header file for module: SSM.
*/

/** \dir solver/SSM
* @brief Contains code, documentation, and scripts for module: SSM.
*/

#pragma once

#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/problem/reaction/reaction.hpp"
#include "modules/solver/solver.hpp"

namespace korali
{
namespace solver
{
;

/**
* @brief Class declaration for module: SSM.
*/
class SSM : public Solver
{
  public: 
  /**
  * @brief Total duration of a stochastic reaction simulation.
  */
   double _simulationLength;
  /**
  * @brief Number of bins to calculate the mean trajectory at termination.
  */
   size_t _diagnosticsNumBins;
  /**
  * @brief Number of trajectory simulations per Korali generation (checkpoints are generated between generations).
  */
   size_t _simulationsPerGeneration;
  /**
  * @brief [Internal Use] The current time of the simulated trajectory.
  */
   double _time;
  /**
  * @brief [Internal Use] The number of reactions to simulate.
  */
   size_t _numReactions;
  /**
  * @brief [Internal Use] The current number of each reactant in the simulated trajectory.
  */
   std::vector<int> _numReactants;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] The simulation time associated to each bin.
  */
   std::vector<double> _binTime;
  /**
  * @brief [Internal Use] Stores the number of reactants per bin for each trajectory and reactant.
  */
   std::vector<std::vector<int>> _binCounter;
  /**
  * @brief [Internal Use] Stores the number of reactants per bin for each trajectory and reactant.
  */
   std::vector<std::vector<std::vector<int>>> _binnedTrajectories;
  /**
  * @brief [Internal Use] Counter that keeps track of completed simulations.
  */
   size_t _completedSimulations;
  /**
  * @brief [Termination Criteria] Max number of trajectory simulations.
  */
   size_t _maxNumSimulations;
  
 
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
   * @brief Storage for the pointer to the reaction problem
   */
  problem::Reaction *_problem;

  /**
   * @brief Resets the initial conditions of a new trajectory simulation.
   * @param numReactants initial reactants for new simulation
   * @param time starting time of new simulation
   */
  void reset(std::vector<int> numReactants, double time = 0.);

  /**
   * @brief Simulates a trajectory for all reactants based on provided reactions.
   */
  virtual void advance() = 0;

  /**
   * @brief Updates the values of the binned trajectories for each reactant.
   */
  void updateBins();

  void setInitialConfiguration() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;
  void finalize() override;
};

} //solver
} //korali
;
