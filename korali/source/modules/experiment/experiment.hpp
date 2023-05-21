/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: Experiment.
*/

/** \dir experiment
* @brief Contains code, documentation, and scripts for module: Experiment.
*/

#pragma once

#include "auxiliar/koraliJson.hpp"
#include "auxiliar/libco/libco.h"
#include "config.hpp"
#include "modules/module.hpp"
#include "variable/variable.hpp"
#include <chrono>
#include <functional>
#include <vector>

namespace korali
{
;

/**
* @brief Class declaration for module: Experiment.
*/
class Solver;
/**
* @brief Class declaration for module: Experiment.
*/
class Problem;
/**
* @brief Class declaration for module: Experiment.
*/
class Engine;

/**
* @brief Class declaration for module: Experiment.
*/
class Experiment : public Module
{
  public: 
  /**
  * @brief Specifies the initializing seed for the generation of random numbers. If 0 is specified, Korali will automatically derivate a new seed base on the current time.
  */
   size_t _randomSeed;
  /**
  * @brief Indicates that the engine must preserve the state of their RNGs for reproducibility purposes.
  */
   int _preserveRandomNumberGeneratorStates;
  /**
  * @brief Represents the distributions to use during execution.
  */
   std::vector<korali::distribution::Univariate*> _distributions;
  /**
  * @brief Sample coordinate information.
  */
   std::vector<korali::Variable*> _variables;
  /**
  * @brief Represents the configuration of the problem to solve.
  */
   korali::Problem* _problem;
  /**
  * @brief Represents the state and configuration of the solver algorithm.
  */
   korali::Solver* _solver;
  /**
  * @brief Specifies the path of the results directory.
  */
   std::string _fileOutputPath;
  /**
  * @brief If true, Korali stores a different generation file per generation with incremental numbering. If disabled, Korali stores the latest generation files into a single file, overwriting previous results.
  */
   int _fileOutputUseMultipleFiles;
  /**
  * @brief Specifies whether the partial results should be saved to the results directory.
  */
   int _fileOutputEnabled;
  /**
  * @brief Specifies how often (in generations) will partial result files be saved on the results directory. The default, 1, indicates that every generation's results will be saved. 0 indicates that only the latest is saved.
  */
   size_t _fileOutputFrequency;
  /**
  * @brief Specifies whether the sample information should be saved to samples.json in the results path.
  */
   int _storeSampleInformation;
  /**
  * @brief Specifies how much information will be displayed on console when running Korali.
  */
   std::string _consoleOutputVerbosity;
  /**
  * @brief Specifies how often (in generations) will partial results be printed on console. The default, 1, indicates that every generation's results will be printed.
  */
   size_t _consoleOutputFrequency;
  /**
  * @brief [Internal Use] Indicates the current generation in execution.
  */
   size_t _currentGeneration;
  /**
  * @brief [Internal Use] Indicates whether execution has reached a termination criterion.
  */
   int _isFinished;
  /**
  * @brief [Internal Use] Specifies the Korali run's unique identifier. Used to distinguish run results when two or more use the same output directory.
  */
   size_t _runID;
  /**
  * @brief [Internal Use] Indicates the current time when saving a result file.
  */
   std::string _timestamp;
  
 
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
  

  Experiment();

  void initialize() override;
  void finalize() override;

  /**
   * @brief JSON object to store the experiment's configuration
   */
  KoraliJson _js;

  /**
   * @brief A pointer to the Experiment's logger object.
   */
  Logger *_logger;

  /**
   * @brief A pointer to the parent engine
   */
  Engine *_engine;

  /**
   * @brief JSON object to details of all the samples that have been executed, if requested by the user.
   */
  KoraliJson _sampleInfo;

  /**
   * @brief Experiment Identifier
   */
  size_t _experimentId;

  /**
   * @brief Experiment's coroutine (thread). It is swapped among other experiments, and sample threads.
   */
  cothread_t _thread;

  /**
   * @brief Flag to indicate that the experiment has been initialized to prevent it from re-initializing upon resuming
   */
  bool _isInitialized;

  /**
   * @brief [Profiling] Measures the amount of time taken by saving results
   */
  double _resultSavingTime;

  /**
   * @brief For testing purposes, this field establishes whether the engine is the one to run samples (default = false) or a custom function (true)
   */
  bool _overrideEngine = false;

  /**
   * @brief For testing purposes, this field establishes which custom function to use to override the engine on sample execution for testing.
   */
  std::function<void(Sample &)> _overrideFunction;

  /**
   * @brief Gets an item from the JSON object at the current pointer position.
   * @param key A pybind11 object acting as JSON key (number or string).
   * @return A pybind11 object
   */
  pybind11::object getItem(const pybind11::object key);

  /**
   * @brief Sets an item on the JSON object at the current pointer position.
   * @param key A pybind11 object acting as JSON key (number or string).
   * @param val The value of the item to set.
   */
  void setItem(const pybind11::object key, const pybind11::object val);

  /**
   * @brief Load the state of an experiment from a Korali result file.
   * @param path Path from which to load the experiment state.
   * @return true, if file was found; false, otherwise
   */
  bool loadState(const std::string &path);

  /**
   * @brief Saves the state into the experiment's result path.
   */
  void saveState();

  /**
   * @brief Start the execution of the current experiment.
   */
  void run();

  /**
   * @brief C++ wrapper for the getItem operator.
   * @param key A C++ string acting as JSON key.
   * @return The referenced JSON object content.
   */
  knlohmann::json &operator[](const std::string &key);

  /**
   * @brief Initializes seed to a random value based on current time if not set by the user (i.e. Random Seed is 0).
   * @param js Json object onto which to store the Experiment data.
   */
  void setSeed(knlohmann::json &js);
};

} //korali
;
