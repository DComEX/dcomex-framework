/** \file
* @brief Header file for the base Korali Module class.
*********************************************************************************************/

#pragma once


#include "auxiliar/koraliJson.hpp"
#include "auxiliar/kstring.hpp"
#include "auxiliar/logger.hpp"
#include "auxiliar/math.hpp"
#include <chrono>

/*! \namespace Korali
    \brief The Korali namespace includes all Korali-specific functions, variables, and modules.
*/
namespace korali
{
class Experiment;
class Conduit;
class Engine;

/*! \class Module
    \brief Represents the basic building block of all Korali modules.
*/
class Module
{
  public:
  virtual ~Module() = default;

  /**
  * @brief Stores the name of the module type selected. Determines which C++ class is constructed upon initialization.
  */
  std::string _type;

  /**
  * @brief Stores a pointer to its containing experiment.
  */
  korali::Experiment *_k;

  /**
  * @brief Instantiates the requested module class given its type and returns its pointer.
  * @param js JSON file containing the module's configuration and type.
  * @param e Korali Experiment to serve as parent to the module.
  * @return Pointer with the newly created module.
  */
  static Module *getModule(knlohmann::json &js, korali::Experiment *e);

  /**
   * @brief Initializes Module upon creation. May allocate memory, set initial states, and initialize external code.
   */
  virtual void initialize();

  /**
   * @brief Sets pointer to the current Korali engine, if the module requires running samples
   * @param engine Engine pointer
   */
  virtual void setEngine(korali::Engine* engine);

  /**
   * @brief Finalizes Module. Deallocates memory and produces outputs.
   */
  virtual void finalize();

  /**
   * @brief Returns the module type.
   * @return A string containing the exact type with which it was created.
   */
  virtual std::string getType();

  /**
   * @brief Determines whether the module can trigger termination of an experiment run.
   * @return True, if it should trigger termination; false, otherwise.
   */
  virtual bool checkTermination();

  /**
   * @brief Obtains the entire current state and configuration of the module.
   * @param js JSON object onto which to save the serialized state of the module.
   */
  virtual void getConfiguration(knlohmann::json &js);

  /**
   * @brief Sets the entire state and configuration of the module, given a JSON object.
   * @param js JSON object from which to deserialize the state of the module.
   */
  virtual void setConfiguration(knlohmann::json &js);

  /**
   * @brief Applies the module's default configuration upon its creation.
   * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  virtual void applyModuleDefaults(knlohmann::json &js);

  /**
   * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  virtual void applyVariableDefaults();

  /**
  * @brief Runs the operation specified in the operation field. It checks recursively whether the function was found by the current module or its parents
  * @param sample Sample to operate on
  * @param operation An operation accepted by this module or its parents
  * @return True, if operation found and executed; false, otherwise.
  */
  virtual bool runOperation(std::string operation, korali::Sample &sample);
};

/**
 * @brief Storage for profiling information.
*/
extern knlohmann::json __profiler;

/**
 * @brief Start time for the current Korali run.
*/
extern std::chrono::time_point<std::chrono::high_resolution_clock> _startTime;

/**
 * @brief End time for the current Korali run.
*/
extern std::chrono::time_point<std::chrono::high_resolution_clock> _endTime;

/**
 * @brief Cumulative time for all Korali runs during the current application execution.
*/
extern double _cumulativeTime;

} // namespace korali

