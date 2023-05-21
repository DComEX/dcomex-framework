/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: Distribution.
*/

/** \dir distribution
* @brief Contains code, documentation, and scripts for module: Distribution.
*/

#pragma once

#include "modules/module.hpp"
#include <gsl/gsl_rng.h>
#include <map>

namespace korali
{
;

/**
* @brief Class declaration for module: Distribution.
*/
class Distribution : public Module
{
  public: 
  /**
  * @brief Defines the name of the distribution.
  */
   std::string _name;
  /**
  * @brief Defines the random seed of the distribution.
  */
   size_t _randomSeed;
  /**
  * @brief Stores the current state of the distribution in hexadecimal notation.
  */
   gsl_rng* _range;
  
 
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
   * @brief Map to store the link between parameter names and their pointers.
   */
  std::map<std::string, double *> _conditionalsMap;

  /**
   * @brief Auxiliar variable to hold pre-calculated data to avoid re-processing information.
   */
  double _aux;

  /**
   * @brief Indicates whether or not this distribution contains conditional variables.
   */
  bool _hasConditionalVariables;

  /**
   * @brief Creates and sets the RNG range (state and seed) of the random distribution
   * @param rangeString The range to load, in string of hexadecimal values form
   * @return Pointer to the new range.
   */
  gsl_rng *setRange(const std::string rangeString);

  /**
   * @brief Gets a hexadecimal string from a given range's state and seed
   * @param range Range to read from
   * @return Hexadecimal string produced.
   */
  std::string getRange(gsl_rng *range) const;

  /**
   * @brief Updates the parameters of the distribution based on conditional variables.
   */
  virtual void updateDistribution(){};

  /**
   * @brief Gets the pointer to a distribution property.
   * @param property The name of the property to update
   * @return Pointer to the property
   */
  virtual double *getPropertyPointer(const std::string &property) { return NULL; };
};

} //korali
;
