/** \file
 * @brief Contains the definition of a Korali Variable
 *****************************************************************************************************/

#pragma once

#include "modules/distribution/univariate/univariate.hpp"
#include <vector>

namespace korali
{
/**
 * \class Variable
 * @brief Contains information for Korali Experiment's variables.
 * This header file is automatically filled with the configuration of Korali's modules, which require that variables contain specific fields.
 */
class Variable
{
  public:
  /**
* @brief [Module: Problem] Defines the name of the variable.
*/
  std::string _name;
/**
* @brief [Module: Bayesian] Indicates the name of the distribution to use as prior distribution.
*/
  std::string _priorDistribution;
/**
* @brief [Module: Bayesian] Stores the the index number of the selected prior distribution.
*/
  size_t _distributionIndex;
/**
* @brief [Module: Design] Indicates what the variable descibes.
*/
  std::string _type;
/**
* @brief [Module: Design] Lower bound for the variable's value.
*/
  double _lowerBound;
/**
* @brief [Module: Design] Upper bound for the variable's value.
*/
  double _upperBound;
/**
* @brief [Module: Design] Indicates the distribution of the variable.
*/
  std::string _distribution;
/**
* @brief [Module: Design] Number of Samples per Direction.
*/
  size_t _numberOfSamples;
/**
* @brief [Module: Propagation] Contains predetermined values for the variables to evaluate.
*/
  std::vector<double> _precomputedValues;
/**
* @brief [Module: Propagation] Contains values sampled from prior.
*/
  std::vector<double> _sampledValues;
/**
* @brief [Module: Reaction] The initial amount of the reactant.
*/
  int _initialReactantNumber;
/**
* @brief [Module: VRACER] Initial standard deviation of the Gaussian distribution from which the given action is sampled.
*/
  float _initialExplorationNoise;
/**
* @brief [Module: Quadrature] Number of Gridpoints along given axis.
*/
  size_t _numberOfGridpoints;
/**
* @brief [Module: Optimizer] [Hint] Initial value at or around which the algorithm shall start looking for an optimum.
*/
  double _initialValue;
/**
* @brief [Module: Optimizer] [Hint] Initial mean for the proposal distribution. This value must be defined between the variable's Mininum and Maximum settings (by default, this value is given by the center of the variable domain).
*/
  double _initialMean;
/**
* @brief [Module: Optimizer] [Hint] Initial standard deviation of the proposal distribution for a variable (by default, this value is given by 30% of the variable domain width).
*/
  double _initialStandardDeviation;
/**
* @brief [Module: Optimizer] [Hint] Lower bound for the standard deviation updates of the proposal distribution for a variable. Korali increases the scaling factor sigma if this value is undershot.
*/
  double _minimumStandardDeviationUpdate;
/**
* @brief [Module: Optimizer] [Hint] Locations to evaluate the Objective Function.
*/
  std::vector<double> _values;
/**
* @brief [Module: CMAES] Specifies the granularity of a discrete variable, a granularity of 1.0 means that the variable can only take values in (.., -1.0, 0.0, +1.0, +2.0, ..) where the levels are set symmetric around the initial mean (here 0.0).
*/
  double _granularity;

};

} // namespace korali
