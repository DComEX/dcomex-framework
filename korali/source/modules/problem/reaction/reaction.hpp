/** \namespace problem
* @brief Namespace declaration for modules of type: problem.
*/

/** \file
* @brief Header file for module: Reaction.
*/

/** \dir problem/reaction
* @brief Contains code, documentation, and scripts for module: Reaction.
*/

#pragma once

#include "auxiliar/reactionParser.hpp"
#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
;

/**
 * @brief Structure to store reaction information
 */
struct reaction_t
{
  /**
   * @brief The rate of the reaction.
   */
  double rate;

  /**
   * @brief Ids of the reactants.
   */
  std::vector<int> reactantIds;

  /**
   * @brief Stoichiometries of the reactants.
   */
  std::vector<int> reactantStoichiometries;

  /**
   * @brief Ids of the products of the reaction.
   */
  std::vector<int> productIds;

  /**
   * @brief Stoichiometries of the products.
   */
  std::vector<int> productStoichiometries;

  /**
   * @brief Flag to declare reactants as reservois (remain unchanged).
   */
  std::vector<bool> isReactantReservoir = {};

  /**
   * @brief Constructor for type reaction_t.
   * @param rate the rate of the reaction
   * @param reactantIds ids of reactants
   * @param reactantSCs stoichiometry coefficients of reactants
   * @param productIds ids of products
   * @param productSCs stoichiometry coefficients of products
   * @param isReactantReservoir indicators if reactant is reservoir
   */
  reaction_t(double rate,
             std::vector<int> reactantIds,
             std::vector<int> reactantSCs,
             std::vector<int> productIds,
             std::vector<int> productSCs,
             std::vector<bool> isReactantReservoir)
    : rate(rate), reactantIds(std::move(reactantIds)), reactantStoichiometries(std::move(reactantSCs)), productIds(std::move(productIds)), productStoichiometries(std::move(productSCs)), isReactantReservoir(std::move(isReactantReservoir))
  {
    if (this->reactantIds.size() > 0 && this->isReactantReservoir.size() == 0)
    {
      this->isReactantReservoir.resize(this->reactantIds.size());
      this->isReactantReservoir.assign(this->reactantIds.size(), false);
    }
  }
};

/**
* @brief Class declaration for module: Reaction.
*/
class Reaction : public Problem
{
  /**
   * Class for the reaction problem type based on the implementation by Luca Amoudruz https://github.com/amlucas/SSM
   */

  public: 
  /**
  * @brief [Internal Use] Complete description of all reactions.
  */
   knlohmann::json _reactions;
  /**
  * @brief [Internal Use] Maps the reactants name to an internal index.
  */
   std::map<std::string, int> _reactantNameToIndexMap;
  /**
  * @brief [Internal Use] Maps the reactants name to an internal index.
  */
   std::vector<int> _initialReactantNumbers;
  /**
  * @brief [Internal Use] TODO
  */
   std::vector<std::vector<int>> _stateChange;
  
 
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
   * @brief Container for all reactions.
   */
  std::vector<reaction_t> _reactionVector;

  void initialize() override;

  /**
   * @brief Compute the propensity of a reaction.
   * @param reactionIndex The index of the reaction
   * @param reactantNumbers Current number of reactants in simulation
   * @return the propensity of reaction
   */
  double computePropensity(size_t reactionIndex, const std::vector<int> &reactantNumbers) const;

  /**
   * @brief Compute the gradient of the propensity of a reaction wrt reactants.
   * @param reactionIndex The index of the reaction
   * @param reactantNumbers Current number of reactants in simulation
   * @param dI reactantindex for gradient computation
   * @return the gradient of the propensity of reaction
   */
  double computeGradPropensity(size_t reactionIndex, const std::vector<int> &reactantNumbers, size_t dI) const;

  /**
   * @brief Computes F value, the sum of weighted differentials of two reactions.
   * @param reactionIndex The index of the reaction
   * @param otherReactionIndex the index of the second reaction, to access the state change values
   * @param reactantNumbers Current number of reactants in simulation
   * @return value F
   */
  double computeF(size_t reactionIndex, size_t otherReactionIndex, const std::vector<int> &reactantNumbers);

  /**
   * @brief Calculate the maximum allowed firings of a reactant in a reaction.
   * @param reactionIndex The index of the reaction
   * @param reactantNumbers Current number of reactants in simulation
   * @return maximum allowed firings of reaction
   */
  double calculateMaximumAllowedFirings(size_t reactionIndex, const std::vector<int> &reactantNumbers) const;

  /**
   * @brief Initializes the state change matrix.
   * @param numReactants number of reactants in reaction experiment
   */
  void setStateChange(size_t numReactants);

  /**
   * @brief Apply changes to reactants based on reaction.
   * @param reactionIndex The index of the reaction
   * @param reactantNumbers Current number of reactants in simulation
   * @param numFirings Number of repeated firings of reaction
   */
  void applyChanges(size_t reactionIndex, std::vector<int> &reactantNumbers, int numFirings = 1) const;
};

} //problem
} //korali
;
