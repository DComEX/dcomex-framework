#pragma once


/** \file
* @brief Implements a parser for reaction equations
*        based on the implementation by Luca Amoudruz https://github.com/amlucas/SSM
******************************************************************************/

#include <string>
#include <tuple>
#include <vector>

namespace korali
{
    /**
    * @brief Struct of reaction details constructed from reaction equation.
    */
    struct ParsedReactionString
    {
    
        /**
        * @brief Vector containing all reactants in the reaction.
        */
        std::vector<std::string> reactantNames;

        /**
        * @brief The stoichiometries asssociated to the reactants.
        */
        std::vector<int> reactantSCs;

        /**
        * @brief Vector containing all products.
        */
        std::vector<std::string> productNames;
        
        /**
        * @brief The stoichiometries asssociated to the products.
        */
        std::vector<int> productSCs;

        /**
        * @brief Boolean vector indicating if a reactant is a reservoir and remains unchanged.
        */
        std::vector<bool> isReactantReservoir;
    };

    /**
    * @brief Parses a string and creates a struct of type ParsedReactionString
    * @param s the reaction equation.
    * @return struct containing reaction details.
    */
    ParsedReactionString parseReactionString(std::string s);

} // namespace korali
