/** \file
* @brief Auxiliary library for Korali's essential string operations.
**************************************************************************************/

#pragma once


#include <string>

namespace korali
{
/**
* @brief Generates lower case string of provided string
* @param input Input string
* @return The lower case varient of the string
*/
extern std::string toLower(const std::string &input);

/**
* @brief Generates upper case string of provided string
* @param a Input string
* @param b Input string
* @return The upper case variant of the string
*/
extern bool iCompare(const std::string &a, const std::string &b);

} // namespace korali

