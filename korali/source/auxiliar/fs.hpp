/** \file
* @brief Contains auxiliar code for file system (files and folders) manipulation.
******************************************************************************/

#pragma once


#include <string>
#include <vector>

namespace korali
{
/**
 * @brief Creates a new folder and builds the entire path, if necessary.
 * @param dirPath relative path to the new folder.
 */
void mkdir(const std::string dirPath);

/**
 * @brief Lists all files within within a given folder path
 * @param dirPath relative path to the folder to list.
 * @return A list with the path of all files found.
 */

bool dirExists(const std::string dirPath);

} // namespace korali

