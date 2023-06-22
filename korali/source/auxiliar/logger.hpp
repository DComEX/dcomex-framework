/** \file
* @brief Contains functions to manage file and console output, verbosity levels, and error reporting.
******************************************************************************************************/

#pragma once


#include <string>

/**
 * @brief Terminates execution, printing an error message and indicates file name and line number where the error occurred.
 */
#define KORALI_LOG_ERROR(...) \
  Logger::logError(__FILE__, __LINE__, __VA_ARGS__)

namespace korali
{
/**
* @brief Logger object for Korali Modules
*/
class Logger
{
  public:
  /**
 * @brief Global variable that contains the verbosity level for the current Korali experiment.
 */
  size_t _verbosityLevel;

  /**
 * @brief Global variable that contains the output file for the current Korali experiment.
 */
  FILE *_outputFile;

  /**
 * @brief parametrized constructor for Korali Logger
 * @param verbosityLevel The verbosity level above which nothing is printed.
 * @param file Output file (default: stdout)
 */
  Logger(const std::string verbosityLevel, FILE *file = stdout);

  /**
  * @brief Gets the numerical value of a verbosity level, given its string value.
  * @param verbosityLevel specifies the verbosity level.
  * @return Numerical value corresponding to verbosity level: { SILENT=0, MINIMAL=1, NORMAL=2, DETAILED=3 }
  */
  size_t getVerbosityLevel(const std::string verbosityLevel);

  /**
  * @brief Checks whether the current verbosity level is enough to authorize the requested level. Serves to filter out non-important messages when low verbosity is chosen.
  * @param verbosityLevel the requested verbosity level
  * @return true, if it is enough; false, otherwise.
  */
  bool isEnoughVerbosity(const std::string verbosityLevel);

  /**
  * @brief Outputs raw data to the console file.
  * @param verbosityLevel the requested verbosity level.
  * @param format Format string of the data (printf-style)
  * @param ... List of arguments for the format string
  */
  void logData(const std::string verbosityLevel, const char *format, ...);

  /**
  * @brief Outputs an information message to the console file.
  * @param verbosityLevel the requested verbosity level.
  * @param format Format string of the data (printf-style)
  * @param ... List of arguments for the format string
  */
  void logInfo(const std::string verbosityLevel, const char *format, ...);

  /**
  * @brief Outputs a warning message to the console file.
  * @param verbosityLevel the requested verbosity level.
  * @param format Format string of the data (printf-style)
  * @param ... List of arguments for the format string
  */
  void logWarning(const std::string verbosityLevel, const char *format, ...);

  /**
  * @brief Outputs an error message to the console file. Overrides any verbosity level, prints, and exits execution with error.
  * @param fileName where the error occurred, given by the __FILE__ macro
  * @param lineNumber number where the error occurred, given by the __LINE__ macro
  * @param format Format string of the data (printf-style)
  * @param ... List of arguments for the format string
  */
  static void logError [[noreturn]] (const char *fileName, const int lineNumber, const char *format, ...);
};

} // namespace korali
