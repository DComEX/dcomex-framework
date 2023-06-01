#include "logger.hpp"
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

namespace korali
{

Logger::Logger(const std::string verbosityLevel, FILE *file)
{
  _verbosityLevel = getVerbosityLevel(verbosityLevel);
  _outputFile = file;
}

size_t Logger::getVerbosityLevel(const std::string verbosityLevel)
{
  if (verbosityLevel == "Silent") return 0;
  if (verbosityLevel == "Minimal") return 1;
  if (verbosityLevel == "Normal") return 2;
  if (verbosityLevel == "Detailed") return 3;
  return 0;
}

bool Logger::isEnoughVerbosity(const std::string verbosityLevel)
{
  size_t messageLevel = getVerbosityLevel(verbosityLevel);

  if (messageLevel <= _verbosityLevel) return true;
  return false;
}

void Logger::logData(const std::string verbosityLevel, const char *format, ...)
{
  if (isEnoughVerbosity(verbosityLevel) == false) return;

  char *outstr = 0;
  va_list ap;
  va_start(ap, format);
  vasprintf(&outstr, format, ap);

  fprintf(_outputFile, "%s", outstr);
  fflush(_outputFile);
  free(outstr);
}

void Logger::logInfo(const std::string verbosityLevel, const char *format, ...)
{
  if (isEnoughVerbosity(verbosityLevel) == false) return;

  std::string newFormat = "[Korali] ";
  newFormat += format;

  char *outstr = 0;
  va_list ap;
  va_start(ap, format);
  vasprintf(&outstr, newFormat.c_str(), ap);

  fprintf(_outputFile, "%s", outstr);
  fflush(_outputFile);
  free(outstr);
}

void Logger::logWarning(const std::string verbosityLevel, const char *format, ...)
{
  if (isEnoughVerbosity(verbosityLevel) == false) return;

  std::string newFormat = "[Korali] Warning: ";
  newFormat += format;

  char *outstr = 0;
  va_list ap;
  va_start(ap, format);
  vasprintf(&outstr, newFormat.c_str(), ap);

  FILE* outFile = _outputFile == stdout ? stderr : _outputFile;
  fprintf(outFile, "%s", outstr);
  fflush(outFile);

  free(outstr);
}

void Logger::logError [[noreturn]] (const char *fileName, const int lineNumber, const char *format, ...)
{
  char *outstr = 0;
  va_list ap;
  va_start(ap, format);
  vasprintf(&outstr, format, ap);

  std::string outString = outstr;
  free(outstr);

  char info[1024];

  snprintf(info, sizeof(info) - 1, " + From %s:%d\n", fileName, lineNumber);
  outString += info;

  throw std::runtime_error(outString.c_str());
}

} // namespace korali
