#include "math.hpp"
#include "auxiliar/logger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unistd.h>

namespace korali
{
bool isanynan(const std::vector<double> &x)
{
  for (size_t j = 0; j < x.size(); j++)
    if (std::isnan(x[j]) == true) return true;

  return false;
}

double vectorNorm(const std::vector<double> &x)
{
  double norm = 0.;
  for (size_t i = 0; i < x.size(); i++) norm += x[i] * x[i];
  return sqrt(norm);
}

std::string getTimestamp()
{
  time_t rawtime;
  time(&rawtime);
  std::string curTime(ctime(&rawtime));
  return curTime.substr(0, curTime.size() - 1);
}

size_t getTimehash()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}


char decimalToHexChar(const uint8_t byte)
{
  if (byte == 0) return '0';
  if (byte == 1) return '1';
  if (byte == 2) return '2';
  if (byte == 3) return '3';
  if (byte == 4) return '4';
  if (byte == 5) return '5';
  if (byte == 6) return '6';
  if (byte == 7) return '7';
  if (byte == 8) return '8';
  if (byte == 9) return '9';
  if (byte == 10) return 'A';
  if (byte == 11) return 'B';
  if (byte == 12) return 'C';
  if (byte == 13) return 'D';
  if (byte == 14) return 'E';
  if (byte == 15) return 'F';
  return '0';
}

uint8_t hexCharToDecimal(const char x)
{
  if (x == '0') return 0;
  if (x == '1') return 1;
  if (x == '2') return 2;
  if (x == '3') return 3;
  if (x == '4') return 4;
  if (x == '5') return 5;
  if (x == '6') return 6;
  if (x == '7') return 7;
  if (x == '8') return 8;
  if (x == '9') return 9;
  if (x == 'A') return 10;
  if (x == 'B') return 11;
  if (x == 'C') return 12;
  if (x == 'D') return 13;
  if (x == 'E') return 14;
  if (x == 'F') return 15;
  return 0;
}

uint8_t hexPairToByte(const char *src)
{
  uint8_t bH = hexCharToDecimal(src[0]);
  uint8_t bL = hexCharToDecimal(src[1]);
  return bH * 16 + bL;
}

void byteToHexPair(char *dst, const uint8_t byte)
{
  uint8_t bH = byte / 16;
  uint8_t bL = byte % 16;
  dst[0] = decimalToHexChar(bH);
  dst[1] = decimalToHexChar(bL);
}

} // namespace korali
