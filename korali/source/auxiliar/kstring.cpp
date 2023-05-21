#include "kstring.hpp"
#include <algorithm>
#include <cctype>
#include <string>

namespace korali
{
std::string toLower(const std::string &input)
{
  auto s = input;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

bool iCompare(const std::string &a, const std::string &b)
{
  return toLower(a) == toLower(b);
}

} // namespace korali
