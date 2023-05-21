#include "modules/distribution/distribution.hpp"
#include "modules/experiment/experiment.hpp"
#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>

namespace korali
{
;

std::string
Distribution::getRange(gsl_rng *range) const
{
  unsigned char *state = (unsigned char *)gsl_rng_state(range);
  size_t n = gsl_rng_size(range);
  char *hexString = (char *)calloc(sizeof(char), n * 2 + 1);

  for (size_t i = 0; i < n; i++)
    byteToHexPair(&hexString[i * 2], state[i]);

  hexString[n * 2] = '\0';

  std::string output = std::string(hexString);
  free(hexString);
  return output;
}

gsl_rng *Distribution::setRange(const std::string rangeString)
{
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

  if (_randomSeed == 0 || _k->_preserveRandomNumberGeneratorStates == false)
    _randomSeed = _k->_randomSeed++;

  gsl_rng_set(rng, _randomSeed);

  void *state = gsl_rng_state(rng);

  size_t n = rangeString.size() >> 1;
  size_t m = gsl_rng_size(rng);

  if (_k->_preserveRandomNumberGeneratorStates == true && rangeString != "")
  {
    if (m != n) KORALI_LOG_ERROR("Invalid GSL state size: %lu != %lu\n", m, n);

    const char *rngHexString = rangeString.c_str();

    for (size_t i = 0; i < n; i++)
      ((char *)state)[i] = (char)hexPairToByte(&rngHexString[i * 2]);
  }

  return rng;
}

void Distribution::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Name"))
 {
 try { _name = js["Name"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ distribution ] \n + Key:    ['Name']\n%s", e.what()); } 
   eraseValue(js, "Name");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Name'] required by distribution.\n"); 

 if (isDefined(js, "Random Seed"))
 {
 try { _randomSeed = js["Random Seed"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ distribution ] \n + Key:    ['Random Seed']\n%s", e.what()); } 
   eraseValue(js, "Random Seed");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Random Seed'] required by distribution.\n"); 

 if (isDefined(js, "Range"))
 {
 try { _range = setRange(js["Range"].get<std::string>());
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ distribution ] \n + Key:    ['Range']\n%s", e.what()); } 
   eraseValue(js, "Range");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Range'] required by distribution.\n"); 

 Module::setConfiguration(js);
 _type = ".";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: distribution: \n%s\n", js.dump(2).c_str());
} 

void Distribution::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Name"] = _name;
   js["Random Seed"] = _randomSeed;
   js["Range"] = getRange(_range);
 Module::getConfiguration(js);
} 

void Distribution::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Name\": \"\", \"Random Seed\": 0, \"Range\": \"\"}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Module::applyModuleDefaults(js);
} 

void Distribution::applyVariableDefaults() 
{

 Module::applyVariableDefaults();
} 

;

} //korali
;
