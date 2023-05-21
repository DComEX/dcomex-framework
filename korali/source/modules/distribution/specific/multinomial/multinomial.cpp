#include "modules/distribution/specific/multinomial/multinomial.hpp"
#include "modules/experiment/experiment.hpp"
#include <gsl/gsl_randist.h>

namespace korali
{
namespace distribution
{
namespace specific
{
;

void Multinomial::getSelections(std::vector<double> &p, std::vector<unsigned int> &n, int N)
{
  gsl_ran_multinomial(_range, p.size(), N, p.data(), n.data());
}

void Multinomial::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Specific::setConfiguration(js);
 _type = "specific/multinomial";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: multinomial: \n%s\n", js.dump(2).c_str());
} 

void Multinomial::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 Specific::getConfiguration(js);
} 

void Multinomial::applyModuleDefaults(knlohmann::json& js) 
{

 Specific::applyModuleDefaults(js);
} 

void Multinomial::applyVariableDefaults() 
{

 Specific::applyVariableDefaults();
} 

;

} //specific
} //distribution
} //korali
;
