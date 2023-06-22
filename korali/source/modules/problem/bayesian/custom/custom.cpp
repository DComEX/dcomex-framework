#include "modules/conduit/conduit.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/bayesian/custom/custom.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
namespace bayesian
{
;

void Custom::initialize()
{
  Bayesian::initialize();

  if (_k->_variables.size() == 0) KORALI_LOG_ERROR("Bayesian inference problems require at least one variable.\n");
}

void Custom::evaluateLoglikelihood(Sample &sample)
{
  sample.run(_likelihoodModel);

  if (!sample.contains("logLikelihood")) KORALI_LOG_ERROR("The specified likelihood model did not assign the value: 'logLikelihood' to the sample.\n");
}

void Custom::evaluateLoglikelihoodGradient(Sample &sample)
{
  if (sample.contains("logLikelihood Gradient") == false)
  {
    sample.run(_likelihoodModel);

    if (sample.contains("logLikelihood Gradient") == false)
      KORALI_LOG_ERROR("The specified likelihood model did not assign the value: 'logLikelihood Gradient' to the sample.\n");
  }
  if (sample["logLikelihood Gradient"].size() != _k->_variables.size()) KORALI_LOG_ERROR("Bayesian problem of type Custom requires likelihood gradient of size %zu (provided size %zu)\n", _k->_variables.size(), sample["loglikelihood Gradient"].size());
}

void Custom::evaluateFisherInformation(Sample &sample)
{
  if (!sample.contains("Fisher Information")) KORALI_LOG_ERROR("The specified likelihood model did not assign the value: 'Fisher Information' to the sample.\n");

  size_t Nth = _k->_variables.size();
  if (sample["Fisher Information"].size() != Nth) KORALI_LOG_ERROR("Bayesian problem of type Custom requires Fisher Information of size %zux%zu\n", Nth, Nth);

  for (size_t d = 0; d < Nth; ++d)
    if (sample["Fisher Information"][d].size() != Nth) KORALI_LOG_ERROR("Bayesian problem of type Custom requires Fisher Information of size %zux%zu\n", Nth, Nth);
}

void Custom::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Likelihood Model"))
 {
 try { _likelihoodModel = js["Likelihood Model"].get<std::uint64_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ custom ] \n + Key:    ['Likelihood Model']\n%s", e.what()); } 
   eraseValue(js, "Likelihood Model");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Likelihood Model'] required by custom.\n"); 

 Bayesian::setConfiguration(js);
 _type = "bayesian/custom";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: custom: \n%s\n", js.dump(2).c_str());
} 

void Custom::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Likelihood Model"] = _likelihoodModel;
 Bayesian::getConfiguration(js);
} 

void Custom::applyModuleDefaults(knlohmann::json& js) 
{

 Bayesian::applyModuleDefaults(js);
} 

void Custom::applyVariableDefaults() 
{

 Bayesian::applyVariableDefaults();
} 

;

} //bayesian
} //problem
} //korali
;
