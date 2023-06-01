#include "modules/solver/SSM/SSA/SSA.hpp"

namespace korali
{
namespace solver
{
namespace ssm
{
;

void SSA::advance()
{
  _cumPropensities.resize(_numReactions);

  double a0 = 0.0;

  // Calculate propensities
  for (size_t k = 0; k < _numReactions; ++k)
  {
    const double a = _problem->computePropensity(k, _numReactants);

    a0 += a;
    _cumPropensities[k] = a0;
  }

  // Sample time step from exponential distribution
  const double r1 = _uniformGenerator->getRandomNumber();

  double tau = -std::log(r1) / a0;

  // Advance time
  _time += tau;

  if (_time > _simulationLength)
    _time = _simulationLength;

  // Exit if no reactions fire
  if (a0 == 0)
    return;

  const double r2 = _cumPropensities.back() * _uniformGenerator->getRandomNumber();

  // Sample a reaction
  size_t selection = 0;
  while (r2 > _cumPropensities[selection])
    selection++;

  // Update the reactants according to chosen reaction
  _problem->applyChanges(selection, _numReactants);
}

void SSA::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 SSM::setConfiguration(js);
 _type = "SSM/SSA";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: SSA: \n%s\n", js.dump(2).c_str());
} 

void SSA::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 SSM::getConfiguration(js);
} 

void SSA::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 SSM::applyModuleDefaults(js);
} 

void SSA::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 SSM::applyVariableDefaults();
} 

bool SSA::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || SSM::checkTermination();
 return hasFinished;
}

;

} //ssm
} //solver
} //korali
;
