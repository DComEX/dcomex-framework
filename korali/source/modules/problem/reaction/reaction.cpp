#include "modules/problem/reaction/reaction.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Reaction::initialize()
{
  if (_k->_variables.size() == 0) KORALI_LOG_ERROR("Reaction problems require at least one variable.\n");

  for (size_t idx = 0; idx < _k->_variables.size(); ++idx)
  {
    _reactantNameToIndexMap[_k->_variables[idx]->_name] = idx;
    _initialReactantNumbers.push_back(_k->_variables[idx]->_initialReactantNumber);
    if (_k->_variables[idx]->_initialReactantNumber < 0)
      KORALI_LOG_ERROR("Initial Reactant numer of variable '%s' smaller 0 (is %d)\n", _k->_variables[idx]->_name.c_str(), _k->_variables[idx]->_initialReactantNumber);
  }

  std::vector<bool> used(_k->_variables.size(), false);

  // Parsing user-defined reactions
  for (size_t i = 0; i < _reactions.size(); i++)
  {
    double rate = _reactions[i]["Rate"].get<double>();
    if (rate <= 0.)
      KORALI_LOG_ERROR("Rate of reaction %zu smaller or equal 0 (is %lf)\n", i, rate);

    std::string eq = _reactions[i]["Equation"];

    auto reaction = parseReactionString(eq);
    std::vector<int> reactantIds, productIds;
    for (auto &name : reaction.reactantNames)
    {
      if (_reactantNameToIndexMap.find(name) != _reactantNameToIndexMap.end())
      {
        reactantIds.push_back(_reactantNameToIndexMap[name]);
        used[_reactantNameToIndexMap[name]] = true;
      }
      else
      {
        KORALI_LOG_ERROR("Variable with name '%s' not defined.\n", name.c_str());
      }
    }
    for (auto &name : reaction.productNames)
    {
      if (_reactantNameToIndexMap.find(name) != _reactantNameToIndexMap.end())
      {
        productIds.push_back(_reactantNameToIndexMap[name]);
        used[_reactantNameToIndexMap[name]] = true;
      }
      else
      {
        KORALI_LOG_ERROR("Variable with name '%s' not defined.\n", name.c_str());
      }
    }

    _reactionVector.emplace_back(rate,
                                 std::move(reactantIds),
                                 std::move(reaction.reactantSCs),
                                 std::move(productIds),
                                 std::move(reaction.productSCs),
                                 std::move(reaction.isReactantReservoir));
  }

  for (size_t idx = 0; idx < _k->_variables.size(); ++idx)
    if (used[idx] == false)
      _k->_logger->logWarning("Normal", "Variable with name '%s' initiailized but not used.\n", _k->_variables[idx]->_name.c_str());
}

double Reaction::computePropensity(size_t reactionIndex, const std::vector<int> &reactantNumbers) const
{
  // Get reaction
  const auto &reaction = _reactionVector[reactionIndex];

  double propensity = reaction.rate;

  for (size_t s = 0; s < reaction.reactantIds.size(); ++s)
  {
    const int nu = reaction.reactantStoichiometries[s];
    const int x = reactantNumbers[reaction.reactantIds[s]];

    int numerator = x;
    int denominator = nu;

    for (int k = 1; k < nu; ++k)
    {
      numerator *= x - k;
      denominator *= k;
    }

    propensity *= (double)numerator / denominator;
  }

  return propensity;
}

double Reaction::computeGradPropensity(size_t reactionIndex, const std::vector<int> &reactantNumbers, size_t dI) const
{
  // Get reaction
  const auto &reaction = _reactionVector[reactionIndex];

  // Init gradient of propensity
  double dadxi = reaction.rate;

  for (size_t s = 0; s < reaction.reactantIds.size(); ++s)
  {
    const size_t is = reaction.reactantIds[s];
    const int nu = reaction.reactantStoichiometries[s];
    const int x = reactantNumbers[is];

    double numerator = 0.;
    double denominator = 0.;

    if (dI == is)
    {
      // Gradient of reactant wrt itself
      denominator = nu;

      for (int k = 0; k < nu; ++k)
      {
        int partialNumerator = 1;
        for (int j = 0; j < nu; ++j)
        {
          if (j != k)
            partialNumerator *= x - j;
        }
        denominator *= std::max(1, k);
        numerator += partialNumerator;
      }
    }
    else
    {
      // Gradient of reactant wrt other
      numerator = x;
      denominator = nu;

      for (int k = 1; k < nu; ++k)
      {
        numerator *= x - k;
        denominator *= k;
      }
    }

    // update gradient
    dadxi *= numerator / denominator;
  }

  return dadxi;
}

double Reaction::computeF(size_t reactionIndex, size_t otherReactionIndex, const std::vector<int> &reactantNumbers)
{
  // Init state change matrix
  if (_stateChange.size() == 0) setStateChange(reactantNumbers.size());

  const auto &reaction = _reactionVector[reactionIndex];

  double f = 0.;
  for (int id : reaction.reactantIds)
    f += computeGradPropensity(reactionIndex, reactantNumbers, id) * _stateChange[otherReactionIndex][id];

  return f;
}

double Reaction::calculateMaximumAllowedFirings(size_t reactionIndex, const std::vector<int> &reactantNumbers) const
{
  const auto &reaction = _reactionVector[reactionIndex];

  int L = std::numeric_limits<int>::max();

  for (size_t s = 0; s < reaction.reactantIds.size(); ++s)
  {
    const int x = reactantNumbers[reaction.reactantIds[s]];
    const int nu = reaction.reactantStoichiometries[s];
    if (nu > 0)
      L = std::min(L, x / nu);
  }

  return L;
}

void Reaction::applyChanges(size_t reactionIndex, std::vector<int> &reactantNumbers, int numFirings) const
{
  const auto &reaction = _reactionVector[reactionIndex];

  for (size_t s = 0; s < reaction.reactantIds.size(); ++s)
  {
    if (!reaction.isReactantReservoir[s])
      reactantNumbers[reaction.reactantIds[s]] -= numFirings * reaction.reactantStoichiometries[s];
  }

  for (size_t s = 0; s < reaction.productIds.size(); ++s)
  {
    reactantNumbers[reaction.productIds[s]] += numFirings * reaction.productStoichiometries[s];
  }
}

void Reaction::setStateChange(size_t numReactants)
{
  _stateChange.resize(_reactionVector.size());
  for (size_t k = 0; k < _reactionVector.size(); ++k)
  {
    _stateChange[k].resize(numReactants, 0);
    applyChanges(k, _stateChange[k]);
  }
}

void Reaction::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Reactions"))
 {
 _reactions = js["Reactions"].get<knlohmann::json>();

   eraseValue(js, "Reactions");
 }

 if (isDefined(js, "Reactant Name To Index Map"))
 {
 try { _reactantNameToIndexMap = js["Reactant Name To Index Map"].get<std::map<std::string, int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reaction ] \n + Key:    ['Reactant Name To Index Map']\n%s", e.what()); } 
   eraseValue(js, "Reactant Name To Index Map");
 }

 if (isDefined(js, "Initial Reactant Numbers"))
 {
 try { _initialReactantNumbers = js["Initial Reactant Numbers"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reaction ] \n + Key:    ['Initial Reactant Numbers']\n%s", e.what()); } 
   eraseValue(js, "Initial Reactant Numbers");
 }

 if (isDefined(js, "State Change"))
 {
 try { _stateChange = js["State Change"].get<std::vector<std::vector<int>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reaction ] \n + Key:    ['State Change']\n%s", e.what()); } 
   eraseValue(js, "State Change");
 }

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Reactant Number"))
 {
 try { _k->_variables[i]->_initialReactantNumber = _k->_js["Variables"][i]["Initial Reactant Number"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reaction ] \n + Key:    ['Initial Reactant Number']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Reactant Number");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Reactant Number'] required by reaction.\n"); 

 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("SSM"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: reaction\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "reaction";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: reaction: \n%s\n", js.dump(2).c_str());
} 

void Reaction::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Reactions"] = _reactions;
   js["Reactant Name To Index Map"] = _reactantNameToIndexMap;
   js["Initial Reactant Numbers"] = _initialReactantNumbers;
   js["State Change"] = _stateChange;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Reactant Number"] = _k->_variables[i]->_initialReactantNumber;
 } 
 Problem::getConfiguration(js);
} 

void Reaction::applyModuleDefaults(knlohmann::json& js) 
{

 Problem::applyModuleDefaults(js);
} 

void Reaction::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Reactant Number\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

;

} //problem
} //korali
;
