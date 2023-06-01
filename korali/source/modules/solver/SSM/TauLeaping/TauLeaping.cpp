#include "modules/solver/SSM/TauLeaping/TauLeaping.hpp"

namespace korali
{
namespace solver
{
namespace ssm
{
;

void TauLeaping::setInitialConfiguration()
{
  // Initialize SSM parameter
  SSM::setInitialConfiguration();

  // TauLeaping parameter checks
  if (_epsilon <= 0. || _epsilon >= 1.)
    KORALI_LOG_ERROR("Epsilon must be in range (0,1), is: %lf\n", _epsilon);

  if (_nc <= 0)
    KORALI_LOG_ERROR("Nc must be larger 0, is: %d\n", _nc);

  if (_acceptanceFactor <= 0.)
    KORALI_LOG_ERROR("Acceptance Factor must be larger 0., is: %lf\n", _acceptanceFactor);
}

void TauLeaping::ssaAdvance()
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
  _problem->applyChanges(selection, _numReactants, 1);
}

void TauLeaping::advance()
{
  _propensities.resize(_numReactions);

  double a0 = 0.0;

  // Calculate propensities
  for (size_t k = 0; k < _numReactions; ++k)
  {
    double a = _problem->computePropensity(k, _numReactants);

    a0 += a;
    _propensities[k] = a;
  }

  // Mark critical reactions
  bool allReactionsAreCritical = true;
  _isCriticalReaction.resize(_numReactions);

  for (size_t k = 0; k < _numReactions; ++k)
  {
    const double a = _propensities[k];
    const double L = _problem->calculateMaximumAllowedFirings(k, _numReactants);

    const bool isCritical = (a > 0) && (L <= _nc);
    _isCriticalReaction[k] = isCritical;

    allReactionsAreCritical = allReactionsAreCritical && isCritical;
  }

  // Estimate maximum tau
  double tauP = allReactionsAreCritical ? std::numeric_limits<double>::infinity() : estimateLargestTau();

  if (_time + tauP > _simulationLength)
    tauP = _simulationLength - _time;

  // Accept or reject step
  if (tauP < _acceptanceFactor / a0)
  {
    // reject (tau leap step is short), execute SSA steps
    for (int i = 0; i < _numSSASteps; ++i)
    {
      ssaAdvance();
      if (_time >= _simulationLength)
        break;

      // last update happens in ssm base class
      if (i < _numSSASteps - 1)
        updateBins();
    }
  }
  else
  {
    // accept, perform tau leap step

    // accumulate propensities of critical reactions
    double a0c = 0;
    for (size_t k = 0; k < _numReactions; ++k)
    {
      if (_isCriticalReaction[k])
        a0c += _propensities[k];
    }

    // sample tauPP from exponential
    const double tauPP = -std::log(_uniformGenerator->getRandomNumber()) / a0c;

    double tau;
    bool anyReactantNegative = false;

    // Sample candidates while avoiding negative reactants
    do
    {
      tau = std::min(tauP, tauPP);

      if (_time + tau > _simulationLength)
        tau = _simulationLength - _time;

      _numFirings.resize(_numReactions, 0);

      for (size_t i = 0; i < _numReactions; ++i)
      {
        if (_isCriticalReaction[i])
        {
          _numFirings[i] = 0;
        }
        else
        {
          _poissonGenerator->_mean = _propensities[i] * tau;
          _numFirings[i] = _poissonGenerator->getRandomNumber();
        }
      }

      // A critical reaction fires
      if (tauPP <= tauP)
      {
        _cumPropensities.resize(_numReactions);
        double cumulative = 0;
        for (size_t i = 0; i < _numReactions; ++i)
        {
          if (_isCriticalReaction[i])
            cumulative += _propensities[i];
          _cumPropensities[i] = cumulative;
        }

        const double u = a0c * _uniformGenerator->getRandomNumber();
        size_t jc = 0;
        while (jc < _numReactions && (!_isCriticalReaction[jc] || u > _cumPropensities[jc]))
        {
          ++jc;
        }

        _numFirings[jc] = 1;
      }

      _candidateNumReactants = _numReactants;

      // Fire reactions and produce candidates
      for (size_t i = 0; i < _numReactions; ++i)
      {
        const int ki = _numFirings[i];
        if (ki > 0)
          _problem->applyChanges(i, _candidateNumReactants, ki);
      }

      // Check if any candidate is negative, if yes half time of tauP and repeat
      anyReactantNegative = false;
      for (auto candidate : _candidateNumReactants)
      {
        if (candidate < 0)
        {
          anyReactantNegative = true;
          tauP /= 2.;
          break;
        }
      }
    } while (anyReactantNegative);

    _time += tau;
    std::copy(_candidateNumReactants.begin(), _candidateNumReactants.end(), _numReactants.begin());
  }
}

double TauLeaping::estimateLargestTau()
{
  _mu.resize(_numReactions, 0.);
  _variance.resize(_numReactions, 0.);

  double a0 = 0.;
  for (size_t j = 0; j < _numReactions; ++j)
  {
    for (size_t jp = 0; jp < _numReactions; ++jp)
    {
      if (_isCriticalReaction[jp])
        continue;

      const double fjjp = _problem->computeF(j, jp, _numReactants);

      _mu[j] += fjjp * _propensities[jp];
      _variance[j] += fjjp * fjjp * _propensities[jp];
    }

    a0 += _propensities[j];
  }

  double tau = std::numeric_limits<double>::max();

  for (size_t i = 0; i < _numReactions; ++i)
  {
    const double muTerm = _epsilon * a0 / std::abs(_mu[i]);
    const double varTerm = _epsilon * _epsilon * a0 * a0 / _variance[i];

    tau = std::min(tau, std::min(muTerm, varTerm));
  }

  return tau;
}

void TauLeaping::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Poisson Generator"))
 {
 _poissonGenerator = dynamic_cast<korali::distribution::univariate::Poisson*>(korali::Module::getModule(js["Poisson Generator"], _k));
 _poissonGenerator->applyVariableDefaults();
 _poissonGenerator->applyModuleDefaults(js["Poisson Generator"]);
 _poissonGenerator->setConfiguration(js["Poisson Generator"]);
   eraseValue(js, "Poisson Generator");
 }

 if (isDefined(js, "Mu"))
 {
 try { _mu = js["Mu"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Mu']\n%s", e.what()); } 
   eraseValue(js, "Mu");
 }

 if (isDefined(js, "Variance"))
 {
 try { _variance = js["Variance"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Variance']\n%s", e.what()); } 
   eraseValue(js, "Variance");
 }

 if (isDefined(js, "Nc"))
 {
 try { _nc = js["Nc"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Nc']\n%s", e.what()); } 
   eraseValue(js, "Nc");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Nc'] required by TauLeaping.\n"); 

 if (isDefined(js, "Epsilon"))
 {
 try { _epsilon = js["Epsilon"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Epsilon']\n%s", e.what()); } 
   eraseValue(js, "Epsilon");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Epsilon'] required by TauLeaping.\n"); 

 if (isDefined(js, "Acceptance Factor"))
 {
 try { _acceptanceFactor = js["Acceptance Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Acceptance Factor']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Acceptance Factor'] required by TauLeaping.\n"); 

 if (isDefined(js, "Num SSA Steps"))
 {
 try { _numSSASteps = js["Num SSA Steps"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Num SSA Steps']\n%s", e.what()); } 
   eraseValue(js, "Num SSA Steps");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Num SSA Steps'] required by TauLeaping.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 SSM::setConfiguration(js);
 _type = "SSM/TauLeaping";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: TauLeaping: \n%s\n", js.dump(2).c_str());
} 

void TauLeaping::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Nc"] = _nc;
   js["Epsilon"] = _epsilon;
   js["Acceptance Factor"] = _acceptanceFactor;
   js["Num SSA Steps"] = _numSSASteps;
 if(_poissonGenerator != NULL) _poissonGenerator->getConfiguration(js["Poisson Generator"]);
   js["Mu"] = _mu;
   js["Variance"] = _variance;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 SSM::getConfiguration(js);
} 

void TauLeaping::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Poisson Generator\": {\"Type\": \"Univariate/Poisson\", \"Mean\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 SSM::applyModuleDefaults(js);
} 

void TauLeaping::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 SSM::applyVariableDefaults();
} 

bool TauLeaping::checkTermination()
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
