#include "modules/solver/SSM/SSM.hpp"

namespace korali
{
namespace solver
{
;

void SSM::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();
  _problem = dynamic_cast<problem::Reaction *>(_k->_problem);

  // Initialize time of bins
  double dt = _simulationLength / (double)_diagnosticsNumBins;
  _binTime.resize(_diagnosticsNumBins);
  // clang-format off
  std::generate(_binTime.begin(), _binTime.end(), [idx = 0, dt]() mutable {
    return idx++ * dt;
  });
  // clang-format on

  // Init simulation counter
  _completedSimulations = 0;

  // Initialize number of reactions to simulate
  _numReactions = _problem->_reactions.size();

  // Initialize bin related memory
  _binCounter = std::vector<std::vector<int>>(_maxNumSimulations, std::vector<int>(_diagnosticsNumBins, 0));
  _binnedTrajectories = std::vector<std::vector<std::vector<int>>>(_variableCount, std::vector<std::vector<int>>(_maxNumSimulations, std::vector<int>(_diagnosticsNumBins, 0)));
}

void SSM::reset(std::vector<int> numReactants, double time)
{
  // Set time and number of reactants
  _time = time;
  _numReactants = std::move(numReactants);
}

void SSM::updateBins()
{
  // Find bin index of current time
  size_t binIndex = _time / _simulationLength * _diagnosticsNumBins;
  _binCounter[_completedSimulations][binIndex] += 1;

  // Accumulate reactants in bin
  for (size_t k = 0; k < _variableCount; k++)
  {
    _binnedTrajectories[k][_completedSimulations][binIndex] += _numReactants[k];
  }
}

void SSM::runGeneration()
{
  if (_k->_currentGeneration == 1)
  {
    setInitialConfiguration();
  }

  for (size_t run = 0; run < _simulationsPerGeneration; ++run)
  {
    // Start new simulation
    reset(_problem->_initialReactantNumbers);
    updateBins();

    // Stimulate until termination
    while (_time < _simulationLength)
    {
      advance();
      updateBins();
    }

    _completedSimulations++;

    if (_completedSimulations >= _maxNumSimulations) return;
  }
}

void SSM::printGenerationBefore() { return; }

void SSM::printGenerationAfter()
{
  _k->_logger->logInfo("Normal", "Completed Simulations: %zu / %zu\n", _completedSimulations, _maxNumSimulations);
}

void SSM::finalize()
{
  // Calculate mean trajectory
  std::vector<std::vector<double>> resultsMeanTrajectory(_variableCount, std::vector<double>(_diagnosticsNumBins, 0.));
  for (size_t k = 0; k < _variableCount; k++)
  {
    for (size_t idx = 0; idx < _diagnosticsNumBins; ++idx)
    {
      size_t binCount = 0;
      for (size_t sim = 0; sim < _maxNumSimulations; ++sim)
      {
        resultsMeanTrajectory[k][idx] += _binnedTrajectories[k][sim][idx];
        binCount += _binCounter[sim][idx];
      }
      if (binCount > 0)
        resultsMeanTrajectory[k][idx] /= (double)binCount;
    }
  }

  // Store results
  (*_k)["Results"]["Time"] = _binTime;
  (*_k)["Results"]["Mean Trajectory"] = resultsMeanTrajectory;
}

void SSM::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Time"))
 {
 try { _time = js["Time"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Time']\n%s", e.what()); } 
   eraseValue(js, "Time");
 }

 if (isDefined(js, "Num Reactions"))
 {
 try { _numReactions = js["Num Reactions"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Num Reactions']\n%s", e.what()); } 
   eraseValue(js, "Num Reactions");
 }

 if (isDefined(js, "Num Reactants"))
 {
 try { _numReactants = js["Num Reactants"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Num Reactants']\n%s", e.what()); } 
   eraseValue(js, "Num Reactants");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Bin Time"))
 {
 try { _binTime = js["Bin Time"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Bin Time']\n%s", e.what()); } 
   eraseValue(js, "Bin Time");
 }

 if (isDefined(js, "Bin Counter"))
 {
 try { _binCounter = js["Bin Counter"].get<std::vector<std::vector<int>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Bin Counter']\n%s", e.what()); } 
   eraseValue(js, "Bin Counter");
 }

 if (isDefined(js, "Binned Trajectories"))
 {
 try { _binnedTrajectories = js["Binned Trajectories"].get<std::vector<std::vector<std::vector<int>>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Binned Trajectories']\n%s", e.what()); } 
   eraseValue(js, "Binned Trajectories");
 }

 if (isDefined(js, "Completed Simulations"))
 {
 try { _completedSimulations = js["Completed Simulations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Completed Simulations']\n%s", e.what()); } 
   eraseValue(js, "Completed Simulations");
 }

 if (isDefined(js, "Simulation Length"))
 {
 try { _simulationLength = js["Simulation Length"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Simulation Length']\n%s", e.what()); } 
   eraseValue(js, "Simulation Length");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Simulation Length'] required by SSM.\n"); 

 if (isDefined(js, "Diagnostics", "Num Bins"))
 {
 try { _diagnosticsNumBins = js["Diagnostics"]["Num Bins"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Diagnostics']['Num Bins']\n%s", e.what()); } 
   eraseValue(js, "Diagnostics", "Num Bins");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Diagnostics']['Num Bins'] required by SSM.\n"); 

 if (isDefined(js, "Simulations Per Generation"))
 {
 try { _simulationsPerGeneration = js["Simulations Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Simulations Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Simulations Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Simulations Per Generation'] required by SSM.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Num Simulations"))
 {
 try { _maxNumSimulations = js["Termination Criteria"]["Max Num Simulations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Termination Criteria']['Max Num Simulations']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Num Simulations");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Num Simulations'] required by SSM.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Solver::setConfiguration(js);
 _type = "SSM";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: SSM: \n%s\n", js.dump(2).c_str());
} 

void SSM::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Simulation Length"] = _simulationLength;
   js["Diagnostics"]["Num Bins"] = _diagnosticsNumBins;
   js["Simulations Per Generation"] = _simulationsPerGeneration;
   js["Termination Criteria"]["Max Num Simulations"] = _maxNumSimulations;
   js["Time"] = _time;
   js["Num Reactions"] = _numReactions;
   js["Num Reactants"] = _numReactants;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Bin Time"] = _binTime;
   js["Bin Counter"] = _binCounter;
   js["Binned Trajectories"] = _binnedTrajectories;
   js["Completed Simulations"] = _completedSimulations;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void SSM::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Simulations Per Generation\": 1, \"Diagnostics\": {\"Num Bins\": 100}, \"Termination Criteria\": {\"Max Num Simulations\": 1}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void SSM::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool SSM::checkTermination()
{
 bool hasFinished = false;

 if (_maxNumSimulations <= _completedSimulations)
 {
  _terminationCriteria.push_back("SSM['Max Num Simulations'] = " + std::to_string(_maxNumSimulations) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;
