#include "engine.hpp"
#include "auxiliar/fs.hpp"
#include "auxiliar/koraliJson.hpp"
#include "modules/conduit/conduit.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/conduit/distributed/distributed.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/solver.hpp"
#include "sample/sample.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace korali
{
std::stack<Engine *> _engineStack;
bool isPythonActive = 0;

Engine::Engine()
{
  #ifdef _KORALI_USE_MPI
  __isMPICommGiven = false;
  #endif

  _cumulativeTime = 0.0;
  _thread = co_active();
  _conduit = NULL;

  // Turn Off GSL Error Handler
  gsl_set_error_handler_off();
}

void Engine::initializeExperiments()
{
  // Initializing experiment's configuration
  for (size_t i = 0; i < _experimentVector.size(); i++)
  {
    _experimentVector[i]->_experimentId = i;
    _experimentVector[i]->_engine = this;
    _experimentVector[i]->initialize();
    _experimentVector[i]->_isFinished = false;
  }
}

void Engine::start()
{
  // Setting Engine configuration defaults
  if (!isDefined(_js.getJson(), "Profiling", "Detail")) _js["Profiling"]["Detail"] = "None";
  if (!isDefined(_js.getJson(), "Profiling", "Path")) _js["Profiling"]["Path"] = "./profiling.json";
  if (!isDefined(_js.getJson(), "Profiling", "Frequency")) _js["Profiling"]["Frequency"] = 60.0;
  if (!isDefined(_js.getJson(), "Conduit", "Type")) _js["Conduit"]["Type"] = "Sequential";
  if (!isDefined(_js.getJson(), "Dry Run")) _js["Dry Run"] = false;

  // Loading configuration values
  _isDryRun = _js["Dry Run"];
  _profilingPath = _js["Profiling"]["Path"];
  _profilingDetail = _js["Profiling"]["Detail"];
  _profilingFrequency = _js["Profiling"]["Frequency"];

  // Checking if its a dry run and return if it is
  if (_isDryRun) return;

  // Only initialize conduit if the Engine being ran is the first one in the process
  auto conduit = dynamic_cast<Conduit *>(Module::getModule(_js["Conduit"], NULL));
  conduit->applyModuleDefaults(_js["Conduit"]);
  conduit->setConfiguration(_js["Conduit"]);
  conduit->initialize();

  // Initializing conduit server
  conduit->initServer();

  // Assigning pointer after starting workers, so they can initialize their own conduit
  _conduit = conduit;

  // Recovering Conduit configuration in case of restart
  _conduit->getConfiguration(_js.getJson()["Conduit"]);

  // Now initializing experiments
  initializeExperiments();

  // Check configuration correctness
  auto js = _js.getJson();
  if (isDefined(js, "Conduit")) eraseValue(js, "Conduit");
  if (isDefined(js, "Dry Run")) eraseValue(js, "Dry Run");
  if (isDefined(js, "Conduit", "Type")) eraseValue(js, "Conduit", "Type");
  if (isDefined(js, "Profiling", "Detail")) eraseValue(js, "Profiling", "Detail");
  if (isDefined(js, "Profiling", "Path")) eraseValue(js, "Profiling", "Path");
  if (isDefined(js, "Profiling", "Frequency")) eraseValue(js, "Profiling", "Frequency");

  if (isEmpty(js) == false) KORALI_LOG_ERROR("Unrecognized settings for Korali's Engine: \n%s\n", js.dump(2).c_str());

  if (_conduit->isRoot())
  {
    // Adding engine to the stack to support Korali-in-Korali execution
    _conduit->stackEngine(this);

    // Setting base time for profiling.
    _startTime = std::chrono::high_resolution_clock::now();
    _profilingLastSave = std::chrono::high_resolution_clock::now();

    while (true)
    {
      // Checking for break signals coming from Python
      bool executed = false;
      for (size_t i = 0; i < _experimentVector.size(); i++)
        if (_experimentVector[i]->_isFinished == false)
        {
          _currentExperiment = _experimentVector[i];
          co_switch(_experimentVector[i]->_thread);
          executed = true;
          saveProfilingInfo(false);
        }
      if (executed == false) break;
    }

    _endTime = std::chrono::high_resolution_clock::now();

    saveProfilingInfo(true);
    _cumulativeTime += std::chrono::duration<double>(_endTime - _startTime).count();

    // Finalizing experiments
    for (size_t i = 0; i < _experimentVector.size(); i++) _experimentVector[i]->finalize();

    // (Workers-Side) Removing the current engine to the conduit's engine stack
    _conduit->popEngine();
  }

  // Finalizing Conduit if last engine in the stack
  _conduit->terminateServer();
  delete _conduit;
}

void Engine::saveProfilingInfo(const bool forceSave)
{
  if (_profilingDetail == "Full")
  {
    auto currTime = std::chrono::high_resolution_clock::now();
    double timeSinceLast = std::chrono::duration<double>(currTime - _profilingLastSave).count();
    if ((timeSinceLast > _profilingFrequency) || forceSave)
    {
      double elapsedTime = std::chrono::duration<double>(currTime - _startTime).count();
      __profiler["Experiment Count"] = _experimentVector.size();
      __profiler["Elapsed Time"] = elapsedTime + _cumulativeTime;
      saveJsonToFile(_profilingPath.c_str(), __profiler);
      _profilingLastSave = std::chrono::high_resolution_clock::now();
    }
  }
}

void Engine::run(Experiment &experiment)
{
  _experimentVector.clear();
  experiment._k->_engine = this;
  _experimentVector.push_back(experiment._k);
  start();
}

void Engine::run(std::vector<Experiment> &experiments)
{
  _experimentVector.clear();
  for (size_t i = 0; i < experiments.size(); i++)
  {
    experiments[i]._k->_engine = this;
    _experimentVector.push_back(experiments[i]._k);
  }
  start();
}

void Engine::serialize(knlohmann::json &js)
{
  for (size_t i = 0; i < _experimentVector.size(); i++)
  {
    _experimentVector[i]->getConfiguration(_experimentVector[i]->_js.getJson());
    js["Experiment Vector"][i] = _experimentVector[i]->_js.getJson();
  }
}

knlohmann::json &Engine::operator[](const std::string &key)
{
  return _js[key];
}
knlohmann::json &Engine::operator[](const unsigned long int &key) { return _js[key]; }
pybind11::object Engine::getItem(const pybind11::object key) { return _js.getItem(key); }
void Engine::setItem(const pybind11::object key, const pybind11::object val) { _js.setItem(key, val); }


#ifdef _KORALI_USE_MPI4PY
#ifndef _KORALI_NO_MPI4PY

void Engine::setMPI4PyComm(mpi4py_comm comm)
{
 korali::setMPI4PyComm(comm);
}

#endif
#endif

} // namespace korali

using namespace korali;

PYBIND11_MODULE(libkorali, m)
{
  #ifdef _KORALI_USE_MPI
  #ifdef _KORALI_USE_MPI4PY
  #ifndef _KORALI_NO_MPI4PY

  // import the mpi4py API
  if (import_mpi4py() < 0) { throw std::runtime_error("Could not load mpi4py API."); }
  m.def("getWorkerMPI4PyComm", &getMPI4PyComm);

  #endif
  #endif
  #endif

  pybind11::class_<Engine>(m, "Engine")
    .def(pybind11::init<>())
    .def("run", [](Engine &k, Experiment &e) {
      isPythonActive = true;
      k.run(e);
    })

    .def("run", [](Engine &k, std::vector<Experiment> &e) {
      isPythonActive = true;
      k.run(e);
    })

    #ifdef _KORALI_USE_MPI
    #ifdef _KORALI_USE_MPI4PY
    #ifndef _KORALI_NO_MPI4PY
     .def("setMPIComm", &Engine::setMPI4PyComm)
    #endif
    #endif
    #endif

    .def("__getitem__", pybind11::overload_cast<pybind11::object>(&Engine::getItem), pybind11::return_value_policy::reference)
    .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&Engine::setItem), pybind11::return_value_policy::reference);


  pybind11::class_<KoraliJson>(m, "koraliJson")
    .def("__getitem__", pybind11::overload_cast<pybind11::object>(&KoraliJson::getItem), pybind11::return_value_policy::reference)
    .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&KoraliJson::setItem), pybind11::return_value_policy::reference);

  pybind11::class_<Sample>(m, "Sample")
    .def("__getitem__", pybind11::overload_cast<pybind11::object>(&Sample::getItem), pybind11::return_value_policy::reference)
    .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&Sample::setItem), pybind11::return_value_policy::reference)
    .def("update", &Sample::update);

  pybind11::class_<Experiment>(m, "Experiment")
    .def(pybind11::init<>())
    .def("__getitem__", pybind11::overload_cast<pybind11::object>(&Experiment::getItem), pybind11::return_value_policy::reference)
    .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&Experiment::setItem), pybind11::return_value_policy::reference)
    .def("loadState", &Experiment::loadState);
}
