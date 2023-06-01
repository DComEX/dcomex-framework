#include "modules/problem/supervisedLearning/supervisedLearning.hpp"

namespace korali
{
namespace problem
{
;

void SupervisedLearning::initialize()
{
  // Checking batch size
  if (_trainingBatchSize == 0) KORALI_LOG_ERROR("Empty input batch provided.\n");
  if (_maxTimesteps == 0) KORALI_LOG_ERROR("Incorrect max timesteps provided: %lu.\n", _maxTimesteps);
  if (_inputSize == 0) KORALI_LOG_ERROR("Empty input vector size provided.\n");
  if (_solutionSize == 0) KORALI_LOG_ERROR("Empty solution vector size provided.\n");
}

void SupervisedLearning::verifyData()
{
  // Checking for empty input and solution data
  if (_inputData.size() == 0) KORALI_LOG_ERROR("Empty input dataset provided.\n");
  if (_solutionData.size() == 0) KORALI_LOG_ERROR("Empty solution dataset provided.\n");

  // Checking that batch entry has the correct size
  if (_trainingBatchSize != _inputData.size())
    KORALI_LOG_ERROR("Training Batch size %lu  different than that of input data (%lu).\n", _inputData.size(), _trainingBatchSize);

  // Checking that all timestep entries have the correct size
  for (size_t b = 0; b < _inputData.size(); b++)
  {
    if (_inputData[b].size() > _maxTimesteps)
      KORALI_LOG_ERROR("More timesteps (%lu) provided in batch %lu than max specified in the configuration (%lu).\n", _inputData[b].size(), b, _maxTimesteps);

    // Checking that all batch entries have the correct size
    for (size_t t = 0; t < _inputData[b].size(); t++)
      if (_inputData[b][t].size() != _inputSize)
        KORALI_LOG_ERROR("Vector size of timestep %lu input data %lu is inconsistent. Size: %lu - Expected: %lu.\n", b, t, _inputData[b][t].size(), _inputSize);
  }

  // Checking batch size for solution data
  if (_trainingBatchSize != _solutionData.size())
    KORALI_LOG_ERROR("Training Batch size of solution data (%lu) is different than that of input data (%lu).\n", _solutionData.size(), _inputData.size());

  // Checking that all solution batch entries have the correct size
  for (size_t b = 0; b < _solutionData.size(); b++)
    if (_solutionData[b].size() != _solutionSize)
      KORALI_LOG_ERROR("Solution vector size of batch %lu is inconsistent. Size: %lu - Expected: %lu.\n", b, _solutionData[b].size(), _solutionSize);
}

void SupervisedLearning::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Training Batch Size"))
 {
 try { _trainingBatchSize = js["Training Batch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Training Batch Size']\n%s", e.what()); } 
   eraseValue(js, "Training Batch Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Training Batch Size'] required by supervisedLearning.\n"); 

 if (isDefined(js, "Testing Batch Size"))
 {
 try { _testingBatchSize = js["Testing Batch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Testing Batch Size']\n%s", e.what()); } 
   eraseValue(js, "Testing Batch Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing Batch Size'] required by supervisedLearning.\n"); 

 if (isDefined(js, "Max Timesteps"))
 {
 try { _maxTimesteps = js["Max Timesteps"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Max Timesteps']\n%s", e.what()); } 
   eraseValue(js, "Max Timesteps");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Max Timesteps'] required by supervisedLearning.\n"); 

 if (isDefined(js, "Input", "Data"))
 {
 try { _inputData = js["Input"]["Data"].get<std::vector<std::vector<std::vector<float>>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Input']['Data']\n%s", e.what()); } 
   eraseValue(js, "Input", "Data");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Input']['Data'] required by supervisedLearning.\n"); 

 if (isDefined(js, "Input", "Size"))
 {
 try { _inputSize = js["Input"]["Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Input']['Size']\n%s", e.what()); } 
   eraseValue(js, "Input", "Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Input']['Size'] required by supervisedLearning.\n"); 

 if (isDefined(js, "Solution", "Data"))
 {
 try { _solutionData = js["Solution"]["Data"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Solution']['Data']\n%s", e.what()); } 
   eraseValue(js, "Solution", "Data");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Solution']['Data'] required by supervisedLearning.\n"); 

 if (isDefined(js, "Solution", "Size"))
 {
 try { _solutionSize = js["Solution"]["Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Solution']['Size']\n%s", e.what()); } 
   eraseValue(js, "Solution", "Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Solution']['Size'] required by supervisedLearning.\n"); 

  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("DeepSupervisor"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: supervisedLearning\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "supervisedLearning";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: supervisedLearning: \n%s\n", js.dump(2).c_str());
} 

void SupervisedLearning::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Training Batch Size"] = _trainingBatchSize;
   js["Testing Batch Size"] = _testingBatchSize;
   js["Max Timesteps"] = _maxTimesteps;
   js["Input"]["Data"] = _inputData;
   js["Input"]["Size"] = _inputSize;
   js["Solution"]["Data"] = _solutionData;
   js["Solution"]["Size"] = _solutionSize;
 Problem::getConfiguration(js);
} 

void SupervisedLearning::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Max Timesteps\": 1, \"Input\": {\"Data\": []}, \"Solution\": {\"Data\": []}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Problem::applyModuleDefaults(js);
} 

void SupervisedLearning::applyVariableDefaults() 
{

 Problem::applyVariableDefaults();
} 

;

} //problem
} //korali
;
