#include "modules/problem/design/design.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Design::initialize()
{
  // Processing state/action variable configuration
  for (size_t i = 0; i < _k->_variables.size(); i++)
  {
    if (_k->_variables[i]->_type == "Parameter")
      _parameterVectorIndexes.push_back(i);
    if (_k->_variables[i]->_type == "Design")
      _designVectorIndexes.push_back(i);
    if (_k->_variables[i]->_type == "Measurement")
      _measurementVectorIndexes.push_back(i);
  }

  _parameterVectorSize = _parameterVectorIndexes.size();
  _designVectorSize = _designVectorIndexes.size();
  _measurementVectorSize = _measurementVectorIndexes.size();

  if (_parameterVectorSize == 0) KORALI_LOG_ERROR("No parameter variables have been defined.\n");
  if (_designVectorSize == 0) KORALI_LOG_ERROR("No design variables have been defined.\n");
  if (_measurementVectorSize == 0) KORALI_LOG_ERROR("No measurement variables have been defined.\n");
}

void Design::runModel(Sample &sample)
{
  // Evaluating Sample
  sample.run(_model);
}

void Design::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Parameter Vector Size"))
 {
 try { _parameterVectorSize = js["Parameter Vector Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Parameter Vector Size']\n%s", e.what()); } 
   eraseValue(js, "Parameter Vector Size");
 }

 if (isDefined(js, "Design Vector Size"))
 {
 try { _designVectorSize = js["Design Vector Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Design Vector Size']\n%s", e.what()); } 
   eraseValue(js, "Design Vector Size");
 }

 if (isDefined(js, "Measurement Vector Size"))
 {
 try { _measurementVectorSize = js["Measurement Vector Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Measurement Vector Size']\n%s", e.what()); } 
   eraseValue(js, "Measurement Vector Size");
 }

 if (isDefined(js, "Parameter Vector Indexes"))
 {
 try { _parameterVectorIndexes = js["Parameter Vector Indexes"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Parameter Vector Indexes']\n%s", e.what()); } 
   eraseValue(js, "Parameter Vector Indexes");
 }

 if (isDefined(js, "Design Vector Indexes"))
 {
 try { _designVectorIndexes = js["Design Vector Indexes"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Design Vector Indexes']\n%s", e.what()); } 
   eraseValue(js, "Design Vector Indexes");
 }

 if (isDefined(js, "Measurement Vector Indexes"))
 {
 try { _measurementVectorIndexes = js["Measurement Vector Indexes"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Measurement Vector Indexes']\n%s", e.what()); } 
   eraseValue(js, "Measurement Vector Indexes");
 }

 if (isDefined(js, "Model"))
 {
 try { _model = js["Model"].get<std::uint64_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Model']\n%s", e.what()); } 
   eraseValue(js, "Model");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Model'] required by design.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Type"))
 {
 try { _k->_variables[i]->_type = _k->_js["Variables"][i]["Type"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Type']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_k->_variables[i]->_type == "Design") validOption = true; 
 if (_k->_variables[i]->_type == "Parameter") validOption = true; 
 if (_k->_variables[i]->_type == "Measurement") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Type'] required by design.\n", _k->_variables[i]->_type.c_str()); 
}
   eraseValue(_k->_js["Variables"][i], "Type");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Type'] required by design.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Lower Bound"))
 {
 try { _k->_variables[i]->_lowerBound = _k->_js["Variables"][i]["Lower Bound"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Lower Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Lower Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Lower Bound'] required by design.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Upper Bound"))
 {
 try { _k->_variables[i]->_upperBound = _k->_js["Variables"][i]["Upper Bound"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Upper Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Upper Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Upper Bound'] required by design.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Distribution"))
 {
 try { _k->_variables[i]->_distribution = _k->_js["Variables"][i]["Distribution"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Distribution']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Distribution'] required by design.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Number Of Samples"))
 {
 try { _k->_variables[i]->_numberOfSamples = _k->_js["Variables"][i]["Number Of Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ design ] \n + Key:    ['Number Of Samples']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Number Of Samples");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Number Of Samples'] required by design.\n"); 

 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Designer"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: design\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "design";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: design: \n%s\n", js.dump(2).c_str());
} 

void Design::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Model"] = _model;
   js["Parameter Vector Size"] = _parameterVectorSize;
   js["Design Vector Size"] = _designVectorSize;
   js["Measurement Vector Size"] = _measurementVectorSize;
   js["Parameter Vector Indexes"] = _parameterVectorIndexes;
   js["Design Vector Indexes"] = _designVectorIndexes;
   js["Measurement Vector Indexes"] = _measurementVectorIndexes;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Type"] = _k->_variables[i]->_type;
   _k->_js["Variables"][i]["Lower Bound"] = _k->_variables[i]->_lowerBound;
   _k->_js["Variables"][i]["Upper Bound"] = _k->_variables[i]->_upperBound;
   _k->_js["Variables"][i]["Distribution"] = _k->_variables[i]->_distribution;
   _k->_js["Variables"][i]["Number Of Samples"] = _k->_variables[i]->_numberOfSamples;
 } 
 Problem::getConfiguration(js);
} 

void Design::applyModuleDefaults(knlohmann::json& js) 
{

 Problem::applyModuleDefaults(js);
} 

void Design::applyVariableDefaults() 
{

 std::string defaultString = "{\"Lower Bound\": -Infinity, \"Upper Bound\": Infinity, \"Distribution\": \" \", \"Number Of Samples\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

bool Design::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Run Model")
 {
  runModel(sample);
  return true;
 }

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Design.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;
