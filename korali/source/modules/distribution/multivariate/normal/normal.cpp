#include "modules/distribution/multivariate/normal/normal.hpp"
#include "modules/experiment/experiment.hpp"
#include <auxiliar/logger.hpp>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>

namespace korali
{
namespace distribution
{
namespace multivariate
{
;

void Normal::getDensity(double *x, double *result, const size_t n)
{
  if (_workVector.size() != n)
    KORALI_LOG_ERROR("multivariate::Normal::getDensity Error - requested %lu densities, but distribution is configured with %lu.\n", n, _workVector.size());

  gsl_vector_view _input_view = gsl_vector_view_array(x, n);

  gsl_ran_multivariate_gaussian_pdf(&_input_view.vector, &_mean_view.vector, &_sigma_view.matrix, result, &_work_view.vector);
}

void Normal::getLogDensity(double *x, double *result, const size_t n)
{
  if (_workVector.size() != n)
    KORALI_LOG_ERROR("multivariate::Normal::getLogDensity Error - requested %lu densities, but distribution is configured with %lu.\n", n, _workVector.size());

  gsl_vector_view _input_view = gsl_vector_view_array(x, n);

  gsl_ran_multivariate_gaussian_log_pdf(&_input_view.vector, &_mean_view.vector, &_sigma_view.matrix, result, &_work_view.vector);
}

void Normal::getRandomVector(double *x, const size_t n)
{
  gsl_vector_view _output_view = gsl_vector_view_array(x, n);

  gsl_ran_multivariate_gaussian(_range, &_mean_view.vector, &_sigma_view.matrix, &_output_view.vector);
}

void Normal::updateDistribution()
{
  size_t covarianceMatrixSize = _sigma.size();

  size_t sideSize = sqrt(covarianceMatrixSize);
  if ((sideSize * sideSize) != covarianceMatrixSize)
    KORALI_LOG_ERROR("Size of Multivariate Normal covariance matrix size (%lu) is not a perfect square number.\n", covarianceMatrixSize);

  size_t meanSize = _meanVector.size();
  if (sideSize != meanSize) KORALI_LOG_ERROR("Size of Multivariate Normal mean vector (%lu) is not the same as the side of covariance matrix (%lu).\n", meanSize, sideSize);

  _workVector.resize(meanSize);

  _sigma_view = gsl_matrix_view_array(_sigma.data(), sideSize, sideSize);
  _mean_view = gsl_vector_view_array(_meanVector.data(), meanSize);
  _work_view = gsl_vector_view_array(_workVector.data(), meanSize);
}

void Normal::setProperty(const std::string &propertyName, const std::vector<double> &values)
{
  bool recognizedProperty = false;
  if (propertyName == "Mean Vector")
  {
    _meanVector = values;
    recognizedProperty = true;
  }
  if (propertyName == "Sigma")
  {
    _sigma = values;
    recognizedProperty = true;
  }
  if (recognizedProperty == false) KORALI_LOG_ERROR("Unrecognized property: %s for the Multivariate Normal distribution", propertyName.c_str());
}

void Normal::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Work Vector"))
 {
 try { _workVector = js["Work Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ normal ] \n + Key:    ['Work Vector']\n%s", e.what()); } 
   eraseValue(js, "Work Vector");
 }

 if (isDefined(js, "Mean Vector"))
 {
 try { _meanVector = js["Mean Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ normal ] \n + Key:    ['Mean Vector']\n%s", e.what()); } 
   eraseValue(js, "Mean Vector");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mean Vector'] required by normal.\n"); 

 if (isDefined(js, "Sigma"))
 {
 try { _sigma = js["Sigma"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ normal ] \n + Key:    ['Sigma']\n%s", e.what()); } 
   eraseValue(js, "Sigma");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Sigma'] required by normal.\n"); 

 Multivariate::setConfiguration(js);
 _type = "multivariate/normal";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: normal: \n%s\n", js.dump(2).c_str());
} 

void Normal::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Mean Vector"] = _meanVector;
   js["Sigma"] = _sigma;
   js["Work Vector"] = _workVector;
 Multivariate::getConfiguration(js);
} 

void Normal::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Mean Vector\": [], \"Sigma\": []}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Multivariate::applyModuleDefaults(js);
} 

void Normal::applyVariableDefaults() 
{

 Multivariate::applyVariableDefaults();
} 

;

} //multivariate
} //distribution
} //korali
;
