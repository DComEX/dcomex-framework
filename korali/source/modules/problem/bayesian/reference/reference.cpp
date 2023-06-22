#include "modules/conduit/conduit.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/bayesian/reference/reference.hpp"
#include "sample/sample.hpp"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>

#define STDEV_EPSILON 0.00000000001

namespace korali
{
namespace problem
{
namespace bayesian
{
;

void Reference::initialize()
{
  Bayesian::initialize();

  if (_referenceData.size() == 0) KORALI_LOG_ERROR("Bayesian (%s) problems require defining reference data.\n", _likelihoodModel.c_str());
  if (_k->_variables.size() < 1) KORALI_LOG_ERROR("Bayesian (%s) inference problems require at least one variable.\n", _likelihoodModel.c_str());
}

void Reference::evaluateLoglikelihood(Sample &sample)
{
  sample.run(_computationalModel);
  if (_likelihoodModel == "Normal")
    loglikelihoodNormal(sample);
  else if (_likelihoodModel == "Positive Normal")
    loglikelihoodPositiveNormal(sample);
  else if (_likelihoodModel == "StudentT")
    loglikelihoodStudentT(sample);
  else if (_likelihoodModel == "Positive StudentT")
    loglikelihoodPositiveStudentT(sample);
  else if (_likelihoodModel == "Poisson")
    loglikelihoodPoisson(sample);
  else if (_likelihoodModel == "Geometric")
    loglikelihoodGeometric(sample);
  else if (_likelihoodModel == "Negative Binomial")
    loglikelihoodNegativeBinomial(sample);
  else
    KORALI_LOG_ERROR("Bayesian problem (%s) not recognized.\n", _likelihoodModel.c_str());
}

double Reference::compute_normalized_sse(std::vector<double> f, std::vector<double> g, std::vector<double> y)
{
  double sse = 0.;
  for (size_t i = 0; i < y.size(); i++)
  {
    double diff = (y[i] - f[i]) / g[i];
    sse += diff * diff;
  }
  return sse;
}

void Reference::loglikelihoodNormal(Sample &sample)
{
  size_t Nd = _referenceData.size();
  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto stdDevs = KORALI_GET(std::vector<double>, sample, "Standard Deviation");

  if (stdDevs.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Standard Deviation array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, stdDevs.size());
  if (refEvals.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Reference Evaluations array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, refEvals.size());

  double sse = -Inf;
  sse = compute_normalized_sse(refEvals, stdDevs, _referenceData);

  double loglike = 0.;
  for (size_t i = 0; i < stdDevs.size(); i++)
  {
    if (stdDevs[i] < 0.0) KORALI_LOG_ERROR("Negative (%lf) detected for the Standard Deviation.\n", stdDevs[i]);
    if (stdDevs[i] < STDEV_EPSILON) stdDevs[i] = STDEV_EPSILON; // Adding epsilon for numerical stability
    loglike -= log(stdDevs[i]);
  }

  loglike -= 0.5 * (Nd * _log2pi + sse);
  sample["logLikelihood"] = loglike;
}

void Reference::loglikelihoodPositiveNormal(Sample &sample)
{
  size_t Nd = _referenceData.size();
  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto stdDevs = KORALI_GET(std::vector<double>, sample, "Standard Deviation");

  if (stdDevs.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Standard Deviation array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, stdDevs.size());
  if (refEvals.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Reference Evaluations array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, refEvals.size());

  double loglike = 0.;
  for (size_t i = 0; i < stdDevs.size(); i++)
  {
    double m = refEvals[i];
    double s = stdDevs[i];

    if (s <= 0.0) KORALI_LOG_ERROR("Negative or zero value (%lf) detected for the Standard Deviation.\n", s);
    if (m < 0.0) KORALI_LOG_ERROR("Negative value (%lf) detected in Reference Evaluation.\n", m);
    if (_referenceData[i] < 0.0) KORALI_LOG_ERROR("Negative value (%lf) detected in Reference Data.\n", _referenceData[i]);

    double z = (_referenceData[i] - m) / s;

    loglike -= 0.5 * (_log2pi + z * z);
    loglike -= log(s);
    loglike -= log(1. - gsl_cdf_gaussian_P(-m / s, 1.0));
  }

  sample["logLikelihood"] = loglike;
}

void Reference::loglikelihoodStudentT(Sample &sample)
{
  size_t Nd = _referenceData.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto nus = KORALI_GET(std::vector<double>, sample, "Degrees Of Freedom");

  if (nus.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Degrees Of Freedom array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, nus.size());
  if (refEvals.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Reference Evaluations array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, refEvals.size());

  double loglike = 0.;
  for (size_t i = 0; i < nus.size(); i++)
  {
    if (nus[i] <= 0.0) KORALI_LOG_ERROR("Negative or zero value (%lf) detected for the Degrees Of Freedom.\n", nus[i]);
    loglike += log(gsl_ran_tdist_pdf(_referenceData[i] - refEvals[i], nus[i]));
  }

  sample["logLikelihood"] = loglike;
}

void Reference::loglikelihoodPositiveStudentT(Sample &sample)
{
  size_t Nd = _referenceData.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto nus = KORALI_GET(std::vector<double>, sample, "Degrees Of Freedom");

  if (nus.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Degrees Of Freedom array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, nus.size());
  if (refEvals.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Reference Evaluations array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, refEvals.size());

  double loglike = 0.;
  for (size_t i = 0; i < Nd; i++)
  {
    if (nus[i] <= 0.0) KORALI_LOG_ERROR("Negative or zero value (%lf) detected for the Degrees Of Freedom.\n", nus[i]);
    if (refEvals[i] < 0.0) KORALI_LOG_ERROR("Negative value (%lf) detected in Reference Evaluation.\n", refEvals[i]);
    if (_referenceData[i] < 0.0) KORALI_LOG_ERROR("Negative value (%lf) detected in Reference Data.\n", _referenceData[i]);

    loglike += log(gsl_ran_tdist_pdf(_referenceData[i] - refEvals[i], nus[i]));
    loglike -= log(1.0 - gsl_cdf_tdist_P(-refEvals[i], nus[i]));
  }

  sample["logLikelihood"] = loglike;
}

void Reference::loglikelihoodPoisson(Sample &sample)
{
  size_t Nd = _referenceData.size();
  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");

  if (refEvals.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Reference Evaluations array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, refEvals.size());

  double loglike = 0.;
  for (size_t i = 0; i < Nd; i++)
  {
    if (refEvals[i] <= 0.0) KORALI_LOG_ERROR("Negative value (%lf) detected in Reference Evaluation.\n", refEvals[i]);
    if (_referenceData[i] < 0.0) KORALI_LOG_ERROR("Negative value (%lf) detected in Reference Data.\n", _referenceData[i]);
    loglike += log(gsl_ran_poisson_pdf(_referenceData[i], refEvals[i]));
  }

  sample["logLikelihood"] = loglike;
}

void Reference::loglikelihoodGeometric(Sample &sample)
{
  size_t Nd = _referenceData.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");

  if (refEvals.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Reference Evaluations array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, refEvals.size());

  double loglike = 0.;
  for (size_t i = 0; i < Nd; i++)
  {
    if (refEvals[i] < 0.0) KORALI_LOG_ERROR("Negative value (%lf) detected in Reference Evaluation.\n", refEvals[i]);
    if (_referenceData[i] < 0.0) KORALI_LOG_ERROR("Negative value (%lf) detected in Reference Data.\n", _referenceData[i]);
    loglike += log(gsl_ran_geometric_pdf(_referenceData[i] + 1.0, 1.0 / (1.0 + refEvals[i])));
  }

  sample["logLikelihood"] = loglike;
}

void Reference::loglikelihoodNegativeBinomial(Sample &sample)
{
  size_t Nd = _referenceData.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto dispersion = KORALI_GET(std::vector<double>, sample, "Dispersion");

  if (refEvals.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Reference Evaluations array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, refEvals.size());
  if (dispersion.size() != Nd) KORALI_LOG_ERROR("This Bayesian (%s) problem requires a %lu-sized Dispersion array. Provided: %lu.\n", _likelihoodModel.c_str(), Nd, dispersion.size());

  double loglike = 0.0;

  for (size_t i = 0; i < Nd; i++)
  {
    double y = _referenceData[i];
    if (y < 0) KORALI_LOG_ERROR("Negative Binomial Likelihood not defined for negative Reference Data (provided %lf.\n", y);

    loglike -= gsl_sf_lngamma(y + 1.);

    double m = refEvals[i];

    if (m <= 0)
    {
      sample["logLikelihood"] = -Inf;
      return;
    }

    double r = dispersion[i];

    double p = m / (m + r);

    loglike += gsl_sf_lngamma(y + r);
    loglike -= gsl_sf_lngamma(r);
    loglike += r * log(1 - p);
    loglike += y * log(p);
  }

  sample["logLikelihood"] = loglike;
}

void Reference::evaluateLoglikelihoodGradient(Sample &sample)
{
  double eval = sample["F(x)"];
  if (isfinite(eval))
  {
    if (_likelihoodModel == "Normal")
      gradientLoglikelihoodNormal(sample);
    else if (_likelihoodModel == "Positive Normal")
      gradientLoglikelihoodPositiveNormal(sample);
    else if (_likelihoodModel == "Negative Binomial")
      gradientLoglikelihoodNegativeBinomial(sample);
    else
      KORALI_LOG_ERROR("Gradient not yet implemented for logLikelihood model of type '%s'.", _likelihoodModel.c_str());
  }
  else
  {
    sample["logLikelihood Gradient"] = std::vector<double>(_k->_variables.size(), 0.0);
  }
}

void Reference::gradientLoglikelihoodNormal(Sample &sample)
{
  size_t Nd = _referenceData.size();
  size_t Nth = _k->_variables.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto stdDevs = KORALI_GET(std::vector<double>, sample, "Standard Deviation");
  auto gradientF = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Mean");
  auto gradientG = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Standard Deviation");

  if (gradientF.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);
  if (gradientG.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Standard Deviation for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);

  std::vector<double> llkgradient(Nth, 0.0);
  for (size_t i = 0; i < Nd; ++i)
  {
    if (gradientF[i].size() != Nth) KORALI_LOG_ERROR("Bayesian Reference Mean gradient calculation requires gradients of size %zu (provided size %zu)\n", Nth, gradientF[i].size());
    if (gradientG[i].size() != Nth) KORALI_LOG_ERROR("Bayesian Reference Standard Deviation gradient calculation requires gradients of size %zu (provided size %zu)\n", Nth, gradientG[i].size());

    double invStdDev = 1.0 / stdDevs[i];
    double invStdDev2 = invStdDev * invStdDev;
    double invStdDev3 = invStdDev2 * invStdDev;

    double dif = _referenceData[i] - refEvals[i];

    for (size_t d = 0; d < Nth; ++d)
    {
      if (!isfinite(gradientF[i][d])) _k->_logger->logWarning("Normal", "Non-finite value detected in Gradient Mean.\n");
      if (!isfinite(gradientG[i][d])) _k->_logger->logWarning("Normal", "Non-finite value detected in Gradient Standard Deviation.\n");
      double tmpGrad = -invStdDev * gradientG[i][d] + invStdDev2 * dif * gradientF[i][d] + invStdDev3 * dif * dif * gradientG[i][d];
      llkgradient[d] += tmpGrad;
    }
  }

  sample["logLikelihood Gradient"] = llkgradient;
}

void Reference::gradientLoglikelihoodPositiveNormal(Sample &sample)
{
  size_t Nd = _referenceData.size();
  size_t Nth = _k->_variables.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto stdDevs = KORALI_GET(std::vector<double>, sample, "Standard Deviation");
  auto gradientF = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Mean");
  auto gradientG = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Standard Deviation");

  if (gradientF.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);
  if (gradientG.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Standard Deviation for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);

  std::vector<double> llkgradient(Nth, 0.0);
  for (size_t i = 0; i < Nd; ++i)
  {
    if (gradientF[i].size() != Nth) KORALI_LOG_ERROR("Bayesian Reference Mean gradient calculation requires gradients of size %zu (provided size %zu)\n", Nth, gradientF[i].size());
    if (gradientG[i].size() != Nth) KORALI_LOG_ERROR("Bayesian Reference Standard Deviation gradient calculation requires gradients of size %zu (provided size %zu)\n", Nth, gradientG[i].size());

    double mu = refEvals[i];
    double sig = stdDevs[i];

    double invsig = 1.0 / sig;
    double invsig2 = invsig * invsig;
    double invsig3 = invsig2 * invsig;

    double Z = 1.0 - gsl_cdf_gaussian_P(-mu / sig, 1.0);
    double invZ = 1.0 / Z;

    double phims = gsl_ran_gaussian_pdf(-mu / sig, 1.0);

    double dif = _referenceData[i] - refEvals[i];

    for (size_t d = 0; d < Nth; ++d)
    {
      if (!isfinite(gradientF[i][d])) _k->_logger->logWarning("Normal", "Non-finite value detected in Gradient Mean.\n");
      if (!isfinite(gradientG[i][d])) _k->_logger->logWarning("Normal", "Non-finite value detected in Gradient Standard Deviation.\n");
      llkgradient[d] += (-invsig * gradientG[i][d] + invsig2 * dif * gradientF[i][d] + invsig3 * dif * dif * gradientG[i][d]);
      llkgradient[d] += invZ * phims * (-1.0 * invsig * gradientF[i][d] + invsig2 * mu * gradientG[i][d]);
    }
  }

  sample["logLikelihood Gradient"] = llkgradient;
}

void Reference::gradientLoglikelihoodNegativeBinomial(Sample &sample)
{
  size_t Nd = _referenceData.size();
  size_t Nth = _k->_variables.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto dispersion = KORALI_GET(std::vector<double>, sample, "Dispersion");
  auto gradientMean = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Mean");
  auto gradientDispersion = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Dispersion");

  if (gradientMean.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean for each reference evaluation (provided %zu required %zu).", gradientMean.size(), Nd);
  if (gradientDispersion.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Dispersion for each reference evaluation (provided %zu required %zu).", gradientDispersion.size(), Nd);

  std::vector<double> llkgradient(Nth, 0.0);

  double r, m, k, tmpsum;
  for (size_t i = 0; i < Nd; i++)
  {
    r = dispersion[i];
    m = refEvals[i];
    k = _referenceData[i];

    tmpsum = r + m;

    for (size_t d = 0; d < Nth; ++d)
    {
      llkgradient[d] += r * (k - m) * gradientMean[i][d] / (m * tmpsum);
      llkgradient[d] += (gsl_sf_psi(r + k) - gsl_sf_psi(r)) * gradientDispersion[i][d];
      llkgradient[d] += gradientDispersion[i][d] * (m * log(r / tmpsum) + r * log(r / tmpsum) + m - k) / tmpsum;
    }
  }

  sample["logLikelihood Gradient"] = llkgradient;
}

void Reference::evaluateLogLikelihoodHessian(Sample &sample)
{
  double eval = sample["F(x)"];
  if (isfinite(eval))
  {
    if (_likelihoodModel == "Normal")
      hessianLogLikelihoodNormal(sample);
    else if (_likelihoodModel == "Positive Normal")
      hessianLogLikelihoodPositiveNormal(sample);
    else if (_likelihoodModel == "Negative Binomial")
      hessianLogLikelihoodNegativeBinomial(sample);
    else
      KORALI_LOG_ERROR("Hessian not yet implemented for logLikelihood model of type '%s'.", _likelihoodModel.c_str());
  }
  else
  {
    sample["logLikelihood Hessian"] = std::vector<double>(_k->_variables.size() * _k->_variables.size(), 0.0);
  }
}

void Reference::hessianLogLikelihoodNormal(korali::Sample &sample)
{
  size_t Nd = _referenceData.size();
  size_t Nth = _k->_variables.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto stdDevs = KORALI_GET(std::vector<double>, sample, "Standard Deviation");

  auto gradientF = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Mean");
  auto gradientG = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Standard Deviation");

  auto hessianF = KORALI_GET(std::vector<std::vector<double>>, sample, "Hessian Mean");
  auto hessianG = KORALI_GET(std::vector<std::vector<double>>, sample, "Hessian Standard Deviation");

  if (gradientF.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);
  if (gradientG.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Standard Deviation for each reference evaluation (provided %zu required %zu).", gradientG.size(), Nd);
  if (hessianF.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a Hessian of the Mean for each reference evaluation (provided %zu required %zu).", hessianF.size(), Nd);
  if (hessianG.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a Hessian of the Standard Deviation for each reference evaluation (provided %zu required %zu).", hessianG.size(), Nd);

  std::vector<double> hessian(Nth * Nth, 0.0);

  for (size_t i = 0; i < Nd; ++i)
  {
    if (gradientF[i].size() != Nth) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean of size %zu for each reference evaluation (provided %zu).", Nth, gradientF[i].size());
    if (gradientG[i].size() != Nth) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Standard Deviation of size %zu for each reference evaluation (provided %zu).", Nth, gradientG[i].size());
    if (hessianF[i].size() != Nth * Nth) KORALI_LOG_ERROR("Bayesian problem requires a Hessian of the Mean of size %zu for each reference evaluation (provided %zu).", Nth * Nth, hessianF[i].size());
    if (hessianG[i].size() != Nth * Nth) KORALI_LOG_ERROR("Bayesian problem requires a Hessian of the Standard Deviation of size %zu for each reference evaluation (provided %zu).", Nth * Nth, hessianG[i].size());

    double var = stdDevs[i] * stdDevs[i];
    double stdevinv = 1. / stdDevs[i];
    double varinv = 1. / var;
    double stdev3inv = varinv * stdevinv;
    double var2inv = varinv * varinv;

    double diff = refEvals[i] - _referenceData[i];

    double tmp;
    for (size_t k = 0; k < Nth; ++k)
    {
      for (size_t l = 0; l <= k; ++l)
      {
        tmp = diff * diff * (stdev3inv * hessianG[i][k * Nth + l] - 3 * var2inv * gradientG[i][k] * gradientG[i][l]) - varinv * (gradientF[i][k] * gradientF[i][l] + diff * hessianF[i][k * Nth + l]) - 2 * stdev3inv * diff * (gradientF[i][k] * gradientG[i][l] + gradientF[i][l] * gradientG[i][k]) + varinv * gradientG[i][k] * gradientG[i][l] - stdevinv * hessianG[i][k * Nth + l];
        hessian[k * Nth + l] += tmp;
        if (l < k) hessian[l * Nth + k] += tmp;
      }
    }
  }
  sample["logLikelihood Hessian"] = hessian;
}

void Reference::hessianLogLikelihoodPositiveNormal(korali::Sample &sample)
{
  KORALI_LOG_ERROR("Hessian not yet implemented for Positive Normal logLikelihood model.");
}

void Reference::hessianLogLikelihoodNegativeBinomial(korali::Sample &sample)
{
  size_t Nd = _referenceData.size();
  size_t Nth = _k->_variables.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto dispersions = KORALI_GET(std::vector<double>, sample, "Dispersion");

  auto gradientM = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Mean");
  auto gradientR = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Dispersion");

  auto hessianM = KORALI_GET(std::vector<std::vector<double>>, sample, "Hessian Mean");
  auto hessianR = KORALI_GET(std::vector<std::vector<double>>, sample, "Hessian Dispersion");

  if (gradientM.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean for each reference evaluation (provided %zu required %zu).", gradientM.size(), Nd);
  if (gradientR.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Dispersion for each reference evaluation (provided %zu required %zu).", gradientM.size(), Nd);
  if (hessianM.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a Hessian of the Mean for each reference evaluation (provided %zu required %zu).", gradientM.size(), Nd);
  if (hessianR.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a Hessian of the Dispersion for each reference evaluation (provided %zu required %zu).", gradientM.size(), Nd);

  std::vector<double> hessian(Nth * Nth, 0.0);
  for (size_t i = 0; i < Nd; ++i)
  {
    if (gradientM[i].size() != Nth) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean of size %zu for each reference evaluation (provided %zu).", Nth, gradientM[i].size());
    if (gradientR[i].size() != Nth) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Standard Deviation of size %zu for each reference evaluation (provided %zu).", Nth, gradientR[i].size());
    if (hessianM[i].size() != Nth * Nth) KORALI_LOG_ERROR("Bayesian problem requires a Hessian of the Mean of size %zu for each reference evaluation (provided %zu).", Nth * Nth, hessianM[i].size());
    if (hessianR[i].size() != Nth * Nth) KORALI_LOG_ERROR("Bayesian problem requires a Hessian of the Standard Deviation of size %zu for each reference evaluation (provided %zu).", Nth * Nth, hessianR[i].size());

    double d = _referenceData[i];

    double m = refEvals[i];
    double r = dispersions[i];
    double dr = d + r;
    double mr = m + r;

    double m2 = m * m;
    double m3 = m2 * m;
    double r2 = r * r;
    double r3 = r2 * r;

    double C1 = 1.0 / (r * (mr * mr));
    double C2 = d / (m * m * (mr * mr));

    double logC = log(r / (m + r));

    for (size_t k = 0; k < Nth; ++k)
    {
      for (size_t l = 0; l <= k; ++l)
      {
        double Hm = hessianM[i][k * Nth + l];
        double Hr = hessianR[i][k * Nth + l];

        double Gmk = gradientM[i][k];
        double Gml = gradientM[i][l];
        double Gm2 = Gmk * Gml;

        double Grk = gradientR[i][k];
        double Grl = gradientR[i][l];
        double Gr2 = Grk * Grl;

        double tmp1 = Hr * (gsl_sf_psi(dr) - gsl_sf_psi(r)) + Grk * Grl * (gsl_sf_psi_1(dr) - gsl_sf_psi_1(r));
        double tmp2 = C1 * (-Hm * r3 - m * r2 * Hm - m * r * (Gmk * Grl + Gml * Grk) + r2 * Gm2 + m * r2 * Hr + m2 * r * Hr + r3 * Hr * logC + 2 * m * r2 * Hr * logC + m2 * r * Hr * logC + m * Gr2);
        double tmp3 = C2 * (-r2 * Gm2 + m2 * (r * (Hm - Hr) + Gmk * Grl + Gml * Grk) + m * r * (r * Hm - 2 * Gm2) - m3 * Hr);

        hessian[k * Nth + l] += (tmp1 + tmp2 + tmp3);
        if (l < k) hessian[l * Nth + k] += (tmp1 + tmp2 + tmp3);
      }
    }
  }
  sample["logLikelihood Hessian"] = hessian;
}

void Reference::evaluateFisherInformation(Sample &sample)
{
  auto eval = KORALI_GET(double, sample, "F(x)");

  if (isfinite(eval))
  {
    if (_likelihoodModel == "Normal")
      fisherInformationLoglikelihoodNormal(sample);
    else if (_likelihoodModel == "Positive Normal")
      fisherInformationLoglikelihoodPositiveNormal(sample);
    else if (_likelihoodModel == "Negative Binomial")
      fisherInformationLoglikelihoodNegativeBinomial(sample);
    else
      KORALI_LOG_ERROR("Fisher Information not yet implemented for logLikelihood model of type '%s'.", _likelihoodModel.c_str());
  }
  else
  {
    sample["Fisher Information"] = std::vector<double>(_k->_variables.size() * _k->_variables.size(), 0.0);
  }
}

void Reference::fisherInformationLoglikelihoodNormal(Sample &sample)
{
  size_t Nd = _referenceData.size();
  size_t Nth = _k->_variables.size();

  auto stdDevs = KORALI_GET(std::vector<double>, sample, "Standard Deviation");
  auto gradientF = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Mean");
  auto gradientG = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Standard Deviation");

  if (gradientF.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);
  if (gradientG.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Standard Deviation for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);

  std::vector<double> FIM(Nth * Nth, 0.0);
  for (size_t i = 0; i < Nd; ++i)
  {
    double var = stdDevs[i] * stdDevs[i];
    double varinv = 1. / var;

    double tmp;
    for (size_t k = 0; k < Nth; ++k)
    {
      for (size_t l = 0; l < k; ++l)
      {
        tmp = varinv * gradientF[i][k] * gradientF[i][l] + 2. * varinv * gradientG[i][k] * gradientG[i][l];
        FIM[k * Nth + l] += tmp;
        FIM[l * Nth + k] += tmp;
      }
      FIM[k * Nth + k] += (varinv * gradientF[i][k] * gradientF[i][k] + 2. * varinv * gradientG[i][k] * gradientG[i][k]);
    }
  }
  sample["Fisher Information"] = FIM;
}

void Reference::fisherInformationLoglikelihoodPositiveNormal(Sample &sample)
{
  size_t Nd = _referenceData.size();
  size_t Nth = _k->_variables.size();

  auto refEvals = KORALI_GET(std::vector<double>, sample, "Reference Evaluations");
  auto stdDevs = KORALI_GET(std::vector<double>, sample, "Standard Deviation");
  auto gradientF = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Mean");
  auto gradientG = KORALI_GET(std::vector<std::vector<double>>, sample, "Gradient Standard Deviation");

  if (gradientF.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Mean for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);
  if (gradientG.size() != Nd) KORALI_LOG_ERROR("Bayesian problem requires a gradient of the Standard Deviation for each reference evaluation (provided %zu required %zu).", gradientF.size(), Nd);

  std::vector<double> FIM(Nth * Nth, 0.0);
  for (size_t i = 0; i < Nd; ++i)
  {
    double mu = refEvals[i];
    double sig = stdDevs[i];
    double var = sig * sig;

    double phims = gsl_ran_ugaussian_pdf(mu / sig);
    double phims2 = phims * phims;

    double Z = 1.0 - gsl_cdf_ugaussian_P(-mu / sig);
    double invZ = 1.0 / Z;
    double invZ2 = invZ * invZ;

    double invvar = 1. / var;
    double invsig3 = invvar / sig;
    double invsig4 = invvar * invvar;
    double invsig5 = invvar * invsig3;

    double Imu = invvar - invZ2 * invvar * phims2 - invZ * mu * invsig3 * phims;
    double Isig = 2. * invvar - 5. * invZ * mu * invsig3 * phims - invZ2 * mu * mu * invsig4 * phims2 - invZ * mu * mu * mu * invsig5 * phims;
    double Ims = invZ * (var + mu * mu) * invsig4 * phims + invZ2 * mu * invsig3 * phims2;

    double tmp;
    for (size_t k = 0; k < Nth; ++k)
    {
      for (size_t l = 0; l < k; ++l)
      {
        tmp = gradientF[i][k] * gradientF[i][l] * Imu + (gradientF[i][k] * 2 * sig * gradientG[i][l] + gradientF[i][l] * 2 * sig * gradientG[i][k]) * Ims + 4 * var * gradientG[i][k] * gradientG[i][l] * Isig;
        FIM[k * Nth + l] += tmp;
        FIM[l * Nth + k] += tmp;
      }
      FIM[k * Nth + k] += (gradientF[i][k] * gradientF[i][k] * Imu + (4 * sig * gradientF[i][k] * gradientG[i][k]) * Ims + 4 * var * gradientG[i][k] * gradientG[i][k] * Isig);
    }
  }
  sample["Fisher Information"] = FIM;
}

void Reference::fisherInformationLoglikelihoodNegativeBinomial(Sample &sample)
{
  KORALI_LOG_ERROR("Fisher Information not yet implemented for logLikelihood model of type '%s'.", _likelihoodModel.c_str());
}

void Reference::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Computational Model"))
 {
 try { _computationalModel = js["Computational Model"].get<std::uint64_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reference ] \n + Key:    ['Computational Model']\n%s", e.what()); } 
   eraseValue(js, "Computational Model");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Computational Model'] required by reference.\n"); 

 if (isDefined(js, "Reference Data"))
 {
 try { _referenceData = js["Reference Data"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reference ] \n + Key:    ['Reference Data']\n%s", e.what()); } 
   eraseValue(js, "Reference Data");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reference Data'] required by reference.\n"); 

 if (isDefined(js, "Likelihood Model"))
 {
 try { _likelihoodModel = js["Likelihood Model"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reference ] \n + Key:    ['Likelihood Model']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_likelihoodModel == "Normal") validOption = true; 
 if (_likelihoodModel == "Positive Normal") validOption = true; 
 if (_likelihoodModel == "StudentT") validOption = true; 
 if (_likelihoodModel == "Positive StudentT") validOption = true; 
 if (_likelihoodModel == "Poisson") validOption = true; 
 if (_likelihoodModel == "Geometric") validOption = true; 
 if (_likelihoodModel == "Negative Binomial") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Likelihood Model'] required by reference.\n", _likelihoodModel.c_str()); 
}
   eraseValue(js, "Likelihood Model");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Likelihood Model'] required by reference.\n"); 

 Bayesian::setConfiguration(js);
 _type = "bayesian/reference";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: reference: \n%s\n", js.dump(2).c_str());
} 

void Reference::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Computational Model"] = _computationalModel;
   js["Reference Data"] = _referenceData;
   js["Likelihood Model"] = _likelihoodModel;
 Bayesian::getConfiguration(js);
} 

void Reference::applyModuleDefaults(knlohmann::json& js) 
{

 Bayesian::applyModuleDefaults(js);
} 

void Reference::applyVariableDefaults() 
{

 Bayesian::applyVariableDefaults();
} 

;

} //bayesian
} //problem
} //korali
;
