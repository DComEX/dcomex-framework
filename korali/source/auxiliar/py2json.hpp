/***************************************************************************
* Copyright (c) 2019, Martin Renou                                         *
* Modified by Sergio Martin to support Function storing                    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
****************************************************************************/

/** \file
 * @brief Functions to support direct conversion of Python/C++ objects to JSON and vice versa
 *********************************************************************************************/

#pragma once


#include "auxiliar/json.hpp"
#include "modules/experiment/experiment.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <gsl/gsl_rng.h>
#include <string>
#include <vector>

namespace korali
{
class Sample;

/**
 * @brief Stores all functions inserted as parameters to experiment's configuration
 */
extern std::vector<std::function<void(Sample &)> *> _functionVector;
} // namespace korali

/*! \namespace knlohmann
  \brief The knlohmann namespace includes all Korali-Json auxiliar functions and class methods.
*/
namespace knlohmann
{
/**
 * @brief Struct containing serializer/deserializer object for Pybind11 and JSON objects.
 */
template <>
struct adl_serializer<pybind11::object>
{
  static pybind11::object from_json(const json &j);
  static void to_json(json &j, const pybind11::object &obj);
};

/**
 * @brief Serializes Korali model functions into a JSON-acceptable number containing its pointer.
 * @param j The JSON object to write.
 * @param obj The Korali function to serialize
 */
inline void adl_serializer<std::function<void(korali::Sample &)>>::to_json(json &j, const std::function<void(korali::Sample &)> &obj)
{
  auto x = new std::function<void(korali::Sample &)>(obj);
  j = korali::_functionVector.size();
  korali::_functionVector.push_back(x);
}

/**
    * @brief Struct containing serializer/deserializer object for korali::Experiment and JSON objects.
    */
template <>
struct adl_serializer<korali::Experiment>
{
  static void to_json(json &j, const korali::Experiment &obj);
};

/**
    * @brief Serializes a korali::Experiment into a JSON-acceptable number containing its pointer.
    * @param j The JSON object to write.
    * @param obj The korali::Experiment  to serialize
    */
inline void adl_serializer<korali::Experiment>::to_json(json &j, const korali::Experiment &obj)
{
  obj._js.getCopy(j);
}

/*! \namespace detail
  \brief Implementations details for the json serialization objects
*/
namespace detail
{
/**
 * @brief Deserializes JSON objects to Pybind11
 * @param j The JSON object to deserialize.
 * @return the Pybind11 object to create.
 */
inline pybind11::object from_json_impl(const json &j)
{
  if (j.is_null())
  {
    return pybind11::none();
  }
  else if (j.is_boolean())
  {
    return pybind11::bool_(j.get<bool>());
  }
  else if (j.is_number())
  {
    double number = j.get<double>();
    if (number == std::floor(number))
    {
      return pybind11::int_(j.get<long>());
    }
    else
    {
      return pybind11::float_(number);
    }
  }
  else if (j.is_string())
  {
    return pybind11::str(j.get<std::string>());
  }
  else if (j.is_array())
  {
    pybind11::list obj;
    for (const auto &el : j)
    {
      obj.attr("append")(from_json_impl(el));
    }
    return std::move(obj);
  }
  else // Object
  {
    pybind11::dict obj;
    for (json::const_iterator it = j.cbegin(); it != j.cend(); ++it)
    {
      obj[pybind11::str(it.key())] = from_json_impl(it.value());
    }
    return std::move(obj);
  }
}

/**
 * @brief Serializes Pybind11 objects to JSON objects
 * @param j The Pybind11 object to serialize.
 * @return The serialized JSON object.
 */
inline json to_json_impl(const pybind11::handle &obj)
{
  if (obj.is_none())
  {
    return nullptr;
  }
  if (pybind11::isinstance<pybind11::function>(obj))
  {
    auto x = new std::function<void(korali::Sample &)>(obj.cast<std::function<void(korali::Sample &)>>());
    auto j = korali::_functionVector.size();
    korali::_functionVector.push_back(x);
    return j;
  }
  if (pybind11::isinstance<pybind11::bool_>(obj))
  {
    return obj.cast<bool>();
  }
  if (pybind11::isinstance<pybind11::int_>(obj))
  {
    return obj.cast<long>();
  }
  if (pybind11::isinstance<pybind11::float_>(obj))
  {
    return obj.cast<double>();
  }
  if (pybind11::isinstance<pybind11::str>(obj))
  {
    return obj.cast<std::string>();
  }
  if (pybind11::isinstance<pybind11::tuple>(obj) || pybind11::isinstance<pybind11::list>(obj))
  {
    auto out = json::array();
    for (const pybind11::handle &value : obj)
    {
      out.push_back(to_json_impl(value));
    }
    return out;
  }
  if (pybind11::isinstance<pybind11::dict>(obj))
  {
    auto out = json::object();
    for (const pybind11::handle &key : obj)
    {
      out[pybind11::str(key).cast<std::string>()] = to_json_impl(obj[key]);
    }
    return out;
  }
  if (pybind11::isinstance<pybind11::object>(obj))
  {
    return obj.cast<korali::Experiment>();
  }

  throw std::runtime_error("to_json not implemented for this type of object: " + obj.cast<std::string>());
}
} // namespace detail

/**
 * @brief Wrapper for deserializing JSON objects to Pybind11 objects
 * @param j The JSON object to deserialize.
 * @return obj The deserialized Pybind11 object.
 */
inline pybind11::object adl_serializer<pybind11::object>::from_json(const json &j)
{
  return detail::from_json_impl(j);
}

/**
 * @brief Wrapper for serializing Pybind11 objects to JSON objects
 * @param j The JSON object to write.
 * @param obj The Pybind11 object to serialize
 */
inline void adl_serializer<pybind11::object>::to_json(json &j, const pybind11::object &obj)
{
  j = detail::to_json_impl(obj);
}

} // namespace knlohmann

