#include "auxiliar/koraliJson.hpp"
#include "auxiliar/logger.hpp"
#include "auxiliar/py2json.hpp"
#include "sample/sample.hpp"

namespace korali
{
KoraliJson::KoraliJson()
{
  _opt = &_js;
}

void KoraliJson::traverseKey(pybind11::object key)
{
  if (pybind11::isinstance<pybind11::str>(key))
  {
    std::string keyStr = key.cast<std::string>();
    _opt = &((*_opt)[keyStr]);
    return;
  }

  if (pybind11::isinstance<pybind11::int_>(key))
  {
    int keyInt = key.cast<int>();
    _opt = &((*_opt)[keyInt]);
    return;
  }
}

void KoraliJson::setItem(const pybind11::object key, const pybind11::object val)
{
  traverseKey(key);

  *_opt = val;
  _opt = &_js;
}

pybind11::object KoraliJson::getItem(const pybind11::object key)
{
  traverseKey(key);

  if (isElemental(*_opt))
  {
    auto tmp = _opt;
    _opt = &_js;
    return *tmp;
  }

  return pybind11::cast(this);
}

knlohmann::json &KoraliJson::operator[](const std::string &key)
{
  return _js[key];
}

knlohmann::json &KoraliJson::operator[](const unsigned long int &key)
{
  return _js[key];
}

knlohmann::json &KoraliJson::getJson()
{
  return _js;
}

void KoraliJson::getCopy(knlohmann::json &dst) const
{
  dst = _js;
}

void KoraliJson::setJson(knlohmann::json &js)
{
  _js = js;
}

bool KoraliJson::contains(const std::string &key)
{
  if (_js.find(key) == _js.end()) return false;
  return true;
}

} // namespace korali
