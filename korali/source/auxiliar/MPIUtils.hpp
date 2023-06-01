/** \file
* @brief Contains the helper definitions for MPI
******************************************************************************/

#pragma once

#include <config.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#ifdef _KORALI_USE_MPI

#include <mpi.h>

namespace korali
{

/**
* @brief Communicator storage for the current Korali Worker
*/
extern MPI_Comm __KoraliGlobalMPIComm;

/**
* @brief Communicator storage for the current Korali Worker
*/
extern MPI_Comm __koraliWorkerMPIComm;

/**
  * @brief Sets global MPI communicator
  * @param comm The MPI communicator to use
  * @return The return code of MPI_Comm_Dup
  */
extern int setKoraliMPIComm(const MPI_Comm &comm);

/**
   * @brief Returns MPI communicator for the current Korali Worker
   * @return A pointer to the MPI Communicator
   */
extern void *getWorkerMPIComm();

  /**
* @brief Remembers whether the MPI was given by the used. Otherwise use MPI_COMM_WORLD
*/
extern bool __isMPICommGiven;

#define __KORALI_MPI_MESSAGE_JSON_TAG 1

}

#ifdef _KORALI_USE_MPI4PY
#ifndef _KORALI_NO_MPI4PY

#include <mpi4py/mpi4py.h>

// MPI Communicator handler for pybind11
// Credits to H. Tittich
// https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py

struct mpi4py_comm
{
  mpi4py_comm() = default;
  mpi4py_comm(MPI_Comm value) : value(value) {}
  operator MPI_Comm () { return value; }

  MPI_Comm value;
};

namespace pybind11 {
 namespace detail {
  template <> struct type_caster<mpi4py_comm>
  {
    public:
      PYBIND11_TYPE_CASTER(mpi4py_comm, _("mpi4py_comm"));

      // Python -> C++
      bool load(handle src, bool)
      {
        PyObject *py_src = src.ptr();

        // Check that we have been passed an mpi4py communicator
        if (PyObject_TypeCheck(py_src, &PyMPIComm_Type))  value.value = *PyMPIComm_Get(py_src);
        else return false;

        return !PyErr_Occurred();
      }

      // C++ -> Python
      static handle cast(mpi4py_comm src, return_value_policy, handle) {  return PyMPIComm_New(src.value);  }

      void initialize()
      {
       import_mpi4py();
      }
  };
 }
} // namespace pybind11::detail

namespace korali
{
 extern mpi4py_comm getMPI4PyComm();

 /**
  * @brief Sets global MPI communicator via MPI4py
  * @param comm The MPI4py communicator to use
  */
 extern void setMPI4PyComm(mpi4py_comm comm);
}

#endif
#endif

#else // #ifdef _KORALI_USE_MPI == false

namespace korali
{

 /**
 * @brief Dummy communicator storage for the current Korali Worker
 */
 typedef long int MPI_Comm;

 /**
    * @brief Error handler for when MPI is not defined
    * @param ... accepts any parameters since it will fail anyway
    * @return Error code -1
    */
 extern int setKoraliMPIComm(...);

 /**
   * @brief Error handler for when MPI is not defined
   * @return A NULL pointer
   */
 extern void *getWorkerMPIComm();

}

#endif // #ifdef _KORALI_USE_MPI
