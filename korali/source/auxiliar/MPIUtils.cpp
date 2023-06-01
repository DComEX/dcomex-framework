/** \file
* @brief Contains the helper definitions for MPI
******************************************************************************/

#include <auxiliar/MPIUtils.hpp>
#include <auxiliar/logger.hpp>

namespace korali
{

#ifdef _KORALI_USE_MPI

MPI_Comm __KoraliGlobalMPIComm;
MPI_Comm __koraliWorkerMPIComm;
bool __isMPICommGiven;

int setKoraliMPIComm(const MPI_Comm &comm)
{
  __isMPICommGiven = true;
  return MPI_Comm_dup(comm, &__KoraliGlobalMPIComm);
}

void *getWorkerMPIComm()
{
 return &__koraliWorkerMPIComm;
}

#ifdef _KORALI_USE_MPI4PY
#ifndef _KORALI_NO_MPI4PY

mpi4py_comm getMPI4PyComm()
{
 return __koraliWorkerMPIComm;
}

void setMPI4PyComm(mpi4py_comm comm)
{
 setKoraliMPIComm(comm);
}

#endif
#endif

#else

int setKoraliMPIComm(...)
{
  KORALI_LOG_ERROR("Trying to setup MPI communicator but Korali was installed without support for MPI.\n");
  return -1;
}

void *getWorkerMPIComm()
{
  KORALI_LOG_ERROR("Trying to setup MPI communicator but Korali was installed without support for MPI.\n");
  return NULL;
}

#endif

}
