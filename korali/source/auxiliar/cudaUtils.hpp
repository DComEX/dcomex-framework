/** \file
* @brief Contains auxiliar error reporting functions to CUDA and cuDNN
*        Credits to Motoki Sato (https://gist.github.com/aonotas)
******************************************************************************************************/

#pragma once

#ifdef _KORALI_USE_CUDNN

  #include <cuda.h>
  #include <cudnn.h>

  // Define some error checking macros.
  #define cudaErrCheck(stat)                     \
    {                                            \
      cudaErrCheck_((stat), __FILE__, __LINE__); \
    }
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
  if (stat != cudaSuccess)
  {
    fprintf(stderr, "[Korali] CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    exit(-1);
  }
}

  #define cudnnErrCheck(stat)                     \
    {                                             \
      cudnnErrCheck_((stat), __FILE__, __LINE__); \
    }
inline void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line)
{
  if (stat != CUDNN_STATUS_SUCCESS)
  {
    fprintf(stderr, "[Korali] cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
    exit(-1);
  }
}

#endif // _KORALI_USE_CUDNN

