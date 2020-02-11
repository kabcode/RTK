/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

/*****************
 *  rtk #includes *
 *****************/
#include "rtkCudaUtilities.hcu"
#include "rtkConfiguration.h"
#include "rtkCudaResampleImageFilter.hcu"

/*****************
 *  C   #includes *
 *****************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/*****************
 * CUDA #includes *
 *****************/
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template<unsigned int TDimension>
__global__ void Copykernel(CudaImageProps<TDimension>* in, CudaImageProps<TDimension>* out)
{
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto j = blockIdx.y * blockDim.y + threadIdx.y;
  const auto k = blockIdx.z * blockDim.z + threadIdx.z;

  int sz[TDimension] = out->size;
  if (i >= sz[0] || j >= sz[1] || k >= sz[2])
    return;

    out->data[i + sz[0] * (j + sz[1] * k)] = in->data[i + sz[0] * (j + sz[1] * k)];
}

template <unsigned int TDimension>
void
CUDA_resample(
      CudaImageProps<TDimension>* h_in,
      CudaImageProps<TDimension>* h_out
)
{
  CudaImageProps<TDimension>* dev_in;
  cudaMalloc(dev_in, sizeof(CudaImageProps<TDimension>));


  CudaImageProps<TDimension>* dev_out;
  Copykernel<<<1,32>>>(dev_in, dev_out);
}



template void RTK_EXPORT CUDA_resample<2>(CudaImageProps<2>*,CudaImageProps<2>*);
template void RTK_EXPORT CUDA_resample<3>(CudaImageProps<3>*,CudaImageProps<3>*);
