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

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

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
__global__ void Copykernel(CudaImageProps<TDimension>* in,
                           CudaImageProps<TDimension>* out,
                           CudaTransformProps<TDimension, TDimension>* trans)
{
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto j = blockIdx.y * blockDim.y + threadIdx.y;
  const auto k = blockIdx.z * blockDim.z + threadIdx.z;

  if(i == 1 && j == 1)
  {
    printf("in->size[%i,%i,%i] \n", in->size[0], in->size[1], in->size[2]);
    printf("out->size[%i,%i,%i]\n", out->size[0], out->size[1], out->size[2]);
    printf("trans->mat[%f,%f,%f]\n", trans->Matrix[0], trans->Matrix[1], trans->Matrix[2]);
    printf("trans->off[%f,%f,%f]\n", trans->Offset[0], trans->Offset[1], trans->Offset[2]);
  }


  if (i >= in->size[0] || j >= in->size[1] || k >= in->size[2])
    return;

  out->data[i + out->size[0] * (j + out->size[1] * k)] = tex3D<float>(in->texObj_in,i,j,k);
}

template <unsigned int TDimension>
void
CUDA_resample(
      CudaImageProps<TDimension>* h_in,
      CudaImageProps<TDimension>* h_out,
      CudaTransformProps<TDimension, TDimension>* h_trans
)
{
  CudaImageProps<TDimension>* dev_in;
  cudaMalloc((void**)&dev_in, sizeof(CudaImageProps<TDimension>));
  cudaCheckErrors("cudaMalloc dev_in");
  cudaMemcpy(dev_in, h_in, sizeof(CudaImageProps<TDimension>), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy dev_in");
  cudaMemcpy(&(dev_in->data), &(h_in->data), sizeof(float*), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy dev_in->data");

  auto channelDesc = cudaCreateChannelDesc<float>();
  auto volExtent = make_cudaExtent(h_in->size[0], h_in->size[1], h_in->size[2]);
  cudaArray* volArray = nullptr;
  cudaMalloc3DArray((cudaArray**)& volArray, &channelDesc, volExtent);
  cudaMemcpy3DParms CopyParams = { 0 };
  CopyParams.srcPtr = make_cudaPitchedPtr((void*)h_in->data, h_in->size[0] * sizeof(float), h_in->size[0], h_in->size[1]);
  CopyParams.dstArray = volArray;
  CopyParams.extent = volExtent;
  CopyParams.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&CopyParams);
  CUDA_CHECK_ERROR;

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = volArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&h_in->texObj_in, &resDesc, &texDesc, nullptr);
  cudaMemcpy(&(dev_in->texObj_in), &(h_in->texObj_in), sizeof(cudaTextureObject_t*), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy dev_in->texObj_in");
 
  CudaImageProps<TDimension>* dev_out;
  cudaMalloc((void**)&dev_out, sizeof(CudaImageProps<TDimension>));
  cudaCheckErrors("cudaMalloc dev_out");
  cudaMemcpy(dev_out, h_out, sizeof(CudaImageProps<TDimension>), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy dev_out");
  cudaMemcpy(&(dev_out->data), &(h_out->data), sizeof(float*), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy dev_in->data");

  CudaTransformProps<TDimension, TDimension>* dev_trans;
  cudaMalloc((void**)&dev_trans, sizeof(CudaTransformProps<TDimension, TDimension>));
  cudaCheckErrors("cudaMalloc dev_trans");
  cudaMemcpy(dev_trans, h_trans, sizeof(CudaTransformProps<TDimension, TDimension>), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy dev_trans");

  dim3 dimBlock = dim3(16, 16, 1);
  dim3 dimGrid = dim3(iDivUp(h_in->size[0], dimBlock.x), iDivUp(h_in->size[1], dimBlock.x));

  Copykernel<<<dimBlock,dimGrid>>>(dev_in, dev_out, dev_trans);
  cudaDeviceSynchronize();
  cudaCheckErrors("Copykernel");

}



template void RTK_EXPORT CUDA_resample<2>(CudaImageProps<2>*,CudaImageProps<2>*, CudaTransformProps<2,2>*);
template void RTK_EXPORT CUDA_resample<3>(CudaImageProps<3>*,CudaImageProps<3>*, CudaTransformProps<3,3>*);
