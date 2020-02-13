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
inline __device__ void multiplyMatVec(float* Mat, float* Vec, float* Return)
{
  #pragma unroll
  for(unsigned int i = 0; i < TDimension; ++i)
  {
    #pragma unroll
    Return[i] = 0;
    for(unsigned int j = 0; j < TDimension; ++j)
    {
      Return[i] =+ Mat[j + TDimension * i] * Vec[j];
    }
  }
}

template<unsigned int TDimension>
inline __device__ void multiplyScalarVec(float scalar, float* Vec, float* Return)
{
  #pragma unroll
  for(unsigned int i = 0; i < TDimension; ++i)
  {
    Return[i] = scalar * Vec[i];
  }
}

template<unsigned int TDimension>
inline __device__ void cwiseVecVec(float* Vec1, float* Vec2, float* Return)
{
  #pragma unroll
  for (unsigned int i = 0; i < TDimension; ++i)
  {
    Return[i] = Vec1[i] * Vec2[i];
  }
}

template<unsigned int TDimension>
inline __device__ void addVecVec(float* Vec1, float* Vec2, float* Return)
{
#pragma unroll
  for (unsigned int i = 0; i < TDimension; ++i)
  {
    Return[i] = Vec1[i] + Vec2[i];
  }
}

template<unsigned int TDimension>
inline __device__ void subVecVec(float* Vec1, float* Vec2, float* Return)
{
#pragma unroll
  for (unsigned int i = 0; i < TDimension; ++i)
  {
    Return[i] = Vec1[i] - Vec2[i];
  }
}

template<unsigned int TDimension>
inline __device__ void flattenMat(float* Mat)
{
#pragma unroll
  for (unsigned int i = 0; i < TDimension; ++i)
  {
    
  }
}

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
    printf("in->direction\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
      in->direction[0], in->direction[1], in->direction[2],
      in->direction[3], in->direction[4], in->direction[5],
      in->direction[6], in->direction[7], in->direction[8]);
    printf("trans->mat\n[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n",
      trans->Matrix[0], trans->Matrix[1], trans->Matrix[2],
      trans->Matrix[3], trans->Matrix[4], trans->Matrix[5],
      trans->Matrix[6], trans->Matrix[7], trans->Matrix[8]);
    printf("trans->off[%f,%f,%f]\n", trans->Offset[0], trans->Offset[1], trans->Offset[2]);
  }


  if (i >= out->size[0] || j >= out->size[1] || k >= out->size[2])
    return;

  float idx_out[] = { i*1.0f, j*1.0f, k*1.0f };
  // compute physical coordinates for output pixel
  float physicalOut[TDimension];
  cwiseVecVec<TDimension>(out->spacing, idx_out, physicalOut);
  multiplyMatVec<TDimension>(out->direction, physicalOut, physicalOut);
  addVecVec<TDimension>(out->origin, physicalOut, physicalOut);

  // apply inverse transform towards input image

  // compute indices for physical coordinates
  float idx_in[] = { 0.f,0.f,0.f };
  subVecVec<TDimension>(physicalOut, in->origin, idx_in);
  float transformToIndexMatrix[TDimension*TDimension];
  multiplyMatVec<TDimension>(in->direction, in->spacing, transformToIndexMatrix);
  // -> Get somehow the inverse of this product (or use the computed physicalPointToIndex matrix from ImageBase.h)

  if (i == 1 && j == 1)
  {
    printf("physicalOut[%f,%f,%f]\n", physicalOut[0], physicalOut[1], physicalOut[2]);
  }

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
/*
  float* h_in_dir;
  cudaMalloc((void**)&h_in_dir, sizeof(float)*TDimension*TDimension);
  cudaCheckErrors("cudaMalloc d_in_dir");
  cudaMemcpy(h_in_dir, &h_in->direction, sizeof(float)*TDimension*TDimension, cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy h_in_dir");
  cudaMemcpy(&(dev_in->direction), &h_in_dir, sizeof(float*), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy dev_in->dir");
  */
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
