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
#include "rtkCudaGradientImageFilter.hcu"

#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int3 c_Size;
__constant__ float3 c_Spacing;
__constant__ float c_Direction[3][3]; // 2D constant array for rotation matrices up to 3D

__global__ void gradient_kernel_2d(cudaTextureObject_t in, float* grad, const int len = 2);
__global__ void gradient_kernel_3d(cudaTextureObject_t in, float* grad, const int len = 3);


void
CUDA_gradient(
  float* dev_in,
  unsigned int* size,
  float* spacing,
  float* direction,
  unsigned int dimension,
  unsigned int boundaryCondition,
  float* dev_out)
{

  auto addressmode = cudaTextureAddressMode(boundaryCondition);
  unsigned int cSize[] = { 1,1,1 };
  float cSpacing[] = { 1,1,1 };
  float cDirection[][3] = { {1.f,0,0},{0.f,1,0},{0.f,0,1} };

  // Output volume size and spacing
  for (unsigned int i = 0; i < dimension; ++i)
  {
    cSize[i] = size[i];
    cSpacing[i] = spacing[i];
    for (auto j = 0; j < dimension; ++j)
    {
      cDirection[i][j] = direction[i + j * dimension];
    }
  }

  long int outputMemorySize = cSize[0] * cSize[1] * cSize[2] * dimension * sizeof(float);
  cudaMemset(dev_out, 0, outputMemorySize);

  cudaMemcpyToSymbol(c_Size, cSize, sizeof(int3));
  cudaMemcpyToSymbol(c_Spacing, cSpacing, sizeof(float3));
  cudaMemcpyToSymbol(c_Direction, cDirection, 3 * 3 * sizeof(float));

  switch (dimension)
  {
  case 1:
  {
    break;
  }
  case 2:
  {
    // Allocate CUDA array in device memory
    auto channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* imgArray = nullptr;
    cudaMallocArray(&imgArray, &channelDesc, cSize[0], cSize[1]);
    cudaMemcpyToArray(imgArray, 0, 0, (void*)dev_in, cSize[0] * cSize[1] * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR;

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = imgArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = addressmode;
    texDesc.addressMode[1] = addressmode;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // Thread Block Dimensions
    auto dimBlock = dim3(16, 16);
    auto blocksInX = iDivUp(cSize[0], dimBlock.x);
    auto blocksInY = iDivUp(cSize[1], dimBlock.y);
    auto dimGrid = dim3(blocksInX, blocksInY);

    gradient_kernel_2d << < dimGrid, dimBlock >> > (texObj, dev_out);
    CUDA_CHECK_ERROR;

    // Clean up
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(imgArray);
    break;
  }
  case 3:
  {
    // Allocate CUDA array in device memory
    auto channelDesc = cudaCreateChannelDesc<float>();
    auto volExtent = make_cudaExtent(cSize[0], cSize[1], cSize[2]);
    cudaArray* volArray = nullptr;
    cudaMalloc3DArray((cudaArray**)& volArray, &channelDesc, volExtent);
    cudaMemcpy3DParms CopyParams = { 0 };
    CopyParams.srcPtr = make_cudaPitchedPtr((void*)dev_in, cSize[0] * sizeof(float), cSize[0], cSize[1]);
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
    texDesc.addressMode[0] = addressmode;
    texDesc.addressMode[1] = addressmode;
    texDesc.addressMode[2] = addressmode;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // Thread Block Dimensions
    dim3 dimBlock = dim3(8, 8, 8);

    int blocksInX = iDivUp(cSize[0], dimBlock.x);
    int blocksInY = iDivUp(cSize[1], dimBlock.y);
    int blocksInZ = iDivUp(cSize[2], dimBlock.z);

    dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
    gradient_kernel_3d << < dimGrid, dimBlock >> > (texObj, dev_out);
    CUDA_CHECK_ERROR;

    // Clean up
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(volArray);
    break;
  }
  default:
    itkGenericExceptionMacro("This dimensionality is not supported.")
  }
}

__global__
void
gradient_kernel_2d(cudaTextureObject_t in, float * grad, const int len)
{
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= c_Size.x || j >= c_Size.y)
    return;

  const float _01 = tex2D<float>(in, i - 0.5, j + 0.5);
  const float _21 = tex2D<float>(in, i + 1.5, j + 0.5);
  const float _10 = tex2D<float>(in, i + 0.5, j - 0.5);
  const float _12 = tex2D<float>(in, i + 0.5, j + 1.5);

  const long int id = len * (i + c_Size.x * j);
  float grads[] = { 0.5f * (_21 - _01) / c_Spacing.x, 0.5f * (_12 - _10) / c_Spacing.y };

  for (unsigned int m = 0; m < 2; ++m)
  {
    float sum = 0.f;
    sum += c_Direction[m][0] * grads[0];
    sum += c_Direction[m][1] * grads[1];
    grad[id + m] = sum;
  }

}

__global__
void
gradient_kernel_3d(cudaTextureObject_t in, float* grad, int len)
{
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto j = blockIdx.y * blockDim.y + threadIdx.y;
  const auto k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y || k >= c_Size.z)
    return;

  //float _000 = tex3D<float>(in, i - 0.5, j - 0.5, k - 0.5);
  //float _001 = tex3D<float>(in, i - 0.5, j - 0.5, k + 0.5);
  //float _002 = tex3D<float>(in, i - 0.5, j - 0.5, k + 1.5);
  //float _010 = tex3D<float>(in, i - 0.5, j + 0.5, k - 0.5);
  const float _011 = tex3D<float>(in, i - 0.5, j + 0.5, k + 0.5);
  //float _012 = tex3D<float>(in, i - 0.5, j + 0.5, k + 1.5);
  //float _020 = tex3D<float>(in, i - 0.5, j + 1.5, k - 0.5);
  //float _021 = tex3D<float>(in, i - 0.5, j + 1.5, k + 0.5);
  //float _022 = tex3D<float>(in, i - 0.5, j + 1.5, k + 1.5);
  //float _100 = tex3D<float>(in, i + 0.5, j - 0.5, k - 0.5);
  const float _101 = tex3D<float>(in, i + 0.5, j - 0.5, k + 0.5);
  //float _102 = tex3D<float>(in, i + 0.5, j - 0.5, k + 1.5);
  const float _110 = tex3D<float>(in, i + 0.5, j + 0.5, k - 0.5);
  //float _111 = tex3D<float>(in, i + 0.5, j + 0.5, k + 0.5);
  const float _112 = tex3D<float>(in, i + 0.5, j + 0.5, k + 1.5);
  //float _120 = tex3D<float>(in, i + 0.5, j + 1.5, k - 0.5);
  const float _121 = tex3D<float>(in, i + 0.5, j + 1.5, k + 0.5);
  //float _122 = tex3D<float>(in, i + 0.5, j + 1.5, k + 1.5);
  //float _200 = tex3D<float>(in, i + 1.5, j - 0.5, k - 0.5);
  //float _201 = tex3D<float>(in, i + 1.5, j - 0.5, k + 0.5);
  //float _202 = tex3D<float>(in, i + 1.5, j - 0.5, k + 1.5);
  //float _210 = tex3D<float>(in, i + 1.5, j + 0.5, k - 0.5);
  const float _211 = tex3D<float>(in, i + 1.5, j + 0.5, k + 0.5);
  //float _212 = tex3D<float>(in, i + 1.5, j + 0.5, k + 1.5);
  //float _220 = tex3D<float>(in, i + 1.5, j + 1.5, k - 0.5);
  //float _221 = tex3D<float>(in, i + 1.5, j + 1.5, k + 0.5);
  //float _222 = tex3D<float>(in, i + 1.5, j + 1.5, k + 1.5);	
  const long int id = len * (i + c_Size.x * (j + k * c_Size.y));

  grad[id + 0] = 0.5f * (_211 - _011) / c_Spacing.x;
  grad[id + 1] = 0.5f * (_121 - _101) / c_Spacing.y;
  grad[id + 2] = 0.5f * (_112 - _110) / c_Spacing.z;
}
