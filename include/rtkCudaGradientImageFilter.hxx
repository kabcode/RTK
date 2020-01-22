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

#ifndef rtkCudaGradientImageFilter_hxx
#define rtkCudaGradientImageFilter_hxx

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkCudaGradientImageFilter.h"
#  include "rtkCudaUtilities.hcu"
#  include "rtkCudaGradientImageFilter.hcu"

#  include <itkMacro.h>
#  include "rtkMacro.h"
#  include "itkCudaUtil.h"

namespace rtk
{

template <typename TInputImage, typename TOperatorValueType, typename TOutputValueType, typename TOutputImageType>
CudaGradientImageFilter<TInputImage, TOperatorValueType, TOutputValueType, TOutputImageType>::CudaGradientImageFilter() :
  : m_UseImageSpcing(true)
{
  m_BoundaryCondition = BoundaryConditions::ZeroFluxNeumann;
}

template <typename TInputImage, typename TOperatorValueType, typename TOutputValueType, typename TOutputImageType>
void CudaGradientImageFilter<TInputImage, TOperatorValueType, TOutputValueType, TOutputImageType>::
ChangeBoundaryCondition(ImageBoundaryCondition<TInputImage>* boundaryCondition)
{
  this->OverrideBoundaryCondition(boundaryCondition);
  if (dynamic_cast<ZeroFluxNeumannBoundaryCondition<TInputImage>*>(boundaryCondition))
  {
    m_CudaBoundaryCondition = BoundaryConditions::ZeroFluxNeumann;
  }
  if (dynamic_cast<ConstantBoundaryCondition<TInputImage>*>(boundaryCondition))
  {
    m_CudaBoundaryCondition = BoundaryConditions::Constant;
  }
}

template <typename TInputImage, typename TOperatorValueType, typename TOutputValueType, typename TOutputImageType>
void
CudaGradientImageFilter<TInputImage, TOperatorValueType, TOutputValueType, TOutputImageType>::
GPUGenerateData()
{
    unsigned int inputSize[OutputImageDimension];
    unsigned int outputSize[OutputImageDimension];
    float inputSpacing[OutputImageDimension];
    float outputSpacing[OutputImageDimension];
    float outputDirection[OutputImageDimension*OutputImageDimension];

    for (unsigned int i = 0; i < OutputImageDimension; i++)
    {
      inputSize[i] = this->GetInput()->GetBufferedRegion().GetSize()[i];
      outputSize[i] = this->GetOutput()->GetBufferedRegion().GetSize()[i];
      inputSpacing[i] = this->GetInput()->GetSpacing()[i];
      outputSpacing[i] = this->GetOutput()->GetSpacing()[i];

      if ((inputSize[i] != outputSize[i]) || (inputSpacing[i] != outputSpacing[i]))
      {
        std::cerr << "CUDAGradientImageFilter only handles input and output regions of equal size and spacing" << std::endl;
        exit(1);
      }


      for (unsigned int j = 0; j < OutputImageDimension; ++j)
      {
        {
          outputDirection[i + j * OutputImageDimension] = this->GetInput()->GetDirection()[i][j];
        }
      }


      if (!this->GetUseImageSpacing())
      {
        outputSpacing[i] = 1;
      }
    }

    if (!this->GetUseImageDirection())
    {
      memset(outputDirection, 0, OutputImageDimension*OutputImageDimension * sizeof(float));
      for (auto i = 0; i < OutputImageDimension; ++i)
      {
        outputDirection[i + i * OutputImageDimension] = 1;
      }
    }

    unsigned int boundaryCondition = 1;
    if (m_CudaBoundaryCondition == BoundaryConditions::ZeroFluxNeumann)
    {
      boundaryCondition = 1;
    }
    if (m_CudaBoundaryCondition == BoundaryConditions::Constant)
    {
      boundaryCondition = 3;
    }

    float *pin = *(float**)(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
    auto outputimage = this->GetOutput();
    float *pout = *(float**)(outputimage->GetCudaDataManager()->GetGPUBufferPointer());

    CUDA_gradient(pin, outputSize, outputSpacing, outputDirection, OutputImageDimension, boundaryCondition, pout);
  }

} // end namespace rtk

#endif // end conditional definition of the class

#endif
