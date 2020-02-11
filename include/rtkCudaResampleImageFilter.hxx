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

#ifndef rtkCudaResampleImageFilter_hxx
#define rtkCudaResampleImageFilter_hxx

#include "rtkConfiguration.h"
 // Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkCudaResampleImageFilter.h"
#  include "rtkCudaUtilities.hcu"
#  include "rtkCudaResampleImageFilter.hcu"

#  include <itkMacro.h>
#  include "rtkMacro.h"
#  include "itkCudaUtil.h"

namespace rtk
{

  template <class TInputImage, class TOutputImage, class TInterpolatorPrecisionType, class TTransformPrecisionType>
  CudaResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::CudaResampleImageFilter()
  {}


  template <class TInputImage, class TOutputImage, class TInterpolatorPrecisionType, class TTransformPrecisionType>
  void
  CudaResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>::GPUGenerateData()
  {
    CudaImageProps<InputImageDimension> h_in;
    for(unsigned int i = 0; i < InputImageDimension; ++i)
    {
      h_in.size[i] = this->GetInput()->GetLargestPossibleRegion().GetSize()[i];
      h_in.spacing[i] = this->GetInput()->GetSpacing()[i];
      h_in.origin[i] = this->GetInput()->GetOrigin()[i];
      h_in.direction[i][0] = this->GetInput()->GetDirection()[i][0];
    }
    
    h_in.data = *static_cast<float**>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());

    CudaImageProps<ImageDimension> h_out;
    for (unsigned int i = 0; i < ImageDimension; ++i)
    {
      h_out.size[i] = this->GetOutput()->GetLargestPossibleRegion().GetSize()[i];
      h_out.spacing[i] = this->GetOutput()->GetSpacing()[i];
      h_out.origin[i] = this->GetOutput()->GetOrigin()[i];
      h_out.direction[i][0] = this->GetOutput()->GetDirection()[i][0];
    }
    h_out.data = *static_cast<float**>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
    CUDA_resample<InputImageDimension>(&h_in, &h_out);
  }
} // end namespace rtk

#endif // end conditional definition of the class

#endif
