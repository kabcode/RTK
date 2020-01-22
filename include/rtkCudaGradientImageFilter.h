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

#ifndef rtkCudaGradientImageFilter_h
#define rtkCudaGradientImageFilter_h

#ifdef RTK_USE_CUDA

#include "itkGradientImageFilter.h"
#include "itkCudaImageToImageFilter.h"

/** \class rtkCudaGradientImageFilter
 * \brief Gradient image filter implemented in CUDA
 * *
 *
 * \author Gordian Kabelitz
 *
 * \ingroup RTK ITKGradient CudaImageToImageFilter
 */

using CudaImageType = itk::CudaImage<float, 3>;
using CudaVectorImageType = itk::CudaImage<itk::CovariantVector<float, 3>>;

namespace rtk
{
  class ITK_EXPORT CudaGradientImageFilter :
    public itk::CudaImageToImageFilter<CudaImageType, CudaImageType,itk::GradientImageFilter<CudaImageType,float, float, CudaVectorImageType>>
  {
    
  };

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
  #include "rtkCudaGradientImageFilter.hxx"
#endif

#endif // end conditional definition of the class

#endif
