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

#ifndef rtkCudaResampleImageFilter_h
#define rtkCudaResampleImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "itkResampleImageFilter.h"
#  include "itkCudaImageToImageFilter.h"
#  include "itkCudaUtil.h"
#  include "itkCudaKernelManager.h"
#  include "RTKExport.h"

/** \class CudaResampleImageFilter
 * \brief Resample image with CUDA implementation
 *
 * \author Gordian Kabelitz
 *
 * \ingroup RTK ImageGridFilter CudaImageToImageFilter
 */

namespace rtk
{

/** Create a helper Cuda Kernel class for CudaImageOps */
itkCudaKernelClassMacro(rtkCudaResampleImageFilterKernel);

template <class TInputImage,
          class TOutputImage = TInputImage,
          class TInterpolatorPrecisionType = float,
          class TTransformPrecisionType = TInterpolatorPrecisionType>
class ITK_EXPORT CudaResampleImageFilter
  : public itk::
      CudaImageToImageFilter<TInputImage, TOutputImage, itk::ResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TInterpolatorPrecisionType>>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaResampleImageFilter);

  /** Standard class type alias. */
  using Self = CudaResampleImageFilter;
  using Superclass = itk::ResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TInterpolatorPrecisionType>;
  using GPUSuperclass = itk::CudaImageToImageFilter<TInputImage, TOutputImage, Superclass>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using VectorType = itk::Vector<float, 3>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaResampleImageFilter, ImageToImageFilter);

  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using InputImagePointer = typename InputImageType::Pointer;
  using InputImageConstPointer = typename InputImageType::ConstPointer;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using InputImageRegionType = typename InputImageType::RegionType;

  static constexpr unsigned int ImageDimension = TOutputImage::ImageDimension;
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;


protected:
  CudaResampleImageFilter();
  ~CudaResampleImageFilter() = default;

  virtual void GPUGenerateData();

private:
 

}; // end of class

} // end namespace rtk

#  ifndef ITK_MANUAL_INSTANTIATION
#    include "rtkCudaResampleImageFilter.hxx"
#  endif

#endif // end conditional definition of the class

#endif
