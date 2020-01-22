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

#include "rtkConfiguration.h"
 // Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#include "itkGradientImageFilter.h"
#include "itkCudaImageToImageFilter.h"
#include "itkCudaUtil.h"
#include "itkCudaKernelManager.h"

enum class BoundaryConditions {
  ZeroFluxNeumann,
  Periodic,
  Constant
};

/** \class rtkCudaGradientImageFilter
 * \brief Gradient image filter implemented in CUDA
 * *
 *
 * \author Gordian Kabelitz
 *
 * \ingroup RTK ITKGradient CudaImageToImageFilter
 */

namespace rtk
{

  /** Create a helper Cuda Kernel class for CudaImageOps */
  itkCudaKernelClassMacro(rtkCudaGradientImageFilterKernel);

  template <class TInputImage = itk::CudaImage<float, 3>,
    typename TOperatorValueType = float,
    typename TOutputValueType = float,
    typename TOutputImage = itk::CudaImage< itk::CovariantVector<float, 3>>
    class ITK_EXPORT CudaGradientImageFilter
    : public itk::CudaImageToImageFilter<TInputImage,
    TOutputImage,
    itk::GradientImageFilter< TInputImage,
    TOperatorValueType,
    TOutputValueType,
    CudaImage< CovariantVector< TOutputValueType,
    TInputImage::ImageDimension >,
    TInputImage::ImageDimension > > >
  {
  public:
    ITK_DISALLOW_COPY_AND_ASSIGN(CudaGradientImageFilter);

    /** Extract dimension from input image. */
    static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;
    static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;
    static_assert(InputImageDimension <= 3 || OutputImageDimension <= 3, "Non supported dimensionality.");


    /** Standard class type alias. */
    using Self = CudaGradientImageFilter;
    using Superclass = itk::GradientImageFilter<TInputImage, TOutputImage>;
    using GPUSuperclass = itk::CudaImageToImageFilter<TInputImage, TOutputImage, Superclass>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    using VectorType = itk::Vector<float, 3>;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(CudaGradientImageFilter, ImageToImageFilter);

    using InputImageType = TInputImage;
    using InputImagePointer = typename InputImageType::Pointer;
    using OutputImageType = TOutputImage;
    using OutputImagePointer = typename OutputImageType::Pointer;

    /** Image type alias support */
    using InputPixelType = typename InputImageType::PixelType;
    using OperatorValueType = TOperatorValueType;
    using OutputValueType = TOutputValueType;
    using OutputPixelType = typename OutputImageType::PixelType;
    using CovariantVectorType = CovariantVector< OutputValueType, Self::OutputImageDimension >;
    using OutputImageRegionType = typename OutputImageType::RegionType;

    /** Set/Get whether or not the filter will use the spacing of the input
    image in its calculations */
    itkSetMacro(UseImageSpacing, bool);
    itkGetConstMacro(UseImageSpacing, bool);
    itkBooleanMacro(UseImageSpacing);

    void ChangeBoundaryCondition(ImageBoundaryCondition< TInputImage >* boundaryCondition);

  protected:
    CudaGradientImageFilter();
    ~CudaGradientImageFilter() = default;

    virtual void
      GPUGenerateData();

  private:
    bool m_UseImageSpacing;
    BoundaryConditions m_CudaBoundaryCondition;

  }; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
  #include "rtkCudaGradientImageFilter.hxx"
#endif

#endif // end conditional definition of the class

#endif
