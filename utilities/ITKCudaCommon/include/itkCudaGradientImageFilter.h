/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
#ifndef itkCudaGradientImageFilter_h
#define itkCudaGradientImageFilter_h

#include "itkGradientImageFilter.h"
#include "itkCudaImageToImageFilter.h"
#include "itkCudaUtil.h"
#include "itkCudaKernelManager.h"

enum class BoundaryConditions {
  ZeroFluxNeumann,
  Periodic,
  Constant
};

/** \class CudaGradientImageFilter
* \brief Trilinear interpolation forward projection implemented in CUDA
*
* Images with a maximum of three dimensions are supported.
*
* \ingroup RTK CudaImageToImageFilter
*/

namespace itk
{

    /** Create a helper Cuda Kernel class for CudaImageOps */
    itkCudaKernelClassMacro(itkCudaGradientImageFilter);

    template <typename TInputImage,
    typename TOperatorValueType = float,
    typename TOutputValueType = float,
    typename TOutputImage = CudaImage< CovariantVector< TOutputValueType, TInputImage::ImageDimension >, TInputImage::ImageDimension >>
    class ITK_EXPORT
    CudaGradientImageFilter : public CudaImageToImageFilter< TInputImage, TOutputImage, 
        GradientImageFilter< TInputImage, TOperatorValueType, TOutputValueType, CudaImage< CovariantVector< TOutputValueType, TInputImage::ImageDimension >, TInputImage::ImageDimension > > >
    {
    public:
        ITK_DISALLOW_COPY_AND_ASSIGN(CudaGradientImageFilter);

        /** Extract dimension from input image. */
        static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;
        static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;
        static_assert(InputImageDimension <= 3 || OutputImageDimension <= 3, "Non supported dimensionality.");

        /** Standard class typedefs. */
        using Self          = CudaGradientImageFilter;
        using Superclass    = GradientImageFilter<TInputImage, TOutputImage>;
        using GPUSuperclass = CudaImageToImageFilter<TInputImage, TOutputImage, Superclass >;
        using Pointer       = SmartPointer<Self>;
        using ConstPointer  = SmartPointer<const Self>;

        /** Method for creation through the object factory. */
        itkNewMacro(Self);

        /** Run-time type information (and related methods). */
        itkTypeMacro(CudaGradientImageFilter, ImageToImageFilter);

        /** Convenient type alias for simplifying declarations. */
        using InputImageType = TInputImage;
        using InputImagePointer = typename InputImageType::Pointer;
        using OutputImageType = TOutputImage;
        using OutputImagePointer = typename OutputImageType::Pointer;

        /** Image type alias support */
        using InputPixelType        = typename InputImageType::PixelType;
        using OperatorValueType     = TOperatorValueType;
        using OutputValueType       = TOutputValueType;
        using OutputPixelType       = typename OutputImageType::PixelType;
        using CovariantVectorType   = CovariantVector< OutputValueType, Self::OutputImageDimension >;
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

        void PrintSelf(std::ostream & os, Indent indent) const override;
        void GPUGenerateData() override;

    private:
        bool m_UseImageSpacing;
        unsigned int m_UnsignedBoundaryCondition;

    }; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaGradientImageFilter.hxx"
#endif

#endif
