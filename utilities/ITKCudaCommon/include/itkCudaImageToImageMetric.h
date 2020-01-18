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
#ifndef itkCudaImageToImageMetric_h
#define itkCudaImageToImageMetric_h

#include "itkImageToImageMetric.h"
#include "itkCudaImage.h"
#include "itkCudaGradientImageFilter.h"


namespace itk
{
/** \class CudaImageToImageMetric
 * \brief Computes similarity between regions of two images using a CUDA-based implementation.
 *
 * This Class is templated over the type of the two input images.
 * It expects a Transform and an Interpolator to be plugged in.
 * This particular class is the base class for a hierarchy of
 * similarity metrics.
 *
 * This class computes a value that measures the similarity
 * between the Fixed image and the transformed Moving image.
 * The Interpolator is used to compute intensity values on
 * non-grid positions resulting from mapping points through
 * the Transform.
 *
 *
 * \ingroup RegistrationMetrics
 *
 * \ingroup ITKCudaRegistrationCommon ImageFeatures
 */

template< typename TFixedImage,
          typename TMovingImage,
          typename TParentImageToImageMetric = ImageToImageMetric< TFixedImage, TMovingImage> >
class ITK_EXPORT CudaImageToImageMetric : public TParentImageToImageMetric
{
public:
    ITK_DISALLOW_COPY_AND_ASSIGN(CudaImageToImageMetric);

    /** Standard class type aliases. */
    using Self = CudaImageToImageMetric;
    using Superclass = TParentImageToImageMetric;
    using Pointer = SmartPointer< Self >;
    using ConstPointer = SmartPointer< const Self >;

    /** Type used for representing point components  */
    using CoordinateRepresentationType = typename Superclass::ParametersValueType;

    /** Run-time type information (and related methods). */
    itkTypeMacro(CudaImageToImageMetric, TParentImageToImageMetric);

    /**  Type of the moving Image. */
    using MovingImageType = TMovingImage;
    using MovingImagePixelType = typename TMovingImage::PixelType;
    using MovingImageConstPointer = typename MovingImageType::ConstPointer;

    /**  Type of the fixed Image. */
    using FixedImageType = TFixedImage;
    using FixedImagePixelType = typename TFixedImage::PixelType;
    using FixedImageConstPointer = typename FixedImageType::ConstPointer;
    using FixedImageRegionType = typename FixedImageType::RegionType;

    /** Constants for the image dimensions */
    static constexpr unsigned int MovingImageDimension = TMovingImage::ImageDimension;
    static constexpr unsigned int FixedImageDimension = TFixedImage::ImageDimension;

    /**  Type of the Transform Base class */
    using TransformType = typename Superclass::TransformType;
    using TransformPointer = typename TransformType::Pointer;
    using InputPointType = typename TransformType::InputPointType;
    using OutputPointType = typename TransformType::OutputPointType;
    using TransformParametersType = typename TransformType::ParametersType;
    using TransformJacobianType = typename TransformType::JacobianType;

    /** Index and Point type alias support */
    using FixedImageIndexType = typename FixedImageType::IndexType;
    using FixedImageIndexValueType = typename FixedImageIndexType::IndexValueType;
    using MovingImageIndexType = typename MovingImageType::IndexType;
    using FixedImagePointType = typename TransformType::InputPointType;
    using MovingImagePointType = typename TransformType::OutputPointType;

    using FixedImageIndexContainer = std::vector< FixedImageIndexType >;

    /**  Type of the Interpolator Base class */
    using InterpolatorType = typename Superclass::InterpolatorType;
    using InterpolatorPointer = typename InterpolatorType::Pointer;

    /** Cuda filter to compute the gradient of the Moving Image */
    using RealType = typename NumericTraits< MovingImagePixelType >::FloatType;
    using GradientPixelType = CovariantVector< RealType, MovingImageDimension >;
    using GradientImageType = CudaImage< GradientPixelType, MovingImageDimension >;
    using GradientImagePointer = SmartPointer<GradientImageType>;
    using GradientImageFilterType = CudaGradientImageFilter< MovingImageType, float, float, GradientImageType >;
    using GradientImageFilterPointer = typename GradientImageFilterType::Pointer;

    /**  Type for the mask of the fixed image. Only pixels that are "inside"
         this mask will be considered for the computation of the metric */
    using FixedImageMaskType = SpatialObject< Self::FixedImageDimension >;
    using FixedImageMaskPointer = typename FixedImageMaskType::Pointer;
    using FixedImageMaskConstPointer = typename FixedImageMaskType::ConstPointer;

    /**  Type for the mask of the moving image. Only pixels that are "inside"
         this mask will be considered for the computation of the metric */
    using MovingImageMaskType = SpatialObject< Self::MovingImageDimension >;
    using MovingImageMaskPointer = typename MovingImageMaskType::Pointer;
    using MovingImageMaskConstPointer = typename MovingImageMaskType::ConstPointer;

    /**  Type of the measure. */
    using MeasureType = typename Superclass::MeasureType;

    /**  Type of the derivative. */
    using DerivativeType = typename Superclass::DerivativeType;

    /**  Type of the parameters. */
    using ParametersType = typename Superclass::ParametersType;

    // macro to set if Cuda is used
    itkSetMacro(GPUEnabled, bool);
    itkGetConstMacro(GPUEnabled, bool);
    itkBooleanMacro(GPUEnabled);

    /** Computes the gradient image and assigns it to m_GradientImage */
    virtual void ComputeGradient() override;

    /** Initialize the Metric by making sure that all the components
     *  are present and plugged together correctly     */
    virtual void Initialize() override;

    virtual MeasureType GPUGetValue(const ParametersType& parameters) const;
    virtual void GPUGetDerivative(const ParametersType &parameters, DerivativeType &derivative);
    virtual void GPUGetValueAndDerivative(const ParametersType &parameters, MeasureType &value, DerivativeType &derivative) const;

protected:
    CudaImageToImageMetric();
    ~CudaImageToImageMetric() override;

    bool m_GPUEnabled;

    GradientImagePointer m_GradientImage;

    virtual void PrintSelf(std::ostream & os, Indent indent) const override;
};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaImageToImageMetric.hxx"
#endif

#endif
