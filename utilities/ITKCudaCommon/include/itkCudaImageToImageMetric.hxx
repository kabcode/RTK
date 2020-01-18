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
#ifndef itkCudaImageToImageMetric_hxx
#define itkCudaImageToImageMetric_hxx

#include "itkCudaImageToImageMetric.h"

namespace itk
{
  /**
   * Constructor
   */
  template <typename TFixedImage, typename TMovingImage, typename TParentImageToImageMetric>
  CudaImageToImageMetric<TFixedImage, TMovingImage, TParentImageToImageMetric>
    ::CudaImageToImageMetric() :
    m_GPUEnabled(true)
  {

  }

  template< typename TFixedImage, typename TMovingImage, typename TParentImageToImageMetric >
  CudaImageToImageMetric< TFixedImage, TMovingImage, TParentImageToImageMetric >
    ::~CudaImageToImageMetric()
  {
  }

  /**
   * Initialize
   */
  template< typename TFixedImage, typename TMovingImage, typename TParentImageToImageMetric >
  void
  CudaImageToImageMetric< TFixedImage, TMovingImage, TParentImageToImageMetric >
  ::Initialize()
  {
    Superclass::Initialize();
  }

  /*
  template<typename TFixedImage, typename TMovingImage, typename TParentImageToImageMetric>
  MeasureType
  CudaImageToImageMetric<TFixedImage, TMovingImage, TParentImageToImageMetric>::GPUGetValue(const ParametersType& parameters) const
  {
    if (!m_GPUEnabled)
    {
      return Superclass::GetValue(parameters);
    }
    else
    {
      // TODO: Something that needs to be done before?
      return GPUGetValue(parameters);
    }
  }
  */
  template<typename TFixedImage, typename TMovingImage, typename TParentImageToImageMetric>
  void
  CudaImageToImageMetric<TFixedImage, TMovingImage, TParentImageToImageMetric>::GPUGetDerivative(const ParametersType& parameters, DerivativeType& derivative)
  {
    MeasureType value;
    if (!m_GPUEnabled)
    {
      Superclass::GetValueAndDerivative(parameters, value, derivative);
    }
    else
    {
      // TODO: Something that needs to be done before?
      GPUGetValueAndDerivative(parameters, value, derivative);
    }
  }

  template<typename TFixedImage, typename TMovingImage, typename TParentImageToImageMetric>
  void
  CudaImageToImageMetric<TFixedImage, TMovingImage, TParentImageToImageMetric>::GPUGetValueAndDerivative(const ParametersType& parameters, MeasureType& value, DerivativeType& derivative) const
  {
    if (!m_GPUEnabled)
    {
      Superclass::GetValueAndDerivative(parameters, value, derivative);
    }
    else
    {
      // TODO: Something that needs to be done before?
      GPUGetValueAndDerivative(parameters, value, derivative);
    }
  }

  /**
   * Compute the gradient image and assign it to m_GradientImage.
   */
  template< typename TFixedImage, typename TMovingImage, typename TParentImageToImageMetric >
  void
  CudaImageToImageMetric< TFixedImage, TMovingImage, TParentImageToImageMetric >::ComputeGradient()
  {
    GradientImageFilterPointer gradientFilter = GradientImageFilterType::New();

    gradientFilter->SetInput(this->m_MovingImage);

    const typename MovingImageType::SpacingType & spacing = this->m_MovingImage->GetSpacing();
    double maximumSpacing = 0.0;
    for (unsigned int i = 0; i < MovingImageDimension; i++)
    {
      if (spacing[i] > maximumSpacing)
      {
        maximumSpacing = spacing[i];
      }
    }

    gradientFilter->SetUseImageDirection(true);
    gradientFilter->Update();

    m_GradientImage = gradientFilter->GetOutput();
  }


  /**
   * PrintSelf
   */
  template <typename TFixedImage, typename TMovingImage, typename TParentImageToImageMetric>
  void
    CudaImageToImageMetric<TFixedImage, TMovingImage, TParentImageToImageMetric>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf(os, indent);
    os << indent << "GPU: " << (m_GPUEnabled ? "Enabled" : "Disabled") << std::endl;
  }

} // end namespace itk

#endif
