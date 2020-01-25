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

#  include "itkGradientImageFilter.h"
#  include "itkCudaImageToImageFilter.h"
#  include "itkCudaUtil.h"
#  include "itkCudaKernelManager.h"
#  include "RTKExport.h"

/** \class CudaGradientImageFilter
 * \brief Trilinear interpolation forward projection implemented in CUDA
 *
 * \author Gordian Kabelitz
 *
 * \ingroup RTK GradientImageFilter CudaImageToImageFilter
 */

enum class BoundaryConditions {
  ZeroFluxNeumann,
  Periodic,
  Constant
};

namespace rtk
{

/** Create a helper Cuda Kernel class for CudaImageOps */
itkCudaKernelClassMacro(rtkCudaGradientImageFilterKernel);

template <class TInputImage = itk::CudaImage<float, 3>, class TPixelType = float, class TOutputImage = itk::CudaImage<itk::CovariantVector<float,3>>>
class ITK_EXPORT CudaGradientImageFilter
  : public itk::
      CudaImageToImageFilter<TInputImage, TOutputImage, itk::GradientImageFilter<TInputImage, TPixelType, float, TOutputImage>>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaGradientImageFilter);

  /** Standard class type alias. */
  using Self = CudaGradientImageFilter;
  using Superclass = itk::GradientImageFilter<TInputImage, TPixelType, float, TOutputImage>;
  using GPUSuperclass = itk::CudaImageToImageFilter<TInputImage, TOutputImage, Superclass>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using VectorType = itk::Vector<float, 3>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaGradientImageFilter, ImageToImageFilter);

  /** Set step size along ray (in mm). Default is 1 mm. */
  itkSetMacro(UseImageSpacing, bool);
  itkGetConstMacro(UseImageSpacing, bool);
  itkBooleanMacro(UseImageSpacing);

  void ChangeBoundaryCondition(itk::ImageBoundaryCondition< TInputImage >* boundaryCondition);

protected:
  CudaGradientImageFilter();
  ~CudaGradientImageFilter() = default;

  virtual void GPUGenerateData();

private:
  bool m_UseImageSpacing;
  BoundaryConditions m_CudaBoundaryConditions;

}; // end of class

} // end namespace rtk

#  ifndef ITK_MANUAL_INSTANTIATION
#    include "rtkCudaGradientImageFilter.hxx"
#  endif

#endif // end conditional definition of the class

#endif
