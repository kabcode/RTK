#include "rtkTest.h"
#include "rtkCudaGradientImageFilter.h"
#include "itkConstantBoundaryCondition.h"

/**
 * \file rtkcudagradientimagefiltertest.cxx
 *
 * \brief Functional test for gradient computation
 *
 *
 * \author Gordian Kabelitz
 */


int main(int , char** )
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using VectorPixelType = itk::CovariantVector<PixelType, Dimension>;
  using ImageType = itk::CudaImage<PixelType, Dimension>;
  using VectorImageType = itk::CudaImage<VectorPixelType, Dimension>;
  using CudaGradientImageFilterType = rtk::CudaGradientImageFilter<ImageType, float, VectorImageType>;


  auto CudaGradientImageFilter = CudaGradientImageFilterType::New();

  itk::ConstantBoundaryCondition<ImageType> BoundaryCondition;
  CudaGradientImageFilter->ChangeBoundaryCondition(&BoundaryCondition);

  auto Inputimage = ImageType::New();
  auto InputVolume = ImageType::New();
  CudaGradientImageFilter->SetInput(0, Inputimage);
  CudaGradientImageFilter->SetInput(1, InputVolume);

  CudaGradientImageFilter->Print(std::cout);
  try
  {
    CudaGradientImageFilter->Update();
  }
  catch (itk::ExceptionObject &EO)
  {
    EO.Print(std::cout);
    return EXIT_FAILURE;
  }
  
  

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
