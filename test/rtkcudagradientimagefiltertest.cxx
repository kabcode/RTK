#include "rtkTest.h"
#include "rtkCudaGradientImageFilter.h"

#include "itkImageFileReader.h"
#include "itkConstantBoundaryCondition.h"

/**
 * \file rtkcudagradientimagefiltertest.cxx
 *
 * \brief Functional test for gradient computation
 *
 *
 * \author Gordian Kabelitz
 */


int main(int argc, char* argv[] )
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using VectorPixelType = itk::CovariantVector<PixelType, Dimension>;
  using ImageType = itk::CudaImage<PixelType, Dimension>;
  using VectorImageType = itk::CudaImage<VectorPixelType, Dimension>;
  using CudaGradientImageFilterType = rtk::CudaGradientImageFilter<ImageType, float, VectorImageType>;


  auto CudaGradientImageFilter = CudaGradientImageFilterType::New();

  auto BoundaryCondition = new itk::ConstantBoundaryCondition<ImageType>();
  CudaGradientImageFilter->ChangeBoundaryCondition(BoundaryCondition);

  auto FileReader = itk::ImageFileReader<ImageType>::New();
  FileReader->SetFileName(argv[1]);
  auto InputVolume = FileReader->GetOutput();
  InputVolume->Update();

  CudaGradientImageFilter->SetInput(InputVolume);
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

  auto FileWriter = itk::ImageFileWriter<VectorImageType>::New();
  FileWriter->SetInput(CudaGradientImageFilter->GetOutput());
  FileWriter->SetFileName("OUTPUT.nrrd");

  try
  {
    FileWriter->Update();
  }
  catch (itk::ExceptionObject& EO)
  {
    EO.Print(std::cout);
    return EXIT_FAILURE;
  }

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
