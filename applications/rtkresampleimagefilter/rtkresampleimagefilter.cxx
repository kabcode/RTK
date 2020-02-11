#include "rtkCudaResampleImageFilter.h"

#include "itkImageFileReader.h"

/**
 * \file rtkcudaresampleimagefiltertest.cxx
 *
 * \brief Functional test for resampling images
 *
 *
 * \author Gordian Kabelitz
 */


int main(int argc, char* argv[])
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::CudaImage<PixelType, Dimension>;
  using CudaResampleImageFilterType = rtk::CudaResampleImageFilter<ImageType, ImageType>;

  auto FileReaderInput = itk::ImageFileReader<ImageType>::New();
  FileReaderInput->SetFileName(argv[1]);
  auto InputVolume = FileReaderInput->GetOutput();
  InputVolume->Update();

  auto FileReaderReference = itk::ImageFileReader<ImageType>::New();
  FileReaderReference->SetFileName(argv[2]);
  auto ReferenceVolume = FileReaderReference->GetOutput();
  ReferenceVolume->Update();

  auto CudaResampleImageFilter = CudaResampleImageFilterType::New();
  CudaResampleImageFilter->SetInput(InputVolume);
  CudaResampleImageFilter->SetReferenceImage(ReferenceVolume);
  CudaResampleImageFilter->UseReferenceImageOn();
  // CudaResampleImageFilter->Print(std::cout);
  try
  {
    CudaResampleImageFilter->Update();
  }
  catch (itk::ExceptionObject &EO)
  {
    EO.Print(std::cout);
    return EXIT_FAILURE;
  }

  auto FileWriter = itk::ImageFileWriter<ImageType>::New();
  FileWriter->SetInput(CudaResampleImageFilter->GetOutput());
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
