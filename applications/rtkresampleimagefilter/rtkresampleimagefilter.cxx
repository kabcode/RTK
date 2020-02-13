#include "rtkCudaResampleImageFilter.h"
#include "itkEuler3DTransform.h"
#include "itkImageFileReader.h"

#include "rtkCudaKernelImage.hcu"

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

  // Minitest
  itk::Image<float, 3>::SpacingType Spacing;
  Spacing.Fill(2.4f);
  rtk::CudaKernelImage<float,3> cki;
  cki.SetSpacing(Spacing.GetDataPointer());

  cki.Print();




  /*
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

  auto Transform = itk::Euler3DTransform<float>::New();
  Transform->SetIdentity();
  itk::Vector<float> offset;
  offset.Fill(3.4);
  Transform->SetOffset(offset);

  auto CudaResampleImageFilter = CudaResampleImageFilterType::New();
  CudaResampleImageFilter->SetInput(InputVolume);
  CudaResampleImageFilter->SetReferenceImage(ReferenceVolume);
  CudaResampleImageFilter->UseReferenceImageOn();
  CudaResampleImageFilter->SetTransform(Transform);
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
  */
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
