
#include "itkCudaGradientImageFilter.h"

using Pixeltype = float;
const unsigned int Dimension = 3;

using CudaImageType = itk::CudaImage<Pixeltype, Dimension>;

int itkCudaGradientImageFilterTest(int argc, char* argv[])
{
  auto CudaImageToImageMetric = itk::CudaGradientImageFilter<CudaImageType>::New();

  auto FixedImage = CudaImageType::New();
  auto MovingImage = CudaImageType::New();

  std::cout << CudaImageToImageMetric->GetNameOfClass() << std::endl;

  return EXIT_SUCCESS;
}
