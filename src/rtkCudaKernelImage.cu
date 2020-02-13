

#include "rtkCudaKernelImage.hcu"

namespace rtk
{
  template<class PixelType, unsigned int TImageDimension>
  void CudaKernelImage<PixelType, TImageDimension>::Print()
  {
    printf("All the information");
    printf("Spacing: %f, %f, %f", spacing[0], spacing[1], spacing[2]);
  }

  
  template<class PixelType, unsigned int TImageDimension>
  void CudaKernelImage<PixelType, TImageDimension>::SetSpacing(std::vector<PixelType> spacing)
  {
    //for (auto i = 0; i < TImageDimension; ++i)
      //this->spacing[i] = spacing[i];
  }

}

template class rtk::CudaKernelImage<float, 2>;
template class rtk::CudaKernelImage<float, 3>;
