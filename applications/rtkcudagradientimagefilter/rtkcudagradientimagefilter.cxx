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

#include "rtkforwardprojections_ggo.h"
#include "rtkGgoFunctions.h"

#ifdef RTK_USE_CUDA
	#include "rtkCudaGradientImageFilter.h"
#endif

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkforwardprojections, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  #ifdef RTK_USE_CUDA
    using OutputImageType = itk::CudaImage< OutputPixelType, Dimension >;
  #else
    using OutputImageType = itk::Image< OutputPixelType, Dimension >;
  #endif
  
  return EXIT_SUCCESS;
}
