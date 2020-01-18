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
#ifndef itkCudaGradientImageFilter_hxx
#define itkCudaGradientImageFilter_hxx

//Conditional definition of the class to pass ITKHeaderTest
#include "itkCudaGradientImageFilter.h"
#include "itkCudaGradientImageFilter.hcu"

#include "itkConstantBoundaryCondition.h"

namespace itk
{
    
    template <typename TInputImage, typename TOperatorValueType, typename TOutputValueType, typename TOutputImageType>
    CudaGradientImageFilter<TInputImage, TOperatorValueType, TOutputValueType, TOutputImageType>::
    CudaGradientImageFilter() :
    m_UseImageSpacing(true)
    {
        m_UnsignedBoundaryCondition = BoundaryCondition::ZeroFluxNeumann; // default is ZeroFluxNeumannBoundaryCondition
    }

    template <typename TInputImage, typename TOperatorValueType, typename TOutputValueType, typename TOutputImageType>
    void CudaGradientImageFilter<TInputImage, TOperatorValueType, TOutputValueType, TOutputImageType>::
    ChangeBoundaryCondition(ImageBoundaryCondition<TInputImage>* boundaryCondition)
    {
        this->OverrideBoundaryCondition(boundaryCondition);
        if(dynamic_cast<ZeroFluxNeumannBoundaryCondition<TInputImage>*>(boundaryCondition))
        {
            m_UnsignedBoundaryCondition = 1;
        }
        if(dynamic_cast<ConstantBoundaryCondition<TInputImage>*>(boundaryCondition))
        {
            m_UnsignedBoundaryCondition = 3;
        }
    }

    template <typename TInputImage, typename TOperatorValueType, typename TOutputValueType, typename TOutputImageType>
    void
    CudaGradientImageFilter<TInputImage, TOperatorValueType, TOutputValueType, TOutputImageType>::
    PrintSelf(std::ostream& os, Indent indent) const
    {
        //GPUSuperclass::PrintSelf(os, indent);
        os << "UseImageSpacing: " << m_UseImageSpacing << std::endl;
        os << "BoundaryCondition: " << m_UnsignedBoundaryCondition << std::endl;
    }

    template <typename TInputImage, typename TOperatorValueType, typename TOutputValueType, typename TOutputImageType>
    void
    CudaGradientImageFilter<TInputImage, TOperatorValueType, TOutputValueType, TOutputImageType>::
    GPUGenerateData()
    {
        
        unsigned int inputSize[OutputImageDimension];
        unsigned int outputSize[OutputImageDimension];
        float inputSpacing[OutputImageDimension];
        float outputSpacing[OutputImageDimension];
        float outputDirection[OutputImageDimension*OutputImageDimension];

        for (unsigned int i = 0; i < OutputImageDimension; i++)
        {
            inputSize[i] = this->GetInput()->GetBufferedRegion().GetSize()[i];
            outputSize[i] = this->GetOutput()->GetBufferedRegion().GetSize()[i];
            inputSpacing[i] = this->GetInput()->GetSpacing()[i];
            outputSpacing[i] = this->GetOutput()->GetSpacing()[i];

            if ((inputSize[i] != outputSize[i]) || (inputSpacing[i] != outputSpacing[i]))
            {
                std::cerr << "CUDAGradientImageFilter only handles input and output regions of equal size and spacing" << std::endl;
                exit(1);
            }
            

            for (unsigned int j = 0; j < OutputImageDimension; ++j)
            {
                {
                    outputDirection[i + j * OutputImageDimension] = this->GetInput()->GetDirection()[i][j];
                }
            }

            
            if (!this->GetUseImageSpacing())
            {
                outputSpacing[i] = 1;
            }
        }

        if(!this->GetUseImageDirection())
        {
            memset(outputDirection, 0, OutputImageDimension*OutputImageDimension * sizeof(float));
            for(auto i = 0; i < OutputImageDimension; ++i)
            {
                outputDirection[i + i * OutputImageDimension] = 1;
            }
        }

        float *pin = *(float**)(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
        auto outputimage = this->GetOutput();
        float *pout = *(float**)(outputimage->GetCudaDataManager()->GetGPUBufferPointer());

        CUDA_gradient(pin, outputSize, outputSpacing, outputDirection, OutputImageDimension, m_UnsignedBoundaryCondition, pout);

    }

} // end namespace rtk

#endif
