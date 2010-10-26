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
#ifndef __itkScalarAnisotropicDiffusionFunction_txx
#define __itkScalarAnisotropicDiffusionFunction_txx

#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodInnerProduct.h"
#include "itkNeighborhoodAlgorithm.h"
#include "itkDerivativeOperator.h"

namespace itk
{

template< class TImage >
void
ScalarAnisotropicDiffusionFunction< TImage >
::ThreadedCalculateAverageGradientMagnitudeSquared(const TImage *ip,
                                                   const typename TImage::RegionType &region,
                                                   int threadId,
                                                   typename ScalarAnisotropicDiffusionFunction< TImage >::AccumulateType &outputTotal) const
{
  typedef ConstNeighborhoodIterator< TImage >                           NI_type;
  typedef NeighborhoodAlgorithm::ImageBoundaryFacesCalculator< TImage > BFC_type;

  AccumulateType                             accumulator;
  BFC_type                                   bfc;
  typename BFC_type::FaceListType            faceList;
  typename NI_type::RadiusType               radius;
  typename BFC_type::FaceListType::iterator  fit;

  unsigned long index_p[ImageDimension];
  unsigned long index_n[ImageDimension];

  // need a 1-neighborhood to calculate first derivative
  radius.Fill(1);

  // Get the various region "faces" that are on the data set boundary.
  faceList = bfc(ip, region, radius);
  fit      = faceList.begin();

  // compute first derivative index offsets in the neighborhood of the iterator
  NI_type iterator = NI_type( radius, ip, *fit);
  for ( unsigned int i = 0; i < ImageDimension; ++i )
    {
    const unsigned long center_i = iterator.Size() / 2;
    const unsigned long stride_i = iterator.GetStride(i);
    index_p[i] = center_i - stride_i;
    index_n[i] = center_i + stride_i;
    }

  // Now do the actual processing
  accumulator = NumericTraits< AccumulateType >::Zero;

  // We compute the derivative manually based on a set of offsets
  // around the iterator
  while ( fit != faceList.end() )
    {
    iterator = NI_type( radius, ip, *fit);

    while ( !iterator.IsAtEnd() )
      {
      for ( unsigned int i = 0; i < ImageDimension; ++i )
        {
        PixelRealType val = iterator.GetPixel(index_n[i])
          - iterator.GetPixel(index_p[i]);
        val *= -0.5 * this->m_ScaleCoefficients[i];
        accumulator += val * val;
        }
      ++iterator;
      }
    ++fit;
    }
  outputTotal = accumulator;
}
} // end namespace itk

#endif
