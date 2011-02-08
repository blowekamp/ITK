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
#ifndef __itkNormalizedCorrelationImageToImageMetric_txx
#define __itkNormalizedCorrelationImageToImageMetric_txx

#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{
/**
 * Constructor
 */
template< class TFixedImage, class TMovingImage >
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::NormalizedCorrelationImageToImageMetric()
{

  this->SetComputeGradient(true);

  m_SubtractMean = false;

  this->m_TD = NULL;
}

/**
 * Destructor
 */
template< class TFixedImage, class TMovingImage >
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::~NormalizedCorrelationImageToImageMetric()
{
  if ( this->m_TD != NULL )
    {
    delete this->m_TD;
    }
  this->m_TD = NULL;
}


/**
 * Initialize
 */
template< class TFixedImage, class TMovingImage >
void
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::Initialize()
throw ( ExceptionObject )
{

  this->Superclass::Initialize();
  this->Superclass::MultiThreadingInitialize();

if ( this->m_TD != NULL )
    {
    delete this->m_TD;
    }
  this->m_TD = new MutableThreaderData;

   this->m_TD->m_ThreaderSFF.resize( this->m_NumberOfThreads );
   this->m_TD->m_ThreaderSMM.resize( this->m_NumberOfThreads );
   this->m_TD->m_ThreaderSFM.resize( this->m_NumberOfThreads );
   this->m_TD->m_ThreaderSF.resize( this->m_NumberOfThreads );
   this->m_TD->m_ThreaderSM.resize( this->m_NumberOfThreads );

   this->m_TD->m_ThreaderDerivativeF.resize( this->m_NumberOfThreads );
   this->m_TD->m_ThreaderDerivativeM.resize( this->m_NumberOfThreads );

  for ( unsigned int threadID = 0; threadID < this->m_NumberOfThreads; threadID++ )
    {
     this->m_TD->m_ThreaderDerivativeM[threadID].SetSize( this->m_NumberOfParameters);
     this->m_TD->m_ThreaderDerivativeF[threadID].SetSize( this->m_NumberOfParameters);
    }

  this->m_TD->m_ThreaderDerivativeD.resize( this->m_NumberOfThreads );
}

template< class TFixedImage, class TMovingImage  >
inline bool
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueThreadProcessSample(unsigned int threadID,
                              unsigned long fixedImageSample,
                              const MovingImagePointType & itkNotUsed(mappedPoint),
                              double movingImageValue) const
{
  this->m_TD->m_ThreaderSFF[threadID] += vnl_math_sqr( this->m_FixedImageSamples[fixedImageSample].value );
  this->m_TD->m_ThreaderSMM[threadID] += vnl_math_sqr( movingImageValue );
  this->m_TD->m_ThreaderSFM[threadID] += this->m_FixedImageSamples[fixedImageSample].value * movingImageValue;
  this->m_TD->m_ThreaderSF[threadID] += this->m_FixedImageSamples[fixedImageSample].value;
  this->m_TD->m_ThreaderSM[threadID] += movingImageValue;

  return true;
}

/**
 * Get the match Measure
 */
template< class TFixedImage, class TMovingImage >
typename NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >::MeasureType
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetValue(const TransformParametersType & parameters) const
{

  itkDebugMacro("GetValue( " << parameters << " ) ");

  if ( !this->m_FixedImage )
    {
    itkExceptionMacro(<< "Fixed image has not been assigned");
    }

  // Set up the parameters in the transform
  this->m_Transform->SetParameters(parameters);
  this->m_Parameters = parameters;

  // fill arrays with zeros
  this->m_TD->m_ThreaderSFF.assign( this->m_NumberOfThreads, 0.0 );
  this->m_TD->m_ThreaderSMM.assign( this->m_NumberOfThreads, 0.0 );
  this->m_TD->m_ThreaderSFM.assign( this->m_NumberOfThreads, 0.0 );
  this->m_TD->m_ThreaderSF.assign( this->m_NumberOfThreads, 0.0 );
  this->m_TD->m_ThreaderSM.assign( this->m_NumberOfThreads, 0.0 );

  // MUST BE CALLED TO INITIATE PROCESSING
  this->GetValueMultiThreadedInitiate();

  itkDebugMacro("Ratio of voxels mapping into moving image buffer: "
                << this->m_NumberOfPixelsCounted << " / "
                << this->m_NumberOfFixedImageSamples
                << std::endl);

  // reduce from the threaded computation by accumulating
  double sff = std::accumulate( this->m_TD->m_ThreaderSFF.begin(), this->m_TD->m_ThreaderSFF.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );
  double smm = std::accumulate( this->m_TD->m_ThreaderSMM.begin(), this->m_TD->m_ThreaderSMM.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );
  double sfm = std::accumulate( this->m_TD->m_ThreaderSFM.begin(), this->m_TD->m_ThreaderSFM.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );
  double sm = std::accumulate( this->m_TD->m_ThreaderSF.begin(), this->m_TD->m_ThreaderSF.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );
  double sf = std::accumulate( this->m_TD->m_ThreaderSM.begin(), this->m_TD->m_ThreaderSM.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );


  double measure = 0.0;

  if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
    {
    sff -= ( sf * sf / this->m_NumberOfPixelsCounted );
    smm -= ( sm * sm / this->m_NumberOfPixelsCounted );
    sfm -= ( sf * sm / this->m_NumberOfPixelsCounted );
    }

  const RealType denom = -1.0 * vcl_sqrt(sff * smm);

  if ( this->m_NumberOfPixelsCounted > 0 && denom != 0.0 )
    {
    measure = sfm / denom;
    }
  else
    {
    measure = NumericTraits< MeasureType >::Zero;
    }

  return measure;
}


/**
 * Get the Derivative Measure
 */
template< class TFixedImage, class TMovingImage >
void
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetDerivative(const TransformParametersType & parameters,
                DerivativeType & derivative) const
{

  if ( !this->m_FixedImage )
    {
    itkExceptionMacro(<< "Fixed image has not been assigned");
    }

  MeasureType value;
  // call the combined version
  this->GetValueAndDerivative(parameters, value, derivative);
}


template< class TFixedImage, class TMovingImage  >
inline bool
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivativeThreadProcessSample(unsigned int threadID,
                                           unsigned long fixedImageSample,
                                           const MovingImagePointType &mappedPoint,
                                           double movingImageValue,
                                           const ImageDerivativesType &gradient) const
{
  double fixedImageValue = this->m_FixedImageSamples[fixedImageSample].value;


  this->m_TD->m_ThreaderSFF[threadID] += vnl_math_sqr( fixedImageValue );
  this->m_TD->m_ThreaderSMM[threadID] += vnl_math_sqr( movingImageValue );
  this->m_TD->m_ThreaderSFM[threadID] += fixedImageValue * movingImageValue;
  this->m_TD->m_ThreaderSF[threadID] += fixedImageValue;
  this->m_TD->m_ThreaderSM[threadID] += movingImageValue;


  FixedImagePointType fixedImagePoint = this->m_FixedImageSamples[fixedImageSample].point;

  // Need to use one of the threader transforms if we're
  // not in thread 0.
  //
  // Use a raw pointer here to avoid the overhead of smart pointers.
  // For instance, Register and UnRegister have mutex locks around
  // the reference counts.
  TransformType *transform;

  if ( threadID > 0 )
    {
    transform = this->m_ThreaderTransform[threadID - 1];
    }
  else
    {
    transform = this->m_Transform;
    }


  // Jacobian should be evaluated at the unmapped (fixed image) point.
  const TransformJacobianType & jacobian = transform->GetJacobian(fixedImagePoint);

  for ( unsigned int par = 0; par < this->m_NumberOfParameters; par++ )
    {
    RealType sumF = NumericTraits< RealType >::Zero;
    RealType sumM = NumericTraits< RealType >::Zero;
    RealType sumD = NumericTraits< RealType >::Zero;

    for ( unsigned int dim = 0; dim < MovingImageDimension; dim++ )
      {
      const RealType differential = jacobian(dim, par) * gradient[dim];
      sumF += fixedImageValue  * differential;
      sumM += movingImageValue * differential;
      sumD += differential;

      }
    this->m_TD->m_ThreaderDerivativeF[threadID][par] += sumF;
    this->m_TD->m_ThreaderDerivativeM[threadID][par] += sumM;
    this->m_TD->m_ThreaderDerivativeD[threadID] += sumD;
    }


  return true;
}

/*
 * Get both the match Measure and theDerivative Measure
 */
template< class TFixedImage, class TMovingImage >
void
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType & value, DerivativeType  & derivative) const
{
 if ( !this->m_FixedImage )
    {
    itkExceptionMacro(<< "Fixed image has not been assigned");
    }

  // Set up the parameters in the transform
  this->m_Transform->SetParameters(parameters);
  this->m_Parameters = parameters;

  // Set output values to zero
  if ( derivative.GetSize() != this->m_NumberOfParameters )
    {
    derivative = DerivativeType(this->m_NumberOfParameters);
    }
  derivative.Fill( 0.0 );

  // fill arrays with zeros
  this->m_TD->m_ThreaderSFF.assign( this->m_NumberOfThreads, 0.0 );
  this->m_TD->m_ThreaderSMM.assign( this->m_NumberOfThreads, 0.0 );
  this->m_TD->m_ThreaderSFM.assign( this->m_NumberOfThreads, 0.0 );
  this->m_TD->m_ThreaderSF.assign( this->m_NumberOfThreads, 0.0 );
  this->m_TD->m_ThreaderSM.assign( this->m_NumberOfThreads, 0.0 );

  // fill derivatives
  for ( unsigned int threadID = 0; threadID < this->m_NumberOfThreads; threadID++ )
    {
    this->m_TD->m_ThreaderDerivativeF[threadID].Fill( 0.0 );
    this->m_TD->m_ThreaderDerivativeM[threadID].Fill( 0.0 );
    }

  this->m_TD->m_ThreaderDerivativeD.assign( this->m_NumberOfThreads, 0.0 );

  // MUST BE CALLED TO INITIATE PROCESSING
  this->GetValueAndDerivativeMultiThreadedInitiate();


  itkDebugMacro("Ratio of voxels mapping into moving image buffer: "
                << this->m_NumberOfPixelsCounted << " / "
                << this->m_NumberOfFixedImageSamples
                << std::endl);

  // reduce from the threaded computation by accumulating
  double sff = std::accumulate( this->m_TD->m_ThreaderSFF.begin(), this->m_TD->m_ThreaderSFF.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );
  double smm = std::accumulate( this->m_TD->m_ThreaderSMM.begin(), this->m_TD->m_ThreaderSMM.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );
  double sfm = std::accumulate( this->m_TD->m_ThreaderSFM.begin(), this->m_TD->m_ThreaderSFM.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );
  double sm = std::accumulate( this->m_TD->m_ThreaderSF.begin(), this->m_TD->m_ThreaderSF.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );
  double sf = std::accumulate( this->m_TD->m_ThreaderSM.begin(), this->m_TD->m_ThreaderSM.end(), itk::NumericTraits<AccumulateType>::ZeroValue() );


  const unsigned int ParametersDimension = this->GetNumberOfParameters();

  // accumulate derivative arrays
  DerivativeType derivativeF(ParametersDimension);
  DerivativeType derivativeM (ParametersDimension);
  for ( unsigned int i = 0; i < ParametersDimension; i++ )
    {
    derivativeF[i] = NumericTraits< typename DerivativeType::ValueType >::Zero;
    derivativeM[i] = NumericTraits< typename DerivativeType::ValueType >::Zero;

    for ( unsigned int threadID = 0; threadID < this->m_NumberOfThreads; threadID++ )
      {
      derivativeF[i] += this->m_TD->m_ThreaderDerivativeF[threadID][i];
      derivativeM[i] += this->m_TD->m_ThreaderDerivativeM[threadID][i];
      }
    }

  RealType differential = std::accumulate( this->m_TD->m_ThreaderDerivativeD.begin(),
                                           this->m_TD->m_ThreaderDerivativeD.end(),
                                           itk::NumericTraits<AccumulateType>::ZeroValue() );


  if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
    {
    sff -= ( sf * sf / this->m_NumberOfPixelsCounted );
    smm -= ( sm * sm / this->m_NumberOfPixelsCounted );
    sfm -= ( sf * sm / this->m_NumberOfPixelsCounted );

    for ( unsigned int i = 0; i < ParametersDimension; i++ )
      {
      derivativeF[i] -= differential * sf / this->m_NumberOfPixelsCounted;
      derivativeM[i] -= differential * sm / this->m_NumberOfPixelsCounted;
      }
    }

  const RealType denom = -1.0 * vcl_sqrt(sff * smm);

  if ( this->m_NumberOfPixelsCounted > 0 && denom != 0.0 )
    {
    for ( unsigned int i = 0; i < ParametersDimension; i++ )
      {
      derivative[i] = ( derivativeF[i] - ( sfm / smm ) * derivativeM[i] ) / denom;
      }
    value = sfm / denom;
    }
  else
    {
    for ( unsigned int i = 0; i < ParametersDimension; i++ )
      {
      derivative[i] = NumericTraits< MeasureType >::Zero;
      }
    value = NumericTraits< MeasureType >::Zero;
    }

}

template< class TFixedImage, class TMovingImage >
void
NormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "SubtractMean: " << m_SubtractMean << std::endl;
}
} // end namespace itk

#endif
