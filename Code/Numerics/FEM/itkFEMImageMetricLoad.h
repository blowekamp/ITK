/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit (ITK)
  Module:    itkFEMImageMetricLoad.h Language:  C++
  Date:      $Date$
  Version:   $Revision$

Copyright (c) 2001 Insight Consortium
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * The name of the Insight Consortium, nor the names of any consortium members,
   nor of any contributors, may be used to endorse or promote products derived
   from this software without specific prior written permission.

  * Modified source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/
#ifndef _itkFEMImageMetricLoad_h_
#define _itkFEMImageMetricLoad_h_

#include "itkImage.h"
#include "itkTranslationTransform.h"

#include "itkImageRegionIteratorWithIndex.h"
#include "itkNeighborhoodIterator.h"
#include "itkSmartNeighborhoodIterator.h"
#include "itkNeighborhoodInnerProduct.h"
#include "itkDerivativeOperator.h"
#include "itkForwardDifferenceOperator.h"
#include "itkLinearInterpolateImageFunction.h"
#include "vnl/vnl_math.h"

#include <itkMutualInformationImageToImageMetric.h>
#include <itkMeanSquaresImageToImageMetric.h>
#include <itkNormalizedCorrelationImageToImageMetric.h>
#include <itkPatternIntensityImageToImageMetric.h>


namespace itk 
{
namespace fem
{

/**
 * \class ImageMetricLoad
 * \brief General image pair load that uses the itkImageToImageMetrics.
 *
 * LoadImageMetric computes FEM gravity loads by using derivatives provided 
 * by itkImageToImageMetrics (e.g. mean squares intensity difference.)
 * The function responsible for this is called Fg, as required by the FEMLoad
 * standards.  It takes a vnl_vector as input.
 * We assume the vector input is of size 2*ImageDimension.
 * The 0 to ImageDimension-1 elements contain the position, p,
 * in the reference (moving) image.  The next ImageDimension to 2*ImageDimension-1
 * elements contain the value of the vector field at that point, v(p).
 *
 * Then, we evaluate the derivative at the point p+v(p) with respect to
 * some region of the target (fixed) image by calling the metric with 
 * the translation parameters as provided by the vector field at p.
 * The metrics return both a scalar similarity value and vector-valued derivative.  
 * The derivative is what gives us the force to drive the FEM registration.
 * These values are computed with respect to some region in the Target image.
 * This region size may be set by the user by calling SetTargetRadius.
 * As the metric derivative computation evolves, performance should improve
 * and more functionality will be available (such as scale selection).
 */ 
template<class TReference,class TTarget> 
class ImageMetricLoad : public LoadElement
{
FEM_CLASS(ImageMetricLoad,LoadElement)
public:

// Necessary typedefs for dealing with images BEGIN
  typedef typename LoadElement::Float Float;

  typedef TReference ReferenceType;
  typedef ReferenceType*  ReferencePointer;
  typedef TTarget       TargetType;
  typedef TargetType*  TargetPointer;
  /* Inherit some enums and typedefs from template */
  enum{ ImageDimension = TReference::ImageDimension };
  typedef ImageRegionIteratorWithIndex<ReferenceType> RefRegionIteratorType; 
  typedef ImageRegionIteratorWithIndex<TargetType>    TarRegionIteratorType; 
  
  typedef SmartNeighborhoodIterator<ReferenceType> 
                                     ReferenceNeighborhoodIteratorType; 
  typedef typename ReferenceNeighborhoodIteratorType::IndexType  
                                     ReferenceNeighborhoodIndexType;
  typedef typename ReferenceNeighborhoodIteratorType::RadiusType 
                                     ReferenceRadiusType;
  typedef SmartNeighborhoodIterator<TargetType> 
                                     TargetNeighborhoodIteratorType; 
  typedef typename TargetNeighborhoodIteratorType::IndexType  
                                     TargetNeighborhoodIndexType;
  typedef typename TargetNeighborhoodIteratorType::RadiusType 
                                     TargetRadiusType;

  typedef   Array<Float>  ParametersType;
  

// IMAGE DATA
  typedef   typename  ReferenceType::PixelType RefPixelType;
  typedef   typename  TargetType::PixelType    TarPixelType;
  typedef   Float PixelType;
  typedef   Float ComputationType;
  typedef   CovariantVector< PixelType, ImageDimension >  CovariantVectorType;
  typedef   Image< CovariantVectorType, ImageDimension >  CovariantVectorImageType;
  typedef   Image< RefPixelType, ImageDimension >       RefImageType;
  typedef   Image< TarPixelType, ImageDimension >       TarImageType;
  typedef   Image< PixelType, ImageDimension >            ImageType;
  typedef   vnl_vector<Float>                             VectorType;

// Necessary typedefs for dealing with images END
 
//------------------------------------------------------------
// Set up the metrics
//------------------------------------------------------------
  typedef double                   CoordinateRepresentationType;
  typedef Transform< CoordinateRepresentationType,ImageDimension, ImageDimension > TransformBaseType;
  typedef TranslationTransform<CoordinateRepresentationType,  ImageDimension >  DefaultTransformType;

 /**  Type of supported metrics. */
  typedef   ImageToImageMetric<TargetType,ReferenceType > MetricBaseType;
  typedef typename MetricBaseType::Pointer             MetricBaseTypePointer;

  typedef   MutualInformationImageToImageMetric<  TargetType, ReferenceType  > MutualInformationMetricType;

  typedef   MeanSquaresImageToImageMetric< TargetType, ReferenceType  > MeanSquaresMetricType;

  typedef   NormalizedCorrelationImageToImageMetric< TargetType,  ReferenceType  > NormalizedCorrelationMetricType;

  typedef   PatternIntensityImageToImageMetric<  TargetType, ReferenceType  > PatternIntensityMetricType;

//  typedef typename MutualInformationMetricType             DefaultMetricType;
  typedef typename NormalizedCorrelationMetricType             DefaultMetricType;
//  typedef typename PatternIntensityMetricType             DefaultMetricType;
//  typedef typename MeanSquaresMetricType             DefaultMetricType;
  typedef typename DefaultTransformType::ParametersType         ParametersType;
  typedef typename DefaultTransformType::JacobianType           JacobianType;


//------------------------------------------------------------
// Set up an Interpolator
//------------------------------------------------------------
  typedef LinearInterpolateImageFunction< ReferenceType, double > InterpolatorType;

 
// FUNCTIONS

  /** Set/Get the Metric. FIXME */
//  void SetMetric(MetricBaseTypePointer MP) { m_metric=MP; }; 
  
 /** Define the reference (moving) image. */
  void SetReferenceImage(ReferenceType* R)
  { 
    m_RefImage = R; 
    // GET DATA SIZE  BUG!! FIXME!! MUST BE BETTER WAY TO GET SIZE
    RefRegionIteratorType Iter (m_RefImage,m_RefImage->GetLargestPossibleRegion() );
    Iter.GoToEnd();
    typename ReferenceType::IndexType Ind = Iter.GetIndex();    
    for (int i=0; i< ImageDimension; i++) m_RefSize[i]=Ind[i]+1;
  };

 /** Define the target (fixed) image. */ 
  void SetTargetImage(TargetType* T)
  { 
     m_TarImage=T; 
     // GET DATA SIZE  BUG!! FIXME!! MUST BE BETTER WAY TO GET SIZE
     TarRegionIteratorType Iter (m_TarImage,m_TarImage->GetLargestPossibleRegion() );
     Iter.GoToEnd();
     typename ReferenceType::IndexType Ind = Iter.GetIndex();   
     for (int i=0; i< ImageDimension; i++) m_TarSize[i]=Ind[i]+1; 
  };


  ReferencePointer GetReferenceImage() { return m_RefImage; };
  TargetPointer GetTargetImage() { return m_TarImage; };

  /** Define the reference (moving) image region size. */
  void SetReferenceRadius(ReferenceRadiusType R) {m_RefRadius  = R; };
  
  /** Define the target (fixed) image region size. */ 
  void SetTargetRadius(TargetRadiusType T) {m_TarRadius  = T; };       
  

  void SetSolution(Solution::ConstPointer ptr) {  m_Solution=ptr; }
  Solution::ConstPointer GetSolution() {  return m_Solution; }

  /**
   *  This method returns the total metric evaluated over the image with respect to the current solution.
   */
  Float GetMetric (VectorType  InVec);
  
  // FIXME - WE ASSUME THE 2ND VECTOR (INDEX 1) HAS THE INFORMATION WE WANT
  Float GetSolution(unsigned int i,unsigned int which=0){  return m_Solution->GetSolutionValue(i,which); }
  
  void InitializeMetric(void);
  ImageMetricLoad(); // cannot be private until we always use smart pointers
  Float EvaluateMetricGivenSolution ( Element::ArrayType* el, Float step);
 
/**
 * Compute the image based load - implemented with ITK metric derivatives.
 */
  VectorType Fe1(VectorType);
  VectorType Fe(VectorType,VectorType);
 
  
  static Baseclass* NewImageMetricLoad(void)
  { return new ImageMetricLoad; }

  virtual void Read( std::istream& f, void* info )
  {
    Superclass::Read(f,info);
  }
  
  void Write( std::ostream& f, int clid ) const
  {
    // if not set already, se set the clid
    if (clid<0) clid=CLID;

    // call the parent's write function
    Superclass::Write(f,clid);
  }


  ImageMetricLoad(const ImageMetricLoad& I) 
  { 
    std::cout << " copy " << I.m_RefSize << std::endl;
  };  

protected:

private:  
  
  typename CovariantVectorImageType::Pointer          m_DerivativeImage;
  ReferencePointer                                    m_RefImage;
  TargetPointer                                       m_TarImage;
  ReferenceRadiusType                                 m_RefRadius; /** used by the neighborhood iterator */
  TargetRadiusType                                    m_TarRadius; /** used by the metric to set region size for fixed image*/ 
  typename ReferenceType::SizeType                    m_RefSize;
  typename TargetType::SizeType                       m_TarSize;

  typename Solution::ConstPointer                     m_Solution;
  typename MetricBaseTypePointer                      m_Metric;
  typename TransformBaseType::Pointer                 m_Transform;
  typename InterpolatorType::Pointer                  m_Interpolator;


};




}} // end namespace fem/itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFEMImageMetricLoad.txx"
#endif

#endif
