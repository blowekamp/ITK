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
#ifndef __itkNormalizedCorrelationImageToImageMetric_h
#define __itkNormalizedCorrelationImageToImageMetric_h

#include "itkImageToImageMetric.h"
#include "itkPoint.h"

namespace itk
{
/** \class NormalizedCorrelationImageToImageMetric
 * \brief Computes similarity between two images to be registered
 *
 * This metric computes the correlation between pixels in the fixed image
 * and pixels in the moving image. The spatial correspondance between
 * fixed and moving image is established through a Transform. Pixel values are
 * taken from the fixed image, their positions are mapped to the moving
 * image and result in general in non-grid position on it. Values at these
 * non-grid position of the moving image are interpolated using a user-selected
 * Interpolator. The correlation is normalized by the autocorrelations of both
 * the fixed and moving images.
 *
 * \ingroup RegistrationMetrics
 */
template< class TFixedImage, class TMovingImage >
class ITK_EXPORT NormalizedCorrelationImageToImageMetric:
  public ImageToImageMetric< TFixedImage, TMovingImage >
{
public:

  /** Standard class typedefs. */
  typedef NormalizedCorrelationImageToImageMetric         Self;
  typedef ImageToImageMetric< TFixedImage, TMovingImage > Superclass;
  typedef SmartPointer< Self >                            Pointer;
  typedef SmartPointer< const Self >                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NormalizedCorrelationImageToImageMetric, ImageToImageMetric);

  /** Types transferred from the base class */
  typedef typename Superclass::RealType                RealType;
  typedef typename Superclass::TransformType           TransformType;
  typedef typename Superclass::TransformPointer        TransformPointer;
  typedef typename Superclass::TransformParametersType TransformParametersType;
  typedef typename Superclass::TransformJacobianType   TransformJacobianType;
  typedef typename Superclass::GradientPixelType       GradientPixelType;
  typedef typename Superclass::OutputPointType         OutputPointType;
  typedef typename Superclass::InputPointType          InputPointType;

  typedef typename Superclass::MeasureType             MeasureType;
  typedef typename Superclass::DerivativeType          DerivativeType;
  typedef typename Superclass::ImageDerivativesType    ImageDerivativesType;
  typedef typename Superclass::FixedImageType          FixedImageType;
  typedef typename Superclass::FixedImagePointType     FixedImagePointType;
  typedef typename Superclass::MovingImageType         MovingImageType;
  typedef typename Superclass::MovingImagePointType    MovingImagePointType;
  typedef typename Superclass::FixedImageConstPointer  FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer MovingImageConstPointer;

/** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);


  /**
   *  Initialize the Metric by
   *  (1) making sure that all the components are present and plugged
   *      together correctly,
   *  (2) uniformly select NumberOfSpatialSamples within
   *      the FixedImageRegion, and
   *  (3) allocate memory for pdf data structures. */
  virtual void Initialize(void) throw ( ExceptionObject );

  /** Get the derivatives of the match measure. */
  void GetDerivative(const TransformParametersType & parameters,
                     DerivativeType & Derivative) const;

  /**  Get the value for single valued optimizers. */
  MeasureType GetValue(const TransformParametersType & parameters) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative(const TransformParametersType & parameters,
                             MeasureType & Value, DerivativeType & Derivative) const;

  /** Set/Get SubtractMean boolean. If true, the sample mean is subtracted
   * from the sample values in the cross-correlation formula and
   * typically results in narrower valleys in the cost fucntion.
   * Default value is false. */
  itkSetMacro(SubtractMean, bool);
  itkGetConstReferenceMacro(SubtractMean, bool);
  itkBooleanMacro(SubtractMean);
protected:
  NormalizedCorrelationImageToImageMetric();
  virtual ~NormalizedCorrelationImageToImageMetric();
  void PrintSelf(std::ostream & os, Indent indent) const;

private:
  NormalizedCorrelationImageToImageMetric(const Self &); //purposely not
                                                         // implemented
  void operator=(const Self &);                          //purposely not
                                                         // implemented


   inline bool GetValueThreadProcessSample(unsigned int threadID,
                                          unsigned long fixedImageSample,
                                          const MovingImagePointType & mappedPoint,
                                          double movingImageValue) const;

   inline bool GetValueAndDerivativeThreadProcessSample(unsigned int threadID,
                                                        unsigned long fixedImageSample,
                                                        const MovingImagePointType & mappedPoint,
                                                        double movingImageValue,
                                                        const ImageDerivativesType &
                                                        movingImageGradientValue) const;

  typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;

  bool m_SubtractMean;

  DerivativeType *m_ThreaderMSEDerivatives;

  class MutableThreaderData
  {
  public:
    // per thread sums of fix and moving image values
    std::vector<AccumulateType> m_ThreaderSFF;
    std::vector<AccumulateType> m_ThreaderSMM;
    std::vector<AccumulateType> m_ThreaderSFM;
    std::vector<AccumulateType> m_ThreaderSF;
    std::vector<AccumulateType> m_ThreaderSM;

    std::vector<DerivativeType> m_ThreaderDerivativeF;
    std::vector<DerivativeType> m_ThreaderDerivativeM;
    std::vector<AccumulateType> m_ThreaderDerivativeD;
  };

  // threader data
  MutableThreaderData *m_TD;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNormalizedCorrelationImageToImageMetric.txx"
#endif

#endif
