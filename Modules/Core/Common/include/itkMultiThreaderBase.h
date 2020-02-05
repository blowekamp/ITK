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
/*=========================================================================
 *
 *  Portions of this file are subject to the VTK Toolkit Version 3 copyright.
 *
 *  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 *
 *  For complete copyright, license and disclaimer of warranty information
 *  please refer to the NOTICE file at the top of the ITK source tree.
 *
 *=========================================================================*/
#ifndef itkMultiThreaderBase_h
#define itkMultiThreaderBase_h

#include "itkObject.h"
#include "itkThreadSupport.h"
#include "itkObjectFactory.h"
#include "itkIntTypes.h"
#include "itkImageRegion.h"
#include "itkImageIORegion.h"
#include "itkSingletonMacro.h"
#include <functional>
#include <thread>


namespace itk
{
/** \class MultiThreaderBase
 * \brief A class for performing multithreaded execution
 *
 * Multithreaders are a class hierarchy that provides support for
 * multithreaded execution by abstracting away platform-specific
 * details. This class can be used to execute a single
 * method on multiple threads or to parallelize an operation over a
 * given image region or array.
 *
 * \ingroup OSSystemObjects
 *
 * \ingroup ITKCommon
 */

struct MultiThreaderBaseGlobals;
class ProcessObject;

class ITKCommon_EXPORT MultiThreaderBase : public Object
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(MultiThreaderBase);

  /** Standard class type aliases. */
  using Self = MultiThreaderBase;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  static Pointer
  New();

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiThreaderBase, Object);

  /** Get/Set the number of threads to use. It will be clamped to the range
   * [ 1, m_GlobalMaximumNumberOfThreads ], so the caller of this method should
   * check that the requested number of threads was accepted. */
  virtual void
  SetMaximumNumberOfThreads(ThreadIdType numberOfThreads);
  itkGetConstMacro(MaximumNumberOfThreads, ThreadIdType);

  /** Get/Set the number of work units to create. It might be clamped to the range
   * [ 1, SomeMaximumNumber ], so the caller of this method should
   * check that the requested number of work units was accepted. */
  virtual void
  SetNumberOfWorkUnits(ThreadIdType numberOfWorkUnits);
  itkGetConstMacro(NumberOfWorkUnits, ThreadIdType);

  /** Set/Get the maximum number of threads to use when multithreading.  It
   * will be clamped to the range [ 1, ITK_MAX_THREADS ] because several arrays
   * are already statically allocated using the ITK_MAX_THREADS number.
   * Therefore the caller of this method should check that the requested number
   * of threads was accepted. */
  static void
  SetGlobalMaximumNumberOfThreads(ThreadIdType val);
  static ThreadIdType
  GetGlobalMaximumNumberOfThreads();

  /** Set/Get whether to use the to use the thread pool
   * implementation or the spawning implementation of
   * starting threads.
   *
   * Deprecated: use Get/Set GlobalDefaultThreader. */
  itkLegacyMacro(static void SetGlobalDefaultUseThreadPool(const bool GlobalDefaultUseThreadPool));
  itkLegacyMacro(static bool GetGlobalDefaultUseThreadPool());

  /** Currently supported types of multi-threader implementations.
   * Last will change with additional implementations. */
  enum class ThreaderEnum : int8_t
  {
    Platform = 0,
    First = Platform,
    Pool,
    TBB,
    Last = TBB,
    Unknown = -1
  };
#if !defined(ITK_LEGACY_REMOVE)
  using ThreaderType = ThreaderEnum;
#endif

  /** Convert a threader name into its enum type. */
  static ThreaderEnum
  ThreaderTypeFromString(std::string threaderString);

  /** Convert a threader enum type into a string for displaying or logging. */
  static std::string
  ThreaderTypeToString(ThreaderEnum threader)
  {
    switch (threader)
    {
      case ThreaderEnum::Platform:
        return "Platform";
        break;
      case ThreaderEnum::Pool:
        return "Pool";
        break;
      case ThreaderEnum::TBB:
        return "TBB";
        break;
      case ThreaderEnum::Unknown:
      default:
        return "Unknown";
        break;
    }
  }

  /** Set/Get whether the default multi-threader implementation.
   *
   * The default multi-threader type is picked up from ITK_GLOBAL_DEFAULT_THREADER
   * environment variable. Example ITK_GLOBAL_DEFAULT_THREADER=TBB
   * A deprecated ITK_USE_THREADPOOL environment variable is also examined,
   * but it can only choose Pool or Platform multi-threader.
   * Platform multi-threader should be avoided,
   * as it introduces the biggest overhead.
   *
   * If the SetGlobalDefaultThreaderType API is ever used by the developer,
   * the developer's choice is respected over the environment variables. */
  static void
  SetGlobalDefaultThreader(ThreaderEnum threaderType);
  static ThreaderEnum
  GetGlobalDefaultThreader();

  /** Set/Get the value which is used to initialize the NumberOfThreads in the
   * constructor.  It will be clamped to the range [1, m_GlobalMaximumNumberOfThreads ].
   * Therefore the caller of this method should check that the requested number
   * of threads was accepted. */
  static void
  SetGlobalDefaultNumberOfThreads(ThreadIdType val);
  static ThreadIdType
  GetGlobalDefaultNumberOfThreads();

#if !defined(ITK_LEGACY_REMOVE)
  /** Get/Set the number of threads to use.
   * DEPRECATED! Use WorkUnits and MaximumNumberOfThreads instead. */
  itkLegacyMacro(virtual void SetNumberOfThreads(ThreadIdType numberOfThreads))
  {
    this->SetMaximumNumberOfThreads(numberOfThreads);
    this->SetNumberOfWorkUnits(this->GetMaximumNumberOfThreads()); // Might be clamped
  }
  itkLegacyMacro(virtual ThreadIdType GetNumberOfThreads()) { return this->GetNumberOfWorkUnits(); }

  /** This is the structure that is passed to the thread that is
   * created from the SingleMethodExecute. It is passed in as a void *,
   * and it is up to the method to cast correctly and extract the information.
   * The ThreadID is a number between 0 and NumberOfThreads-1 that
   * indicates the id of this thread. The UserData is the
   * (void *)arg passed into the SetSingleMethod.
   *
   * DEPRECATED! Use WorkUnitInfo instead. */
  // clang-format off
ITK_GCC_PRAGMA_DIAG_PUSH()
ITK_GCC_PRAGMA_DIAG(ignored "-Wattributes")
INTEL_PRAGMA_WARN_PUSH
INTEL_SUPPRESS_warning_1292
CLANG_PRAGMA_PUSH
CLANG_SUPPRESS_Wc__14_extensions
  // clang-format on
#  ifdef ITK_LEGACY_SILENT
    struct ThreadInfoStruct
#  else
    struct [[deprecated("Use WorkUnitInfo, ThreadInfoStruct is deprecated since ITK 5.0")]] ThreadInfoStruct
#  endif
      // clang-format off
CLANG_PRAGMA_POP
INTEL_PRAGMA_WARN_POP
  // clang-format on
  {
    ThreadIdType       ThreadID;
    ThreadIdType       NumberOfThreads;
    void *             UserData;
    ThreadFunctionType ThreadFunction;
    enum
    {
      SUCCESS,
      ITK_EXCEPTION,
      ITK_PROCESS_ABORTED_EXCEPTION,
      STD_EXCEPTION,
      UNKNOWN
    } ThreadExitCode;
  };
  // clang-format off
ITK_GCC_PRAGMA_DIAG_POP()
  // clang-format on
#endif // ITK_LEGACY_REMOVE

  /** This is the structure that is passed to the thread that is
   * created from the SingleMethodExecute. It is passed in as a void *,
   * and it is up to the method to cast correctly and extract the information.
   * The WorkUnitID is a number between 0 and NumberOfWorkUnits-1 that
   * indicates the id of this work unit. The UserData is the
   * (void *)arg passed into the SetSingleMethod. */
  struct WorkUnitInfo
  {
    ThreadIdType       WorkUnitID;
    ThreadIdType       NumberOfWorkUnits;
    void *             UserData;
    ThreadFunctionType ThreadFunction;
    enum class ThreadExitCodeType : uint8_t
    {
      SUCCESS,
      ITK_EXCEPTION,
      ITK_PROCESS_ABORTED_EXCEPTION,
      STD_EXCEPTION,
      UNKNOWN
    } ThreadExitCode;
#if !defined(ITK_LEGACY_REMOVE)
    static constexpr ThreadExitCodeType SUCCESS = ThreadExitCodeType::SUCCESS;
    static constexpr ThreadExitCodeType ITK_EXCEPTION = ThreadExitCodeType::ITK_EXCEPTION;
    static constexpr ThreadExitCodeType ITK_PROCESS_ABORTED_EXCEPTION =
      ThreadExitCodeType::ITK_PROCESS_ABORTED_EXCEPTION;
    static constexpr ThreadExitCodeType STD_EXCEPTION = ThreadExitCodeType::STD_EXCEPTION;
    static constexpr ThreadExitCodeType UNKNOWN = ThreadExitCodeType::UNKNOWN;
#endif
  };

  /** Execute the SingleMethod (as define by SetSingleMethod) using
   * m_NumberOfWorkUnits threads. As a side effect the m_NumberOfWorkUnits will be
   * checked against the current m_GlobalMaximumNumberOfThreads and clamped if
   * necessary. */
  virtual void
  SingleMethodExecute() = 0;

  /** Set the SingleMethod to f() and the UserData field of the
   * WorkUnitInfo that is passed to it will be data. This method
   * must be of type itkThreadFunctionType and must take a single argument of
   * type void *. */
  virtual void
  SetSingleMethod(ThreadFunctionType, void * data) = 0;

  template <unsigned int VDimension>
  using TemplatedThreadingFunctorType = std::function<void(const ImageRegion<VDimension> &)>;
  using ThreadingFunctorType = std::function<void(const IndexValueType index[], const SizeValueType size[])>;
  using ArrayThreadingFunctorType = std::function<void(SizeValueType)>;

  /** Parallelize an operation over an array. If filter argument is not nullptr,
   * this function will update its progress as each index is completed.
   *
   * This implementation simply delegates parallelization to the old interface
   * SetSingleMethod+SingleMethodExecute. This method is meant to be overloaded! */
  virtual void
  ParallelizeArray(SizeValueType             firstIndex,
                   SizeValueType             lastIndexPlus1,
                   ArrayThreadingFunctorType aFunc,
                   ProcessObject *           filter);

  /** Break up region into smaller chunks, and call the function with chunks as parameters.
   * If filter argument is not nullptr, this function will update its progress
   * as each work unit is completed. Delegates work to non-templated version. */
  template <unsigned int VDimension>
  ITK_TEMPLATE_EXPORT void
  ParallelizeImageRegion(const ImageRegion<VDimension> &           requestedRegion,
                         TemplatedThreadingFunctorType<VDimension> funcP,
                         ProcessObject *                           filter)
  {
    this->ParallelizeImageRegion(
      VDimension,
      requestedRegion.GetIndex().m_InternalArray,
      requestedRegion.GetSize().m_InternalArray,
      [funcP](const IndexValueType index[], const SizeValueType size[]) {
        ImageRegion<VDimension> region;
        for (unsigned int d = 0; d < VDimension; ++d)
        {
          region.SetIndex(d, index[d]);
          region.SetSize(d, size[d]);
        }
        funcP(region);
      },
      filter);
  }

  /** Similar to ParallelizeImageRegion, but do not split the region along one
   * of the directions. If VDimension is 1, restrictedDirection is ignored
   * and no parallelization occurs. */
  template <unsigned int VDimension>
  ITK_TEMPLATE_EXPORT void
  ParallelizeImageRegionRestrictDirection(unsigned int                              restrictedDirection,
                                          const ImageRegion<VDimension> &           requestedRegion,
                                          TemplatedThreadingFunctorType<VDimension> funcP,
                                          ProcessObject *                           filter)
  {
    if (VDimension <= 1) // Cannot split, no parallelization
    {
      MultiThreaderBase::HandleFilterProgress(filter, 0.0f);
      funcP(requestedRegion);
      MultiThreaderBase::HandleFilterProgress(filter, 1.0f);
    }
    else // Can split, parallelize!
    {
      constexpr unsigned int SplitDimension = (VDimension - 1) ? VDimension - 1 : VDimension;
      using SplitRegionType = ImageRegion<SplitDimension>;

      SplitRegionType splitRegion;
      for (unsigned int splitDimension = 0, dimension = 0; dimension < VDimension; ++dimension)
      {
        if (dimension == restrictedDirection)
        {
          continue;
        }
        splitRegion.SetIndex(splitDimension, requestedRegion.GetIndex(dimension));
        splitRegion.SetSize(splitDimension, requestedRegion.GetSize(dimension));
        ++splitDimension;
      }

      this->ParallelizeImageRegion(
        SplitDimension,
        splitRegion.GetIndex().m_InternalArray,
        splitRegion.GetSize().m_InternalArray,
        [&](const IndexValueType index[], const SizeValueType size[]) {
          ImageRegion<VDimension> restrictedRequestedRegion;
          restrictedRequestedRegion.SetIndex(restrictedDirection, requestedRegion.GetIndex(restrictedDirection));
          restrictedRequestedRegion.SetSize(restrictedDirection, requestedRegion.GetSize(restrictedDirection));
          for (unsigned int splitDimension = 0, dimension = 0; dimension < VDimension; ++dimension)
          {
            if (dimension == restrictedDirection)
            {
              continue;
            }
            restrictedRequestedRegion.SetIndex(dimension, index[splitDimension]);
            restrictedRequestedRegion.SetSize(dimension, size[splitDimension]);
            ++splitDimension;
          }
          funcP(restrictedRequestedRegion);
        },
        filter);
    }
  }

  /** Break up region into smaller chunks, and call the function with chunks as parameters.
   *  This overload does the actual work and should be implemented by derived classes. */
  virtual void
  ParallelizeImageRegion(unsigned int         dimension,
                         const IndexValueType index[],
                         const SizeValueType  size[],
                         ThreadingFunctorType funcP,
                         ProcessObject *      filter);

  /** Updates progress if progress is non-negative and checks for abort.
   * If "abort generate data" is set, throws the ProcessAborted exception. */
  static void
  HandleFilterProgress(ProcessObject * filter, float progress = -1.0f);

protected:
  MultiThreaderBase();
  ~MultiThreaderBase() override;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  struct ArrayCallback
  {
    ArrayThreadingFunctorType  functor;
    const SizeValueType        firstIndex;
    const SizeValueType        lastIndexPlus1;
    ProcessObject *            filter;
    std::thread::id            callingThread;
    std::atomic<SizeValueType> progress;
  };

  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ParallelizeArrayHelper(void * arg);

  struct RegionAndCallback
  {
    ThreadingFunctorType       functor;
    unsigned int               dimension;
    const IndexValueType *     index;
    const SizeValueType *      size;
    ProcessObject *            filter;
    std::thread::id            callingThread;
    SizeValueType              pixelCount;
    std::atomic<SizeValueType> pixelProgress;
  };

  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ParallelizeImageRegionHelper(void * arg);

  /** The number of work units to create. */
  ThreadIdType m_NumberOfWorkUnits;

  /** The number of threads to use.
   *  The m_MaximumNumberOfThreads must always be less than or equal to
   *  the m_GlobalMaximumNumberOfThreads before it is used during the execution
   *  of a threaded method. Its value is initialized to
   *  m_GlobalDefaultNumberOfThreads at construction time. Its value is clamped
   *  to the current m_GlobalMaximumNumberOfThreads in the
   *  SingleMethodExecute() method.
   */
  ThreadIdType m_MaximumNumberOfThreads;

  /** Static function used as a "proxy callback" by multi-threaders.  The
   * threading library will call this routine for each thread, which
   * will delegate the control to the prescribed SingleMethod. This
   * routine acts as an intermediary between the multi-threaders and the
   * user supplied callback (SingleMethod) in order to catch any
   * exceptions thrown by the threads. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  SingleMethodProxy(void * arg);

  /** The method to invoke. */
  ThreadFunctionType m_SingleMethod;

  /** The data to be passed as argument. */
  void * m_SingleData;

private:
  /** Only used to synchronize the global variable across static libraries.*/
  itkGetGlobalDeclarationMacro(MultiThreaderBaseGlobals, PimplGlobals);

  static MultiThreaderBaseGlobals * m_PimplGlobals;
  /** Friends of Multithreader.
   * ProcessObject is a friend so that it can call PrintSelf() on its
   * Multithreader. */
  friend class ProcessObject;

  /** Platform specific number of threads */
  static ThreadIdType
  GetGlobalDefaultNumberOfThreadsByPlatform();
};

ITKCommon_EXPORT std::ostream &
                 operator<<(std::ostream & os, const MultiThreaderBase::ThreaderEnum & threader);

} // end namespace itk
#endif

#ifndef ITK_TEMPLATE_EXPLICIT_MultiThreaderBase

namespace itk
{
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegion<1>(const ImageRegion<1> &, TemplatedThreadingFunctorType<1>, ProcessObject *);
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegion<2>(const ImageRegion<2> &, TemplatedThreadingFunctorType<2>, ProcessObject *);
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegion<3>(const ImageRegion<3> &, TemplatedThreadingFunctorType<3>, ProcessObject *);
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegion<4>(const ImageRegion<4> &, TemplatedThreadingFunctorType<4>, ProcessObject *);
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegion<5>(const ImageRegion<5> &,
                                             TemplatedThreadingFunctorType<5> funcP,
                                             ProcessObject *);


extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegionRestrictDirection<1>(unsigned int,
                                                              const ImageRegion<1> &,
                                                              TemplatedThreadingFunctorType<1>,
                                                              ProcessObject *);
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegionRestrictDirection<2>(unsigned int,
                                                              const ImageRegion<2> &,
                                                              TemplatedThreadingFunctorType<2>,
                                                              ProcessObject *);
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegionRestrictDirection<3>(unsigned int,
                                                              const ImageRegion<3> &,
                                                              TemplatedThreadingFunctorType<3>,
                                                              ProcessObject *);
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegionRestrictDirection<4>(unsigned int,
                                                              const ImageRegion<4> &,
                                                              TemplatedThreadingFunctorType<4>,
                                                              ProcessObject *);
extern template void ITKCommon_EXPORT_EXPLICIT
                     MultiThreaderBase::ParallelizeImageRegionRestrictDirection<5>(unsigned int,
                                                              const ImageRegion<5> &,
                                                              TemplatedThreadingFunctorType<5>,
                                                              ProcessObject *);

} // namespace itk
#endif
