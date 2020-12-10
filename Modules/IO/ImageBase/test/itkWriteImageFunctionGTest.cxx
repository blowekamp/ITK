/*=========================================================================
 *
 *  Copyright NumFOCUS
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
#include "itkImageFileWriter.h"
#include "itkImage.h"

#include "itkGTest.h"
#include "itksys/SystemTools.hxx"
#include "itkTestDriverIncludeRequiredIOFactories.h"

#define STRING(s) #s
namespace
{

class ITKWriteImageFunctionTest : public ::testing::Test
{
  void
  SetUp() override
  {
    RegisterRequiredFactories();
    itksys::SystemTools::ChangeDirectory(STRING(ITK_TEST_OUTPUT_DIR_STR));
  }
};

using ImageType = itk::Image<float, 2>;

ImageType::Pointer
MakeImage()
{
  auto image = ImageType::New();

  ImageType::RegionType region({ 3, 2 });

  image->SetRegions(region);

  image->Allocate(true);
  return image;
}

} // namespace

TEST_F(ITKWriteImageFunctionTest, ImageTypes)
{
  ImageType::Pointer image_ptr = MakeImage();
  itk::WriteImage(image_ptr, "test1.mha");
  itk::WriteImage(std::move(image_ptr), "test1.mha");

  ImageType::ConstPointer image_cptr = MakeImage();
  itk::WriteImage(image_cptr, "test1.mha");
  itk::WriteImage(std::move(image_cptr), "test1.mha");

  const ImageType::ConstPointer image_ccptr = MakeImage();
  itk::WriteImage(image_ccptr, "test1.mha");
  itk::WriteImage(std::move(image_ccptr), "test1.mha");

  image_ptr = MakeImage();
  ImageType * image_rptr = image_ptr.GetPointer();
  itk::WriteImage(image_rptr, "test1.mha");

  const ImageType * image_crptr = image_cptr.GetPointer();
  itk::WriteImage(image_crptr, "test1.mha");
}
