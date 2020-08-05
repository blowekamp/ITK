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

#include "itkGTest.h"

#include "itkTileImageFilter.h"
#include "itkVectorImage.h"

#include "itkTestDriverIncludeRequiredIOFactories.h"
#include "itkTestingHashImageFilter.h"

namespace
{
class PasteFixture : public ::testing::Test
{
public:
  PasteFixture() = default;
  ~PasteFixture() override = default;

protected:
  void
  SetUp() override
  {
    RegisterRequiredFactories();
  }

  template <typename TImageType>
  static std::string
  MD5Hash(const TImageType * image)
  {

    using HashFilter = itk::Testing::HashImageFilter<TImageType>;
    typename HashFilter::Pointer hasher = HashFilter::New();
    hasher->SetInput(image);
    hasher->Update();
    return hasher->GetHash();
  }

  template <typename TImage>
  struct FixtureUtilities
  {
    static const unsigned int Dimension = TImage::ImageDimension;

    using PixelType = typename TImage::PixelType;
    using OutputPixelType = PixelType;
    using InputImageType = TImage;
    using OutputImageType = TImage;
    using RegionType = typename InputImageType::RegionType;
    using SizeType = typename TImage::SizeType;
    using IndexType = typename TImage::IndexType;

    using FilterType = itk::PasteImageFilter<InputImageType, OutputImageType>;

    // Create a black image or empty
    static typename InputImageType::Pointer
    CreateImage(unsigned int size = 100)
    {
      typename InputImageType::Pointer image = InputImageType::New();

      typename InputImageType::SizeType imageSize;
      imageSize.Fill(size);
      image->SetRegions(RegionType(imageSize));
      image->Allocate();
      image->FillBuffer(0);

      return image;
    }
  };
};
} // namespace


TEST_F(PasteFixture, SetGetPrint)
{
  using Utils = FixtureUtilities<itk::Image<unsigned char, 3>>;

  auto filter = Utils::FilterType::New();
  filter->Print(std::cout);

  EXPECT_STREQ("PasteImageFilter", filter->GetNameOfClass());

  EXPECT_NO_THROW(filter->SetConstant(5));
  EXPECT_EQ(5, filter->GetConstant());
}


TEST_F(PasteFixture, ConstantPaste)
{
  using Utils = FixtureUtilities<itk::Image<int, 2>>;

  auto filter = Utils::FilterType::New();
  auto inputImage = Utils::CreateImage(100);

  filter->SetDestinationImage(inputImage);
  filter->SetDestinationIndex({ 11, 13 });

  constexpr int constantValue = -97;
  filter->SetConstant(constantValue);
  filter->SetSourceRegion(Utils::SizeType{ 3, 3 });

  filter->SetNumberOfWorkUnits(1);
  filter->UpdateLargestPossibleRegion();

  auto outputImage = filter->GetOutput();
  EXPECT_EQ(0, outputImage->GetPixel({ 10, 13 }));
  EXPECT_EQ(0, outputImage->GetPixel({ 11, 12 }));

  EXPECT_EQ(constantValue, outputImage->GetPixel({ 11, 13 }));
  EXPECT_EQ(constantValue, outputImage->GetPixel({ 13, 15 }));

  EXPECT_EQ("e73092e3e58f66f13d32a5aca97ca1d9", MD5Hash(filter->GetOutput()));

  filter->SetNumberOfWorkUnits(100);
  filter->UpdateLargestPossibleRegion();
  EXPECT_EQ("e73092e3e58f66f13d32a5aca97ca1d9", MD5Hash(filter->GetOutput()));
}
