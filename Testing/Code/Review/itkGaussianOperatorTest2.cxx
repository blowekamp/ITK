#include "itkGaussianOperator.h"
#include "itkGaussianDerivativeOperator.h"

#include <iterator>

int itkGaussianOperatorTest2( int argc, char *argv[] )
{
  //typedef itk::GaussianOperator< double, 1 > GaussianOp;
  typedef itk::GaussianDerivativeOperator< double, 1 > GaussianOp;

  if (argc != 5 )
    {
    std::cerr << "Usage: " << argv[0] << " variance error width order" << std::endl;
    return EXIT_FAILURE;
    }

  double       variance = atof(argv[1]);
  double       error = atof(argv[2]);
  unsigned int width = atoi(argv[3]);
  unsigned int order = atoi(argv[4]);

    {
    GaussianOp op;

    op.SetVariance( variance );
    op.SetMaximumError( error );
    op.SetMaximumKernelWidth( width );

    op.SetOrder( order );
    op.SetNormalizeAcrossScale( false );
    op.SetUseDerivativeOperator( false );

    op.CreateDirectional();

    std::cout.precision(16);
    std::cout << "---operator---" << std::endl;
    GaussianOp::Iterator i = op.Begin();
    i += op.Size()/2;
    for( ; i != op.End(); ++i )
      {
      std::cout << *i << std::endl;
      }
    std::cout << "---end--" << std::endl;

    double total = std::accumulate( op.Begin(), op.End(), 0.0 );

    std::cout << "total: " << total << std::endl;

    }

    {
    GaussianOp op;

    op.SetVariance( variance );
    op.SetMaximumError( error );
    op.SetMaximumKernelWidth( width );

    op.SetOrder( order );
    op.SetNormalizeAcrossScale( false );
    op.SetUseDerivativeOperator( true );

    op.CreateDirectional();

    std::cout.precision(16);
    std::cout << "---operator---" << std::endl;
    GaussianOp::Iterator i = op.Begin();
    i += op.Size()/2;
    for( ; i != op.End(); ++i )
      {
      std::cout << *i << std::endl;
      }
    std::cout << "---end--" << std::endl;

    double total = std::accumulate( op.Begin(), op.End(), 0.0 );

    std::cout << "total: " << total << std::endl;

    }

  return EXIT_SUCCESS;

}