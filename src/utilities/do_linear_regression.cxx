//
// $Id$
//
/*!
  \file 
  \ingroup utilities
 
  \brief 
  A simple programme to perform weighted least squares.

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$

  This performs a weighted least squares fit.<br>
  stdev and covariance are computed using the estimated 
  variance chi_square/(n-2).

  The file should contain data in the following format:<br>
  number_of_points<br>
  coordinates<br>
  data<br>
  weights
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/linear_regression.h"
#include "stir/VectorWithOffset.h"

#include <fstream>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{

  if (argc != 2)
  {
    cerr << "Usage : " << argv[0] << " filename\n"
         << "This performs a weighted least squares fit.\n"
	 << "stdev and covariance are computed using the estimated "
	 << "variance chi_square/(n-2).\n\n"
	 << "The file should contain data in the following format:\n\n"
	 << "number_of_points\n"
	 << "coordinates\n"
	 << "data\n"	 
	 << "weights\n" << endl;
    return EXIT_FAILURE;
  }



  ifstream in(argv[1]);
  if (!in)
  {
    cerr << argv[0] 
         << ": Error opening input file " << argv[1] << "\nExiting."
	 << endl;
    return 1;
  }

  int size;
  in >> size;

  VectorWithOffset<float> coordinates(size);
  VectorWithOffset<float> measured_data(size);
  VectorWithOffset<float> weights(size);
  for (int i=0; i<size; i++)
    in >> coordinates[i];
  for (int i=0; i<size; i++)
    in >> measured_data[i];
  for (int i=0; i<size; i++)
    in >> weights[i];

  if (!in)
  {
    cerr << argv[0] 
         << ": Error while reading input file " << argv[1] 
	 << "\nTrying to continue."
	 << endl;    
  }

  double scale=0;
  double constant=0;
  double variance_of_scale=0;
  double variance_of_constant=0;
  double covariance_of_constant_with_scale=0;
  double chi_square = 0;

  linear_regression(
    constant, scale,
    chi_square,
    variance_of_constant,
    variance_of_scale,
    covariance_of_constant_with_scale,
    measured_data,
    coordinates,
    weights);

  cout << "scale = " << scale << " +- " << sqrt(variance_of_scale)
       << ", cst = " << constant << " +- " << sqrt(variance_of_constant)
       << "\nchi_square = " << chi_square
       << "\ncovariance = " << covariance_of_constant_with_scale
       << endl;
  return EXIT_SUCCESS;
}

