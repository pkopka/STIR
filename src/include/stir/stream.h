//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Input/output of basic vector-like types to/from streams

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stream_H__
#define __stream_H__

#include "stir/VectorWithOffset.h"
#include "stir/BasicCoordinate.h"
#include <iostream>
#include <vector>
#ifndef STIR_NO_NAMESPACE
using std::istream;
using std::ostream;
using std::vector;
#endif

START_NAMESPACE_STIR

/*!
  \brief Outputs a VectorWithOffset to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with an endl at the end. 
  
  This can be used for higher dimensional arrays as well, where each 1D subobject 
  will be on its own line.
*/


template <typename elemT>
inline 
ostream& 
operator<<(ostream& str, const VectorWithOffset<elemT>& v);

/*!
  \brief Outputs a BasicCoordinate to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with no endl at the end. 
  */
template <int num_dimensions, typename coordT>
inline 
ostream& 
operator<<(ostream& str, const BasicCoordinate<num_dimensions, coordT>& v);


/*!
  \brief Outputs a vector to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with an endl at the end. 
  
  For each element of the vector ostream::operator<<() will be called.
*/
template <typename elemT>
inline 
ostream& 
operator<<(ostream& str, const vector<elemT>& v);

/*!
  \brief Inputs a vector from a stream.

  Input is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  
  Input is stopped when either the beginning '{', an intermediate ',' or the 
  trailing '}' is not found. The size of the vector will be the number of 
  correctly read elemT elements.
  
  For each element of the vector istream::operator>>() will be called.

  elemT needs to have a default constructor.
*/
template <typename elemT>
inline 
istream& 
operator>>(istream& str, vector<elemT>& v);

/*!
  \brief Inputs a VectorWithOffset from a stream.

  Input is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  
  Input is stopped when either the beginning '{', an intermediate ',' or the 
  trailing '}' is not found.  The size of the vector will be the number of 
  correctly read elemT elements.
  
  v.get_min_index() will be 0 at the end of the call.

  For each element of the vector istream::operator>>() will be called.

  elemT needs to have a default constructor.
*/
template <typename elemT>
inline 
istream& 
operator>>(istream& str, VectorWithOffset<elemT>& v);

/*!
  \brief Inputs a coordinate from a stream.

  Input is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  
  Input is stopped when either the beginning '{', an intermediate ',' or the 
  trailing '}' is not found. The size of the vector will be the number of 
  correctly read elemT elements.
  
  For each element of the vector istream::operator>>() will be called.

  elemT needs to have a default constructor.
*/
template <int num_dimensions, typename coordT>
inline 
istream& 
operator<<(istream& str, BasicCoordinate<num_dimensions, coordT>& v);

END_NAMESPACE_STIR

#include "stir/stream.inl"

#endif
