//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Declaration of class SeparableArrayFunctionObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_SeparableArrayFunctionObject_H__
#define __stir_SeparableArrayFunctionObject_H__

#include "stir/ArrayFunctionObject_1ArgumentImplementation.h"
#include "stir/shared_ptr.h"
#include <vector>
#include "stir/VectorWithOffset.h"

#ifndef STIR_NO_NAMESPACES
using std::vector;
#endif


START_NAMESPACE_STIR



/*!
  \ingroup buildblock
  \brief This class implements an \c n -dimensional ArrayFunctionObject whose operation
  is separable.

  'Separable' means that its operation consists of \c n 1D operations, one on each
  index of the \c n -dimensional array. 
  \see in_place_apply_array_functions_on_each_index()
  
 */
template <int num_dimensions, typename elemT>
class SeparableArrayFunctionObject : 
   public ArrayFunctionObject_1ArgumentImplementation<num_dimensions,elemT>
{
public:
  SeparableArrayFunctionObject ();
  SeparableArrayFunctionObject (const VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >&); 
  bool is_trivial() const;

protected:
 
  VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > > all_1d_array_filters;
  virtual void do_it(Array<num_dimensions,elemT>& array) const;

};


END_NAMESPACE_STIR


#endif //SeparableArrayFunctionObject

