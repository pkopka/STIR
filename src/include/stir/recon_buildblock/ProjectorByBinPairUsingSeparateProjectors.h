//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declares class ProjectorByBinPairUsingSeparateProjectors

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_
#define __stir_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"

START_NAMESPACE_STIR


/*!
  \ingroup recon_buildblock
  \brief A projector pair based on a single matrix
*/
class ProjectorByBinPairUsingSeparateProjectors : 
  public RegisteredParsingObject<ProjectorByBinPairUsingSeparateProjectors,
                                 ProjectorByBinPair> 
{ 
public:
  //! Name which will be used when parsing a ProjectorByBinPair object
  static const char * const registered_name; 

  //! Default constructor 
  ProjectorByBinPairUsingSeparateProjectors();

   //! Constructor that sets the pair
  ProjectorByBinPairUsingSeparateProjectors(const shared_ptr<ForwardProjectorByBin>& forward_projector_ptr,
                                            const shared_ptr<BackProjectorByBin>& back_projector_ptr);


private:

  void set_defaults();
  void initialise_keymap();
};

END_NAMESPACE_STIR


#endif // __stir_recon_buildblock_ProjectorByBinPairUsingSeparateProjectors_h_
