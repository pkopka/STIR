/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013-2014 University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Classes for listmode events for the ECAT 8 format
    
  \author Kris Thielemans
*/

#ifndef __stir_listmode_CListRecordECAT8_32bit_H__
#define __stir_listmode_CListRecordECAT8_32bit_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventECAT8_32bit.h"
#include "stir/listmode/CListDataAnyECAT8_32bit.h"
#include "stir/listmode/CListTimeECAT8_32bit.h"

#include "stir/ByteOrder.h"

START_NAMESPACE_STIR
namespace ecat {


//! A class for a general element of a listmode file for a Siemens scanner using the ECAT8 32bit format.
/*! \ingroup listmode
   We currently only support coincidence events and  a timing flag.
   Here we only support the 32bit version specified by the PETLINK protocol.

   This class is based on Siemens information on the PETLINK protocol, available at
   http://usa.healthcare.siemens.com/siemens_hwem-hwem_ssxa_websites-context-root/wcm/idc/groups/public/@us/@imaging/@molecular/documents/download/mdax/mjky/~edisp/petlink_guideline_j1-00672485.pdf

*/
 class CListRecordECAT8_32bit : public CListRecord // currently no gating yet
{

  //public:

  bool is_time() const
  { return this->any_data.is_time(); }
  /*
  bool is_gating_input() const
  { return this->is_time(); }
  */
  bool is_event() const
  { return this->any_data.is_event(); }
  virtual CListEventECAT8_32bit&  event() 
    { return this->event_data; }
  virtual const CListEventECAT8_32bit&  event() const
    { return this->event_data; }
  virtual CListTimeECAT8_32bit&   time()
    { return this->time_data; }
  virtual const CListTimeECAT8_32bit&   time() const
    { return this->time_data; }

  bool operator==(const CListRecord& e2) const
  {
    return dynamic_cast<CListRecordECAT8_32bit const *>(&e2) != 0 &&
      raw == dynamic_cast<CListRecordECAT8_32bit const &>(e2).raw;
  }	 

 public:     
 CListRecordECAT8_32bit(const shared_ptr<ProjDataInfo>& proj_data_info_sptr) :
  event_data(proj_data_info_sptr)
    {}

  virtual Succeeded init_from_data_ptr(const char * const data_ptr, 
                                       const std::size_t
#ifndef NDEBUG
                                       size // only used within assert, so commented-out otherwise to avoid compiler warnings
#endif
                                       , const bool do_byte_swap)
  {
    assert(size >= 4);
    std::copy(data_ptr, data_ptr+4, reinterpret_cast<char *>(&raw));
    if (do_byte_swap)
      ByteOrder::swap_order(raw);
    this->any_data.init_from_data_ptr(&raw);
    // should in principle check return value, but it's always Succeeded::yes anyway
    if (this->any_data.is_time())
      return this->time_data.init_from_data_ptr(&raw);
     else if (this->any_data.is_event())
      return this->event_data.init_from_data_ptr(&raw);
    else
      return Succeeded::yes;
  }

  virtual std::size_t size_of_record_at_ptr(const char * const /*data_ptr*/, const std::size_t /*size*/, 
                                            const bool /*do_byte_swap*/) const
  { return 4; }

 private:
  CListEventECAT8_32bit  event_data;
  CListTimeECAT8_32bit   time_data; 
  CListDataAnyECAT8_32bit   any_data; 
  boost::int32_t         raw; // this raw field isn't strictly necessary, get rid of it?

};

} // namespace ecat
END_NAMESPACE_STIR

#endif

