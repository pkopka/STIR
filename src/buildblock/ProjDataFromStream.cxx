//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for non-inline functions of class ProjDataFromStream

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataFromStream.h"
#include "stir/Succeeded.h"
#include "stir/Viewgram.h"
#include "stir/Sinogram.h"
#include "stir/SegmentBySinogram.h"
#include "stir/SegmentByView.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "stir/utilities.h"
#include "stir/interfile.h"
#include <numeric>
#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::find;
using std::ios;
using std::iostream;
using std::fstream;
using std::cout;
using std::cerr;
using std::endl;
#endif

#ifdef _MSC_VER
// work-around for compiler bug: VC messes up std namespace
#define FIND std::find
#else
#define FIND find
#endif

START_NAMESPACE_STIR
//---------------------------------------------------------
// constructors
//---------------------------------------------------------

ProjDataFromStream::ProjDataFromStream(shared_ptr<ProjDataInfo> const& proj_data_info_ptr, 
				       shared_ptr<iostream> const& s, const streamoff offs, 
				       const vector<int>& segment_sequence_in_stream_v,
				       StorageOrder o,		      
				       NumericType data_type,
				       ByteOrder byte_order,  
				       float scale_factor)
				       
				       :
                                       ProjData(proj_data_info_ptr),
				       sino_stream(s), offset(offs),
				       segment_sequence(segment_sequence_in_stream_v),
				       storage_order(o),
				       on_disk_data_type(data_type),
				       on_disk_byte_order(byte_order),
				       scale_factor(scale_factor)
{
  assert(storage_order != Unsupported);
  assert(!(data_type == NumericType::UNKNOWN_TYPE));
}

ProjDataFromStream::ProjDataFromStream(shared_ptr<ProjDataInfo> const& proj_data_info_ptr, 
				       shared_ptr<iostream> const& s, const streamoff offs, 
				       StorageOrder o,		      
				       NumericType data_type,
				       ByteOrder byte_order,  
				       float scale_factor)				       
				       :
                                       ProjData(proj_data_info_ptr),
				       sino_stream(s), offset(offs),
				       storage_order(o),
				       on_disk_data_type(data_type),
				       on_disk_byte_order(byte_order),
				       scale_factor(scale_factor)
{
  assert(storage_order != Unsupported);
  assert(!(data_type == NumericType::UNKNOWN_TYPE));

  segment_sequence.resize(proj_data_info_ptr->get_num_segments());

  int segment_num, i;
  for (i= 0, segment_num = proj_data_info_ptr->get_min_segment_num();
       segment_num<=proj_data_info_ptr->get_max_segment_num(); 
       ++i, ++segment_num)
  {
    segment_sequence[i] =segment_num;
  }
}

Viewgram<float> 
ProjDataFromStream::get_viewgram(const int view_num, const int segment_num,
				 const bool make_num_tangential_poss_odd) const
{
  if (sino_stream == 0)
  {
    error("ProjDataFromStream::get_viewgram: stream ptr is 0\n");
  }
  if (! *sino_stream)
  {
    error("ProjDataFromStream::get_viewgram: error in stream state before reading\n");
  }
  
  vector<streamoff> offsets = get_offsets(view_num,segment_num);  
  
  const streamoff segment_offset = offsets[0];
  const streamoff beg_view_offset = offsets[1];
  const streamoff intra_views_offset = offsets[2];
  
  sino_stream->seekg(segment_offset, ios::beg); // start of segment
  sino_stream->seekg(beg_view_offset, ios::cur); // start of view within segment
  
  if (! *sino_stream)
  {
    error("ProjDataFromStream::get_viewgram: error after seekg\n");
  }

  Viewgram<float> viewgram(proj_data_info_ptr, view_num, segment_num);
  float scale = float(1);
  
  if (get_storage_order() == Segment_AxialPos_View_TangPos)
  {    
    for (int ax_pos_num = get_min_axial_pos_num(segment_num); ax_pos_num <= get_max_axial_pos_num(segment_num); ax_pos_num++)
    {
      
      viewgram[ax_pos_num].read_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
      assert(scale == 1);
      sino_stream->seekg(intra_views_offset, ios::cur);
    }
  }
  
  
  else if (get_storage_order() == Segment_View_AxialPos_TangPos)
  {
    viewgram.read_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
    assert(scale == 1);    
  }

  viewgram *= scale_factor;

  if (make_num_tangential_poss_odd &&(get_num_tangential_poss()%2==0))
  {    
    const int new_max_tangential_pos = get_max_tangential_pos_num() + 1;

    viewgram.grow(
     IndexRange2D(get_min_axial_pos_num(segment_num),
     get_max_axial_pos_num(segment_num),
     
     get_min_tangential_pos_num(),
     new_max_tangential_pos));   
  }  

  return viewgram;    
}

vector<streamoff>
ProjDataFromStream::get_offsets(const int view_num, const int segment_num) const

{
  if (!(segment_num >= get_min_segment_num() &&
        segment_num <=  get_max_segment_num()))
    error("ProjDataFromStream::get_offsets: segment_num out of range : %d", segment_num);

  if (!(view_num >= get_min_view_num() &&
        view_num <=  get_max_view_num()))        
    error("ProjDataFromStream::get_offsets: view_num out of range : %d", view_num);


 // cout<<"get_offsets"<<endl;
 // for (int i = 0;i<segment_sequence.size();i++)
 // {
 //   cout<< segment_sequence[i]<<" ";
 // }


 const  int index = 
    FIND(segment_sequence.begin(), segment_sequence.end(), segment_num) - 
    segment_sequence.begin();
   
  streamoff num_axial_pos_offset = 0;
  for (int i=0; i<index; i++)
    num_axial_pos_offset += 
      get_num_axial_poss(segment_sequence[i]);
  
  const streamoff segment_offset = 
    offset + 
    static_cast<streamoff>(num_axial_pos_offset*
                           get_num_tangential_poss() *
			   get_num_views() *
			   on_disk_data_type.size_in_bytes());
  
  if (get_storage_order() == Segment_AxialPos_View_TangPos)
  {
    
    
    const streamoff beg_view_offset =
      (view_num - get_min_view_num()) *get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
    
    const streamoff intra_views_offset = 
      (get_num_views() -1) *get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
    vector<streamoff> temp(3);
    temp[0] = segment_offset;
    temp[1] = beg_view_offset;
    temp[2] = intra_views_offset;
    
    return temp;
  } 
  else //if (get_storage_order() == Segment_View_AxialPos_TangPos)
  {
    const streamoff beg_view_offset =
      (view_num - get_min_view_num()) 
      * get_num_axial_poss(segment_num) 
      * get_num_tangential_poss()
      * on_disk_data_type.size_in_bytes();
    
    
    vector<streamoff> temp(3);
    temp[0] =segment_offset;
    temp[1]= beg_view_offset;
    temp[2] = 0;
    return temp;
    
  }
}

Succeeded
ProjDataFromStream::set_viewgram(const Viewgram<float>& v)
{
  if (sino_stream == 0)
  {
    warning("ProjDataFromStream::set_viewgram: stream ptr is 0\n");
    return Succeeded::no;
  }
  if (! *sino_stream)
  {
    warning("ProjDataFromStream::set_viewgram: error in stream state before writing\n");
    return Succeeded::no;
  }

  // KT 03/07/2001 modified handling of scale_factor etc.
  if (on_disk_data_type.id != NumericType::FLOAT)
  {
    warning("ProjDataFromStream::set_viewgram: non-float output uses original "
	    "scale factor %g which might not be appropriate for the current data\n",
	    scale_factor); 
  }

   if (get_num_tangential_poss() != v.get_proj_data_info_ptr()->get_num_tangential_poss())
  {
    warning("ProjDataFromStream::set_viewgram: num_bins is not correct\n"); 
    return Succeeded::no;
  }

  if (get_num_axial_poss(v.get_segment_num()) != v.get_num_axial_poss())
  {
    warning("ProjDataFromStream::set_viewgram: number of axial positions is not correct\n"); 
    return Succeeded::no;
  }


 
  if (*get_proj_data_info_ptr() != *(v.get_proj_data_info_ptr()))
  {
    warning("ProjDataFromStream::set_viewgram: viewgram has incompatible ProjDataInfo member\n");
   return Succeeded::no;
  }
  int segment_num = v.get_segment_num(); 
  int view_num = v.get_view_num();

  
  vector<streamoff> offsets = get_offsets(view_num,segment_num);  
  const streamoff segment_offset = offsets[0];
  const streamoff beg_view_offset = offsets[1];
  const streamoff intra_views_offset = offsets[2];

  sino_stream->seekp(segment_offset, ios::beg); // start of segment
  sino_stream->seekp(beg_view_offset, ios::cur); // start of view within segment
  
  if (! *sino_stream)
  {
    warning("ProjDataFromStream::set_viewgram: error after seekg\n");
    return Succeeded::no;
  }  
  float scale = scale_factor;
  
  if (get_storage_order() == Segment_AxialPos_View_TangPos)
  {
    for (int ax_pos_num = get_min_axial_pos_num(segment_num); ax_pos_num <= get_max_axial_pos_num(segment_num); ax_pos_num++)
    {
      
      v[ax_pos_num].write_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
      if (scale != scale_factor)
	{
	  warning("ProjDataFromStream::set_viewgram: viewgram (view=%d, segment=%d)"
		  " corrupted due to problems with the scale factor \n",
		  view_num, segment_num);
	  return Succeeded::no;
    }
      
      sino_stream->seekp(intra_views_offset, ios::cur);
    }
    return Succeeded::yes;
  }
  else if (get_storage_order() == Segment_View_AxialPos_TangPos)
  {
    v.write_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
    if (scale != scale_factor)
      {
	warning("ProjDataFromStream::set_viewgram: viewgram (view=%d, segment=%d)"
		" corrupted due to problems with the scale factor \n",
		view_num, segment_num);
	return Succeeded::no;
      }
    return Succeeded::yes;
  }
  else
  {
    warning("ProjDataFromStream::set_viewgram: unsupported storage order\n"); 
    return Succeeded::no;
  }
}




// get offsets for the sino data
vector<streamoff>
ProjDataFromStream::get_offsets_sino(const int ax_pos_num, const int segment_num) const
{
  if (!(segment_num >= get_min_segment_num() &&
        segment_num <=  get_max_segment_num()))
    error("ProjDataFromStream::get_offsets: segment_num out of range : %d", segment_num);

  if (!(ax_pos_num >= get_min_axial_pos_num(segment_num) &&
        ax_pos_num <=  get_max_axial_pos_num(segment_num)))        
    error("ProjDataFromStream::get_offsets: axial_pos_num out of range : %d", ax_pos_num);

  const int index = 
    FIND(segment_sequence.begin(), segment_sequence.end(), segment_num) - 
    segment_sequence.begin();
  
  
  streamoff num_axial_pos_offset = 0;
  for (int i=0; i<index; i++)
    num_axial_pos_offset += 
      get_num_axial_poss(segment_sequence[i]);
  
  
  const streamoff segment_offset = 
    offset + 
    static_cast<streamoff>(num_axial_pos_offset*
                           get_num_tangential_poss() *
			   get_num_views() *
			   on_disk_data_type.size_in_bytes());
  
  if (get_storage_order() == Segment_AxialPos_View_TangPos)
  {
     
    const streamoff beg_ax_pos_offset =
      (ax_pos_num - get_min_axial_pos_num(segment_num))*
           get_num_views() * 
           get_num_tangential_poss()*
           on_disk_data_type.size_in_bytes();

    vector<streamoff> temp(3);
    temp[0] = segment_offset;
    temp[1] = beg_ax_pos_offset;
    temp[2] = 0;
    
    return temp;
  } 
  else //if (get_storage_order() == Segment_View_AxialPos_TangPos)
  {


    const streamoff beg_ax_pos_offset =
      (ax_pos_num - get_min_axial_pos_num(segment_num)) *get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
    
    const streamoff intra_ax_pos_offset = 
      (get_num_axial_poss(segment_num) -1) *get_num_tangential_poss() * on_disk_data_type.size_in_bytes();
   
    
    vector<streamoff> temp(3);
    temp[0] =segment_offset;
    temp[1]= beg_ax_pos_offset;
    temp[2] =intra_ax_pos_offset;
    return temp;
    
  }
}

Sinogram<float>
ProjDataFromStream::get_sinogram(const int ax_pos_num, const int segment_num,
				 const bool make_num_tangential_poss_odd) const
{
  if (sino_stream == 0)
  {
    error("ProjDataFromStream::get_sinogram: stream ptr is 0\n");
  }
  if (! *sino_stream)
  {
    error("ProjDataFromStream::get_sinogram: error in stream state before reading\n");
  }
  
  // Call the get_offset to calculate the offsets, e.g
  // segment offset + view_offset + intra_view_offsets
  vector<streamoff> offsets = get_offsets_sino(ax_pos_num,segment_num);  
  
  const streamoff segment_offset = offsets[0];
  const streamoff beg_ax_pos_offset = offsets[1];
  const streamoff intra_ax_pos_offset = offsets[2];
  
  sino_stream->seekg(segment_offset, ios::beg); // start of segment
  sino_stream->seekg(beg_ax_pos_offset, ios::cur); // start of view within segment
  
  if (! *sino_stream)
  {
    error("ProjDataFromStream::get_sinogram: error after seekg\n");
  }

  Sinogram<float> sinogram(proj_data_info_ptr, ax_pos_num, segment_num);
  float scale = float(1);
  
  if (get_storage_order() == Segment_AxialPos_View_TangPos)
  {    
      sinogram.read_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
      assert(scale == 1);      
  }
  
  
  else if (get_storage_order() == Segment_View_AxialPos_TangPos)
  {
   for (int view = get_min_view_num(); view <= get_max_view_num(); view++)
    {
    sinogram[view].read_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
    assert(scale == 1);
    sino_stream->seekg(intra_ax_pos_offset, ios::cur);
   }    
  }
  sinogram *= scale_factor;

  if (make_num_tangential_poss_odd&&(get_num_tangential_poss()%2==0))
  {
    int new_max_tangential_pos = get_max_tangential_pos_num() + 1;

    sinogram.grow(IndexRange2D(get_min_view_num(),
	get_max_view_num(),
	get_min_tangential_pos_num(),
	new_max_tangential_pos));    
  }
  
  return sinogram;
  
  
}

Succeeded
ProjDataFromStream::set_sinogram(const Sinogram<float>& s)
{
  if (sino_stream == 0)
  {
    warning("ProjDataFromStream::set_sinogram: stream ptr is 0\n");
    return Succeeded::no;
  }
  if (! *sino_stream)
  {
    warning("ProjDataFromStream::set_sinogram: error in stream state before writing\n");
    return Succeeded::no;
  }
  // KT 03/07/2001 modified handling of scale_factor etc.
  if (on_disk_data_type.id != NumericType::FLOAT)
  {
    warning("ProjDataFromStream::set_viewgram: non-float output uses original "
	    "scale factor %g which might not be appropriate for the current data\n",
	    scale_factor); 
  }
  
  if (*get_proj_data_info_ptr() != *(s.get_proj_data_info_ptr()))
  {
    warning("ProjDataFromStream::set_sinogram: Sinogram<float> has incompatible ProjDataInfo member\n");
    return Succeeded::no;
  }
  int segment_num = s.get_segment_num(); 
  int ax_pos_num = s.get_axial_pos_num();
  
  
  vector<streamoff> offsets = get_offsets_sino(ax_pos_num,segment_num);  
  const streamoff segment_offset = offsets[0];
  const streamoff beg_ax_pos_offset = offsets[1];
  const streamoff intra_ax_pos_offset = offsets[2];
  
  sino_stream->seekp(segment_offset, ios::beg); // start of segment
  sino_stream->seekp(beg_ax_pos_offset, ios::cur); // start of view within segment
  
  if (! *sino_stream)
  {
    warning("ProjDataFromStream::set_sinogram: error after seekg\n");
    return Succeeded::no;
  }  
  float scale = scale_factor;
  
  
  if (get_storage_order() == Segment_AxialPos_View_TangPos)
  
    {
      s.write_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
      if (scale != scale_factor)
	{
	  warning("ProjDataFromStream::set_sinogram: sinogram (ax_pos=%d, segment=%d)"
		  " corrupted due to problems with the scale factor \n",
		  ax_pos_num, segment_num);
	  return Succeeded::no;
    }

      return Succeeded::yes;
    }
    
    else if (get_storage_order() == Segment_View_AxialPos_TangPos)
    {
      for (int view = get_min_view_num();view <= get_max_view_num(); view++)
      {
	s[view].write_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
	if (scale != scale_factor)
	  {
	    warning("ProjDataFromStream::set_sinogram: sinogram (ax_pos=%d, segment=%d)"
		    " corrupted due to problems with the scale factor \n",
		    ax_pos_num, segment_num);
	    return Succeeded::no;
	  }
  
	sino_stream->seekp(intra_ax_pos_offset, ios::cur);
      }
      return Succeeded::yes;
    }
    else
    {
      warning("ProjDataFromStream::set_sinogram: unsupported storage order\n"); 
      return Succeeded::no;
    }
  }

streamoff
ProjDataFromStream::get_offset_segment(const int segment_num) const
{
 assert(segment_num >= get_min_segment_num() &&
  segment_num <=  get_max_segment_num());
  { 
      const int index = 
	FIND(segment_sequence.begin(), segment_sequence.end(), segment_num) - 
	segment_sequence.begin();

      streamoff num_axial_pos_offset = 0;
      for (int i=0; i<index; i++)
      num_axial_pos_offset += 
      get_num_axial_poss(segment_sequence[i]);
  
      const streamoff segment_offset = 
	offset + 
	num_axial_pos_offset *
	get_num_tangential_poss() *
	get_num_views()*
	on_disk_data_type.size_in_bytes();
       return segment_offset;
  }
     
}


// TODO the segment version could be written in terms of the above.
// -> No need for get_offset_segment

SegmentBySinogram<float>
ProjDataFromStream::get_segment_by_sinogram(const int segment_num) const
{
  if(sino_stream == 0)
  {
    error("ProjDataFromStream::get_segment_by_sinogram: stream ptr is 0\n");
  }
  if (! *sino_stream)
  {
    error("ProjDataFromStream::get_segment_by_sinogram: error in stream state before reading\n");
  }
    
  streamoff segment_offset = get_offset_segment(segment_num);
  sino_stream->seekg(segment_offset, ios::beg);
  if (! *sino_stream)
  {
    error("ProjDataFromStream::get_segment_by_sinogram: error after seekg\n");
  }
  
  if (get_storage_order() == Segment_AxialPos_View_TangPos)
  {
    SegmentBySinogram<float> segment(proj_data_info_ptr,segment_num);
    {
      float scale = float(1);
      segment.read_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
      assert(scale == 1);
    }
    
    segment *= scale_factor;
    
    return  segment;
    
  }
  else
  {
    // TODO rewrite in terms of get_viewgram
    return SegmentBySinogram<float> (get_segment_by_view(segment_num));
  }
  
  
}

SegmentByView<float>
ProjDataFromStream::get_segment_by_view(const int segment_num) const
{
  
  if(sino_stream == 0)
  {
    error("ProjDataFromStream::get_segment_by_view: stream ptr is 0\n");
  }
  if (! *sino_stream)
  {
    error("ProjDataFromStream::get_segment_by_view: error in stream state before reading\n");
  }
  
  if (get_storage_order() == Segment_View_AxialPos_TangPos)
  {
    
    streamoff segment_offset = get_offset_segment(segment_num);
    sino_stream->seekg(segment_offset, ios::beg);
    
    if (! *sino_stream)
    {
      error("ProjDataFromStream::get_segment_by_sinogram: error after seekg\n");
    }
    
    SegmentByView<float> segment(proj_data_info_ptr,segment_num);
    
    {
      float scale = float(1);
      segment.read_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
      assert(scale == 1);
    }
    
    segment *= scale_factor;
        
    return segment;
  }
  else 
    // TODO rewrite in terms of get_sinogram as this doubles memory temporarily
    return SegmentByView<float> (get_segment_by_sinogram(segment_num));
}

Succeeded
ProjDataFromStream::set_segment(const SegmentBySinogram<float>& segmentbysinogram_v)
{
  if(sino_stream == 0)
  {
    error("ProjDataFromStream::set_segment: stream ptr is 0\n");
  }
  if (! *sino_stream)
  {
    error("ProjDataFromStream::set_segment: error in stream state before writing\n");
  }
  
  if (get_num_tangential_poss() != segmentbysinogram_v.get_num_tangential_poss())
  {
    warning("ProjDataFromStream::set_segmen: num_bins is not correct\n"); 
    return Succeeded::no;
  }
  if (get_num_views() != segmentbysinogram_v.get_num_views())
  {
    warning("ProjDataFromStream::set_segment: num_views is not correct\n"); 
    return Succeeded::no;
  }
  
  int segment_num = segmentbysinogram_v.get_segment_num();
  streamoff segment_offset = get_offset_segment(segment_num);
  
  sino_stream->seekp(segment_offset,ios::beg);
  
  if (! *sino_stream)
  {
    warning("ProjDataFromStream::set_segment: error after seekp\n");
    return Succeeded::no;
  }  
  
  if (get_storage_order() == Segment_AxialPos_View_TangPos)    
  {
    // KT 03/07/2001 handle scale_factor appropriately
    if (on_disk_data_type.id != NumericType::FLOAT)
      {
	warning("ProjDataFromStream::set_segment: non-float output uses original "
		"scale factor %g which might not be appropriate for the current data\n",
		scale_factor); 
      }
    float scale = scale_factor;
    segmentbysinogram_v.write_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
    if (scale != scale_factor)
      {
	warning("ProjDataFromStream::set_segment: segment (%d)"
		" corrupted due to problems with the scale factor \n",
	        segment_num);
	return Succeeded::no;
      }

    return Succeeded::yes;
  }
  else 
  {
    // TODO rewrite in terms of set_viewgram
    const SegmentByView<float> segmentbyview=
      SegmentByView<float>(segmentbysinogram_v);

    set_segment(segmentbyview);   
    return Succeeded::yes;
  }
  
}

Succeeded
ProjDataFromStream::set_segment(const SegmentByView<float>& segmentbyview_v)
{
  if(sino_stream == 0)
  {
    error("ProjDataFromStream::set_segment: stream ptr is 0\n");
  }
  if (! *sino_stream)
  { 
    error("ProjDataFromStream::set_segment: error in stream state before writing\n");
  }
  
  
  if (get_num_tangential_poss() != segmentbyview_v.get_num_tangential_poss())
  {
    warning("ProjDataFromStream::set_segment: num_bins is not correct\n"); 
    return Succeeded::no;
  }
  if (get_num_views() != segmentbyview_v.get_num_views())
  {
    warning("ProjDataFromStream::set_segment: num_views is not correct\n"); 
    return Succeeded::no;
  }
  
  int segment_num = segmentbyview_v.get_segment_num();
  streamoff segment_offset = get_offset_segment(segment_num);
  
  sino_stream->seekp(segment_offset,ios::beg);
  
  if (! *sino_stream)
  {
    warning("ProjDataFromStream::set_segment: error after seekp\n");
    return Succeeded::no;
  }    
  
  if (get_storage_order() == Segment_View_AxialPos_TangPos)    
  {
    // KT 03/07/2001 handle scale_factor appropriately
    if (on_disk_data_type.id != NumericType::FLOAT)
      {
	warning("ProjDataFromStream::set_segment: non-float output uses original "
		"scale factor %g which might not be appropriate for the current data\n",
		scale_factor); 
      }
    float scale = scale_factor;
    segmentbyview_v.write_data(*sino_stream, on_disk_data_type, scale, on_disk_byte_order);
    if (scale != scale_factor)
      {
	warning("ProjDataFromStream::set_segment: segment (%d)"
		" corrupted due to problems with the scale factor \n",
	        segment_num);
	return Succeeded::no;
      }

    return Succeeded::yes;
  }
  else 
  {
    // TODO rewrite in terms of set_sinogram    
    const SegmentBySinogram<float> segmentbysinogram = 
      SegmentBySinogram<float>(segmentbyview_v);
    set_segment(segmentbysinogram);
    return Succeeded::yes;
  }
  
}



ProjDataFromStream* ProjDataFromStream::ask_parameters(const bool on_disk)
{
    
 iostream * p_in_stream;
  
    
    char filename[256];
    cout << endl;
    system("ls *scn *dat *bin");//CL 14/10/98 ADd this printing out of some data files
        
    ask_filename_with_extension(
      filename, 
      "Enter file name of 3D sinogram data : ", ".scn");


    // KT 03/07/2001 initialise to avoid compiler warnings
    ios::openmode  open_mode=ios::in; 
    switch(ask_num("Read (1), Create and write(2), Read/Write (3) : ", 1,3,1))
    {
      case 1: open_mode=ios::in; break;
      case 2: open_mode=ios::out; break;
      case 3: open_mode=ios::in | ios::out; break;
      }
    
    if (on_disk)
    {
      
      //fstream * p_fstream = new fstream;
       p_in_stream = new fstream (filename, open_mode | ios::binary);
       if (!p_in_stream->good())
       {
	 error("ProjDataFromStream::ask_parameters: error opening file %s\n",filename);
       }
      //open_read_binary(*p_fstream, filename);
      //p_in_stream = p_fstream;
    }
    else
    {  
      unsigned long file_size = 0;
      char *memory = 0;
      { 
	fstream input;
	open_read_binary(input, filename);
	memory = (char *)read_stream_in_memory(input, file_size);
      }
      
#ifdef BOOST_NO_STRINGSTREAM
      // This is the old implementation of the strstream class.
      // The next constructor should work according to the doc, but it doesn't in gcc 2.8.1.
      //strstream in_stream(memory, file_size, ios::in | ios::binary);
      // Reason: in_stream contains an internal strstreambuf which is 
      // initialised as buffer(memory, file_size, memory), which prevents
      // reading from it.
      
      strstreambuf * buffer = new strstreambuf(memory, file_size, memory+file_size);
      p_in_stream = new iostream(buffer);
#else
      // TODO this does allocate and copy 2 times

      p_in_stream = new std::stringstream (string(memory, file_size), 
 	                                   open_mode | ios::binary);
	
      delete[] memory;
#endif
      
    } // else 'on_disk' 

   
    // KT 03/07/2001 initialise to avoid compiler warnings    
    ProjDataFromStream::StorageOrder storage_order =
      Segment_AxialPos_View_TangPos;
    {
    int data_org = ask_num("Type of data organisation:\n\
      0: Segment_AxialPos_View_TangPos, 1: Segment_View_AxialPos_TangPos", 
			   0,
                           1,0);
    
    switch (data_org)
    { 
    case 0:
      storage_order = ProjDataFromStream::Segment_AxialPos_View_TangPos;
      break;
    case 1:
      storage_order =ProjDataFromStream::Segment_View_AxialPos_TangPos;
      break;
    }
    }
    
    NumericType data_type;
    {
    int data_type_sel = ask_num("Type of data :\n\
      0: signed 16bit int, 1: unsigned 16bit int, 2: 4bit float ", 0,2,2);
    switch (data_type_sel)
    { 
    case 0:
      data_type = NumericType::SHORT;
      break;
    case 1:
      data_type = NumericType::USHORT;
      break;
    case 2:
      data_type = NumericType::FLOAT;
      break;
    }
    }
    
    
    ByteOrder byte_order;
    { 
      byte_order = 
	ask("Little endian byte order ?",
	ByteOrder::get_native_order() == ByteOrder::little_endian) ?
	ByteOrder::little_endian :
      ByteOrder::big_endian;
    }
    
    long offset_in_file ;
    {
      // find file size
      p_in_stream->seekg(0L, ios::beg);   
      unsigned long file_size = find_remaining_size(*p_in_stream);
      
      offset_in_file = ask_num("Offset in file (in bytes)", 
			     0UL,file_size, 0UL);
    }
    float scale_factor =1;
    
    ProjDataInfo* data_info_ptr =
      ProjDataInfo::ask_parameters();
    
    vector<int> segment_sequence_in_stream; 
    segment_sequence_in_stream = vector<int>(data_info_ptr->get_num_segments());  
    segment_sequence_in_stream[0] =  0; 
    
    for (int i=1; i<= data_info_ptr->get_num_segments()/2; i++)
    { 
      segment_sequence_in_stream[2*i-1] = i;
      segment_sequence_in_stream[2*i] = -i;
    }
    
    
    cerr << "Segment<float> sequence :";
    for (unsigned int i=0; i<segment_sequence_in_stream.size(); i++)
      cerr << segment_sequence_in_stream[i] << "  ";
    cerr << endl;
    
    
    
    ProjDataFromStream* proj_data_ptr =
      new 
      ProjDataFromStream (data_info_ptr,
			  p_in_stream, offset_in_file, 
			  segment_sequence_in_stream,
			  storage_order,data_type,byte_order,  
			  scale_factor);

    cerr << "writing Interfile header for "<< filename << endl;
    write_basic_interfile_PDFS_header(filename, *proj_data_ptr);

    return proj_data_ptr;
    
}

float
ProjDataFromStream::get_scale_factor() const
{ 
  return scale_factor;}



END_NAMESPACE_STIR

  
  
  




