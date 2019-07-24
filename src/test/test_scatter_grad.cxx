//
//
/*!

  \file
  \ingroup test

  \author Ludovica Brusaferri

*/
/*
    Copyright (C) 2015, University College London
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

#include "stir/ProjDataInMemory.h"
#include "stir/inverse_SSRB.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/scatter/ScatterEstimation.h"
#include "stir/scatter/SingleScatterLikelihoodAndGradient.h"
#include "stir/Sinogram.h"
#include "stir/Array.h"
#include "stir/Shape/EllipsoidalCylinder.h"

#include "stir/IO/write_to_file.h"

#include <random>



START_NAMESPACE_STIR


class ScatterGradientTests: public RunTests
{   
public:
  void run_tests();
  void fill_projdata_with_random(ProjData & projdata);
  void fill_image_with_random(VoxelsOnCartesianGrid<float> & image);
};

void
ScatterGradientTests::
fill_projdata_with_random(ProjData & projdata)
{

    std::random_device random_device;
    std::mt19937 random_number_generator(random_device());
    std::uniform_real_distribution<float> number_distribution(0,10);
    Bin bin;
    {
        for (bin.segment_num()=projdata.get_min_segment_num();
             bin.segment_num()<=projdata.get_max_segment_num();
             ++bin.segment_num())
            for (bin.axial_pos_num()=
                 projdata.get_min_axial_pos_num(bin.segment_num());
                 bin.axial_pos_num()<=projdata.get_max_axial_pos_num(bin.segment_num());
                 ++bin.axial_pos_num())
            {

                Sinogram<float> sino = projdata.get_empty_sinogram(bin.axial_pos_num(),bin.segment_num());

                for (bin.view_num()=sino.get_min_view_num();
                     bin.view_num()<=sino.get_max_view_num();
                     ++bin.view_num())
                {
                    for (bin.tangential_pos_num()=
                         sino.get_min_tangential_pos_num();
                         bin.tangential_pos_num()<=
                         sino.get_max_tangential_pos_num();
                         ++bin.tangential_pos_num())
                         sino[bin.view_num()][bin.tangential_pos_num()]= number_distribution(random_number_generator);

                    projdata.set_sinogram(sino);

                 }

              }

        }
}

void
ScatterGradientTests::fill_image_with_random(VoxelsOnCartesianGrid<float> & image)
{
    std::random_device random_device;
    std::mt19937 random_number_generator(random_device());
    std::uniform_real_distribution<float> number_distribution(0,10);

    for(int i=0 ; i<image.get_max_z() ; i++){
               for(int j=0 ; j<image.get_max_y(); j++){
                   for(int k=0 ; k<image.get_max_z() ; k++){

                       image[i][j][k] = number_distribution(random_number_generator);

                   }
                 }
             }

}

void
ScatterGradientTests::
run_tests()
{

    unique_ptr<SingleScatterLikelihoodAndGradient> sss(new SingleScatterLikelihoodAndGradient());

    Scanner::Type type= Scanner::E931;
    shared_ptr<Scanner> test_scanner(new Scanner(type));

    if(!test_scanner->has_energy_information())
    {
        test_scanner->set_reference_energy(511);
        test_scanner->set_energy_resolution(0.34f);
    }

    check(test_scanner->has_energy_information() == true, "Check the scanner has energy information.");

    shared_ptr<ExamInfo> exam(new ExamInfo);
    exam->set_low_energy_thres(450);
    exam->set_high_energy_thres(650);

    check(exam->has_energy_information() == true, "Check the ExamInfo has energy information.");

    sss->set_exam_info_sptr(exam);

    // Create the original projdata
    shared_ptr<ProjDataInfoCylindricalNoArcCorr> original_projdata_info( dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(
                                                                             ProjDataInfo::ProjDataInfoCTI(test_scanner,
                                                                                                           1, 0,
                                                                                                           test_scanner->get_num_detectors_per_ring()/2,
                                                                                                           test_scanner->get_max_num_non_arccorrected_bins(),
                                                                                                           false)));

    check(original_projdata_info->has_energy_information() == true, "Check the ProjDataInfo has energy information.");

    shared_ptr<VoxelsOnCartesianGrid<float> > tmpl_density( new VoxelsOnCartesianGrid<float>(*original_projdata_info));

    sss->set_template_proj_data_info_sptr(original_projdata_info);


    shared_ptr<VoxelsOnCartesianGrid<float> > water_density(tmpl_density->clone());
    {
        EllipsoidalCylinder phantom(tmpl_density->get_z_size()*tmpl_density->get_voxel_size().z()*0.2,
                                    tmpl_density->get_y_size()*tmpl_density->get_voxel_size().y()*0.15,
                                    tmpl_density->get_x_size()*tmpl_density->get_voxel_size().x()*0.15,
                                    tmpl_density->get_origin());

        CartesianCoordinate3D<int> num_samples(3,3,3);
        phantom.construct_volume(*water_density, num_samples);
        // Water attenuation coefficient.
        *water_density *= 9.687E-02;

    }

    sss->set_density_image_sptr(water_density);
    sss->set_density_image_for_scatter_points_sptr(water_density);
    sss->set_random_point(false);

    shared_ptr<VoxelsOnCartesianGrid<float> > act_density(tmpl_density->clone());
    {
        CartesianCoordinate3D<float> centre(tmpl_density->get_origin());
        centre[3] += 80.f;
        EllipsoidalCylinder phantom(tmpl_density->get_z_size()*tmpl_density->get_voxel_size().z()*2,
                                    tmpl_density->get_y_size()*tmpl_density->get_voxel_size().y()*0.0625,
                                    tmpl_density->get_x_size()*tmpl_density->get_voxel_size().x()*0.0625,
                                    centre);

        CartesianCoordinate3D<int> num_samples(3,3,3);
        phantom.construct_volume(*act_density, num_samples);
    }

    sss->set_activity_image_sptr(act_density);
    sss->default_downsampling();

    shared_ptr<ProjDataInfoCylindricalNoArcCorr> output_projdata_info(sss->get_template_proj_data_info_sptr());
    shared_ptr<ProjDataInMemory> sss_output(new ProjDataInMemory(exam, output_projdata_info));
    sss->set_output_proj_data_sptr(sss_output);

    //check(sss->process_data() == Succeeded::yes ? true : false, "Check Scatter Simulation process");
    int lenght = sss_output->get_num_views()*sss_output->get_num_axial_poss(0)*sss_output->get_num_tangential_poss(); //TODO: the code is for segment zero only
    std::vector<VoxelsOnCartesianGrid<float> > jacobian_array;
    for (int i = 0 ; i <= lenght ; ++i)
    {
       jacobian_array.push_back(*(act_density));
       jacobian_array[i].fill(0);
    }

    sss->get_jacobian(jacobian_array,true,true);

    ProjDataInMemory x(sss_output->get_exam_info_sptr(),sss_output->get_proj_data_info_sptr());
    fill_projdata_with_random(x);

    ProjDataInMemory out = *(sss->get_output_proj_data_sptr());

    float cdot1 = 0;
    float cdot2 = 0;
    for ( int segment_num = x.get_min_segment_num(); segment_num <= x.get_max_segment_num(); ++segment_num)
      {
        for (int axial_pos = x.get_min_axial_pos_num(segment_num); axial_pos <= x.get_max_axial_pos_num(segment_num); ++axial_pos)
          {

            const Sinogram<float> x_sinogram = x.get_sinogram(axial_pos, segment_num);
            const Sinogram<float> Aty_sinogram = out.get_sinogram(axial_pos, segment_num);

            for ( int view_num = x.get_min_view_num(); view_num <= x.get_max_view_num();view_num++)
              {
                for ( int tang_pos = x.get_min_tangential_pos_num(); tang_pos <= x.get_max_tangential_pos_num(); tang_pos++)
                 {
                     cdot2 += x_sinogram[view_num][tang_pos]*Aty_sinogram[view_num][tang_pos];
                  }
               }
             }
          }

    std::cout << cdot1 << "=" << cdot2 << '\n';

//  std::cout << "-------- Testing Upsampling and Downsampling ---------\n";


}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  ScatterGradientTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
