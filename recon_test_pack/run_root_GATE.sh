#! /bin/bash
# Shell script for testing lm_to_projdata with ROOT support
# see README.txt
#
#  Copyright (C) 2016, 2017 UCL
#  This file is part of STIR.
#
#  This file is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 2.1 of the License, or
#  (at your option) any later version.

#  This file is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  See STIR/LICENSE.txt for details
#      
# Author Nikos Efthimiou, Kris Thielemans

echo This script should work with STIR version ">"3.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

if test "$1" = "--help"
  then
    echo "Usage: run_root_GATE.sh [install_dir]"
    echo "See README.txt for more info."
    exit 1
  else
    echo Warning: Unknown option "$1"
    echo rerun with --help for more info.
    exit 1
  fi

  shift 1

done 

if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi

command -v lm_to_projdata >/dev/null 2>&1 || { echo "lm_to_projdata not found or not executable. Aborting." >&2; exit 1; }
command -v root >/dev/null 2>&1 || { echo "root not found or not executable. Aborting." >&2; exit 1; }
echo "Testing the following executable:"
command -v lm_to_projdata

log=lm_to_projdata_input_formats.log
lm_to_projdata --input-formats  > ${log} 2>&1
if ! grep -q ROOT lm_to_projdata_input_formats.log; then
echo GATE support has not been installed in this system. Aborting.
echo '"lm_to_proj_projdata --input_formats" reported'
cat ${log}
exit 1;
else
echo "GATE support detected!"
fi


echo Executing tests on ROOT files generated by GATE simulations, with cylindrical PET scanners





# first delete any files remaining from a previous run
rm -f my_proj*s lm_to_projdata*.log root*.log

ThereWereErrors=0
export INPUT_ROOT_FILE=test_PET_GATE.root
export INPUT=root_header.hroot
export TEMPLATE=template_for_ROOT_scanner.hs

#
# Get the number of events unlisted. 
#
echo
echo ------------- Converting ROOT files to ProjData file -------------
echo
echo Making ProjData for all events
echo Running lm_to_projdata for all events

export OUT_PROJDATA_FILE=my_proj_from_lm_all_events
export EXCLUDE_SCATTERED=0
export EXCLUDE_RANDOM=0
${INSTALL_DIR}lm_to_projdata ./lm_to_projdata.par 2>"./my_root_log_all.log"
all_events=`awk -F ":" '/Number of prompts/ {print $2}' my_root_log_all.log`
echo "Number of prompts stored in this time period:" ${all_events}

log=lm_to_projdata_from_ROOT_all_events.log
lm_to_projdata ./lm_to_projdata.par > ${log} 2>&1 
if [ $? -ne 0 ]; then
  ThereWereErrors=1
  echo "Error running lm_to_projdata."
  echo "CHECK LOG $log"
  error_log_files="${error_log_files} ${log}"
else
  all_events=$(cat ${log}|grep "Number of prompts stored in this time period" | grep -o -E '[0-9]+')

  echo Number of prompts stored in this time period: ${all_events}
  echo
fi
echo Reading all values from ROOT file

#
# Get the number of events directly from the ROOT file
#
echo
echo ------------- Reading values directly from ROOT file -------------
echo
cat << EOF > my_root.input
Coincidences->Draw(">>eventlist","","goff");
Int_t N = eventlist->GetN();
cout<<endl<<"Number of prompts stored in this time period:"<< N<<endl;
EOF
if [ $? -ne 0 ]; then
  ThereWereErrors=1
  echo "Error running root."
  echo "CHECK LOG $log"
  error_log_files="${error_log_files} ${log}"
else
  all_root_num=$(cat ${log}| grep "Number of prompts stored in this time period" | grep -o -E '[0-9]+')
  echo All events in ROOT file : ${all_root_num}
fi
if [ "$all_events" != "$all_root_num" ];then
  ThereWereErrors=1
fi
echo
echo Making ProjData for true events only
echo Running lm_to_projdata *ONLY* for true events

export OUT_PROJDATA_FILE=my_proj_from_lm_true_events
export EXCLUDE_SCATTERED=1
export EXCLUDE_RANDOM=1

log=lm_to_projdata_from_ROOT_true_events.log
lm_to_projdata ./lm_to_projdata.par > ${log} 2>&1 
if [ $? -ne 0 ]; then
  ThereWereErrors=1
  echo "Error running lm_to_projdata."
  echo "CHECK LOG $log"
  error_log_files="${error_log_files} ${log}"
else
  true_events=$(cat ${log}| grep "Number of prompts stored in this time period" | grep -o -E '[0-9]+')
  echo Number of prompts stored in this time period: ${true_events}
fi
echo
echo Reading true values from ROOT file ...

log=root_output_true_events.log
root -b -l ${INPUT_ROOT_FILE} << EOF >& ${log} 
Coincidences->Draw(">>eventlist","eventID1 == eventID2 && comptonPhantom1 == 0 && comptonPhantom2 == 0","goff");
Int_t N = eventlist->GetN();
cout<<endl<<"Number of trues stored in this time period:"<< N<<endl;
EOF
if [ $? -ne 0 ]; then
  ThereWereErrors=1
  echo "Error running root."
  echo "CHECK LOG $log"
  error_log_files="${error_log_files} ${log}"
else
  true_root_num=$(cat ${log}| grep "Number of trues stored in this time period" | grep -o -E '[0-9]+')
  echo True events in ROOT file : ${true_root_num}
fi
if [ "$true_events" != "$true_root_num" ]; then
  ThereWereErrors=1
fi

echo
echo '--------------- End of tests -------------'
echo
if test ${ThereWereErrors} = 1  ; 
then
  echo "Check what went wrong. The following log files might help you:"
  echo "${error_log_files}"
  if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    for log in ${error_log_files}; do
      echo "=========== ${log} =========="
      cat ${log}
    done
  fi
  exit 1
else
  echo "Everything seems to be fine !"
  echo 'You could remove all generated files using "rm -f my_* *.log"'
fi
