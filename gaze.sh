#!/bin/bash
	
if [ -z "$1" ]
then
  echo "Requires valid diretory"
else
  search_dir=$1
  #distance_type=$2
  if test -d "$search_dir"; then
    for entry in "$search_dir"/*
        do
            #echo $distance_type
            filename=$(basename -- "$entry")
            extension="${filename##*.}"
            filename="${filename%.*}"
            #echo $entry
            #rosbag info $entry
            #python2 extract_audio.py --bagfile "$entry"
            #python2 gaze.py --bagfile $entry 
            python3 gantt_chart.py --basefilename $filename 
            #python accuracy.py --basefilename $filename
            
        done
  else
    echo "Requires valid directory"
  fi
fi
