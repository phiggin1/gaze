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
            echo "$filename"
            python2 extract_audio.py --bagfile $entry #--distancetype $distance_type
            python2 gaze.py --bagfile $entry #--distancetype $distance_type
            python3 gantt_chart.py --basefilename $filename #--distancetype $distance_type
            python accuracy.py --basefilename $filename #--distancetype $distance_type
        done
  else
    echo "Requires valid directory"
  fi
fi
