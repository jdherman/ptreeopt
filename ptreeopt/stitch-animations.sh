#!/bin/bash
# Make animation from a directory of pngs
# requires imagemagick

prefix=$1 # filename
L=$2 # image optimization level

# get largest dimensions
w=`identify -format "%w %f\n" temp/${prefix}*.png | sort -n -r -k 2 | head -n 1 | cut -d ' ' -f -1`
h=`identify -format "%h %f\n" temp/${prefix}*.png | sort -n -r -k 2 | head -n 1 | cut -d ' ' -f -1`

# old way
# for f in `ls temp/${prefix}*.png`
# do
#   convert $f -background white -extent ${w}x${h} $f
# done

# create a blank canvas of that size
convert -dispose none -delay 0 \
          -size ${w}x${h} xc:white +antialias \
          -fill white \
        -dispose previous -delay 20 \
          temp/${prefix}*.png  \
        -loop 0 -coalesce $L ${prefix}.gif

