#!/bin/bash

# get largest dimensions
# prefix="tree"
prefix="folsom"
# prefix="convergence"

# L=""
L="-layers optimize"

w=`identify -format "%w %f\n" anim/${prefix}*.png | sort -n -r -k 2 | head -n 1 | cut -d ' ' -f -1`
h=`identify -format "%h %f\n" anim/${prefix}*.png | sort -n -r -k 2 | head -n 1 | cut -d ' ' -f -1`

for f in `ls anim/${prefix}*.png`
do
  convert $f -background white -extent ${w}x${h} $f
done

# create a blank canvas of that size
convert -dispose none -delay 0 \
          -size ${w}x${h} xc:white +antialias \
          -fill White \
        -dispose previous -delay 20 \
          anim/${prefix}*.png  \
        -loop 0 -coalesce $L $1

