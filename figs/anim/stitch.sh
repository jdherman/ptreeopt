#!/bin/bash

dir="img"
# get largest dimensions
prefix="tree"
# prefix="folsom"
# prefix="convergence"

# do not optimize for tree
L=""
# L="-layers OptimizeFrame"
# L="-layers optimize"

w=`identify -format "%w %f\n" ${dir}/${prefix}*.png | sort -n -r -k 2 | head -n 1 | cut -d ' ' -f -1`
h=`identify -format "%h %f\n" ${dir}/${prefix}*.png | sort -n -r -k 2 | head -n 1 | cut -d ' ' -f -1`

# for f in `ls ${dir}/${prefix}*.png`
# do
#   convert $f -background white -extent ${w}x${h} $f
# done

# create a blank canvas of that size
convert -dispose none -delay 0 \
          -size ${w}x${h} xc:white +antialias \
          -fill white \
        -dispose previous -delay 20 \
          ${dir}/${prefix}*.png  \
        -loop 0 -coalesce $L ${prefix}.gif

