#!/bin/bash

## Download sample data to the directory data/
WGET='wget -r -np -nH --cut-dirs 3'
HTTP=https://orca.atmos.washington.edu/~bkerns/code/awovispy/data

## HYCOM grid and bathymetry files.
$WGET $HTTP/regional.grid.a
$WGET $HTTP/regional.grid.b
$WGET $HTTP/regional.depth.a
$WGET $HTTP/regional.depth.b

## HYCOM 3D data files.
for ext in a.gz b txt
do

  for HH in 00 03 06 09 12 15 18 21
  do

    $WGET $HTTP/archv.2018_050_${HH}.$ext
    $WGET $HTTP/archv.2018_051_${HH}.$ext

  done

  $WGET $HTTP/archv.2018_052_00.$ext

done


## WRF files for including met fields.
for HH in 00 03 06 09 12 15 18 21
do

  $WGET $HTTP/wrfout_d01_2018-02-19_${HH}:00:00
  $WGET $HTTP/wrfout_d01_2018-02-20_${HH}:00:00

done

$WGET $HTTP/wrfout_d01_2018-02-21_00:00:00


echo 'Download complete.'

exit 0
