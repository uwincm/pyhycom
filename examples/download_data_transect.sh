#!/bin/bash

## Download sample data to the directory data/
WGET='wget -r -np -nH --cut-dirs 3'
HTTP=https://orca.atmos.washington.edu/~bkerns/code/pyhycom/data

## HYCOM grid and bathymetry files.
$WGET $HTTP/regional.grid.a
$WGET $HTTP/regional.grid.b
$WGET $HTTP/regional.depth.a
$WGET $HTTP/regional.depth.b

## HYCOM 3D data files.
for ext in a.gz b txt
do
  for HH in 00 06 12 18
  do
    for DOY in {091..101}
    do
      $WGET $HTTP/archv.2018_${DOY}_${HH}.$ext
      $WGET $HTTP/archv.2018_${DOY}_${HH}.$ext
    done
  done
  $WGET $HTTP/archv.2018_102_00.$ext
done


## WRF files for including met fields.
for HH in 00 06 12 18
do
  for DD in {01..11}
  do
    $WGET $HTTP/wrfout_d01_2018-04-${DD}_${HH}:00:00
  done
done

$WGET $HTTP/wrfout_d01_2018-04-12_00:00:00


echo 'Download complete.'

exit 0
