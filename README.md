# pyhycom
PYTHON Interface for HYCOM

Originally written by Milan Curcic (University of Miami).
Modified by Brandon Kerns at the University of Washington (bkerns@uw.edu)

See the examples directory for example usage for a simulated transect
along a track (e.g., ship, glider, saildrone)

Many of the functions require regional.grid.[ab] to be present
in the same directory.

In the code, the functions have a documentaton section near the top.
These documentation sections are reproduced below for the main functions.

```Python
def getDims(filename):
    """
    Returns HYCOM domain dimensions for a given
    archive or regional.grid .a file.
    NOTE: This does NOT work for regional.depth files.
    """
```

```Python
def getBathymetry(filename,undef=np.nan):
    """
    Reads a HYCOM bathymetry file (e.g., regional.depth.a)
    and return the bathymetry field.
    Will get dims from regional.grid.a.
    """
```

```Python
def getField(field,filename,undef=np.nan,layers=None,x_range=None,y_range=None):
    """
    A function to read hycom raw binary files (regional.grid.a, archv.*.a and forcing.*.a supported),
    and interpret them as numpy arrays.

    ## BK added layers option to get a set of specified layers instead of the full file.
    ## layers is zero based. Leave it as None (or set it to []) to get all layers.

    ## TODO: Have x_range, and y_range only access the needed data.
             Right now, it will read the entire domain then subset it.
             (For layers, it will skip over the layers not specified).
    """
```

```Python
def get_vertical_profiles_at_points(field_list,filename,points,undef=np.nan):
    """
    F = get_vertical_profiles_at_points(field_list,filename,points,undef=np.nan)

    field_list is a list of field names to get. Alternatively, a string with a single field name.
    filename is the .a file.
    points is a 2D array or list of lists, with each row being a lon and lat.
    It can be just [lon,lat] to get a single profile.

    The script uses nearest neighbor interpolation. This avoids having to deal
    with different vertical coordinates at adjacent points.

    The function will return a dict containing 2D arrays for each variable.
    """
```

```Python
def get_vertical_profiles(field_list,dir,trajectory,undef=np.nan, nz=41, atm_fields=None):
    """
    F = get_vertical_profiles(field_list,dir,trajectory,undef=np.nan)

    field_list is a list of field names to get. Alternatively, a string with a single field name.
    filename is the .a file.

    A trajectory dictionary has keys 'datetime','lon','lat'

    nz is the number of layers to get.

    The script uses nearest neighbor interpolation. This avoids having to deal
    with different vertical coordinates at adjacent points.

    The function will return a dict containing 2D arrays for each variable.
    """
```

```Python
def get_wrf_surface_transect(field_list,dir,trajectory,undef=np.nan, DOM='d01'):

    """
    F = get_wrf_surface_transect(field_list,dir,trajectory,undef=np.nan)

    field_list is a list of field names to get. Alternatively, a string with a single field name.
    filename is the .a file.

    A trajectory dictionary has keys 'datetime','lon','lat'

    The script uses nearest neighbor interpolation. This avoids having to deal
    with different vertical coordinates at adjacent points.

    The function will return a dict containing 1D arrays for each variable.
    """
```

```Python
def getMixedLayerDepth(filename, delT=0.2, delS=0.03, ref_depth=10):
    """
    One definition of mixed layer depth is first level when the temperature
    or salinity difference is greater than a threshold, relative to a reference depth.
    This function first calculates the temperature and salinity based
    mixed layer depths (mldt and mlds, respectively),
    then returns the one that closer to the surface as
    "the" mixed layer depth (mld).
    By default, it will use the threshold of 0.2 C for temperature
    and 0.03 PSU for salinity.

    Returned values are a tuple of (mld,mldt,mlds)
    """
```

```Python
def mixedLayerDepthT(T,d,delT, ref_depth=10.0):
    """
    Computes mixed layer depth given a temperature and depth
    profiles and temperature difference criterion.
    Uses linear interpolation to find mixed layer depth between
    two discrete levels. If criterion is not satisfied,
    returns the last element of the depth list.

    Input arguments:
    T    :: list of vertical temperature profile
    d    :: list of depth values, of same length as T
    delT :: float; Temperature difference criterion in K
    """
```

```Python
def mixedLayerDepthS(S,d,delS, ref_depth=10.0):
    """
    Computes mixed layer depth given a temperature and depth
    profiles and temperature difference criterion.
    Uses linear interpolation to find mixed layer depth between
    two discrete levels. If criterion is not satisfied,
    returns the last element of the depth list.

    Input arguments:
    S    :: list of vertical salinity profile
    d    :: list of depth values, of same length as T
    delS :: float; Salinity difference criterion in K
    """
```
