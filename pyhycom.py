"""
pyhycom.py

A Python interface to HYCOM files.
"""
import numpy as np
import gzip
import os

def open_a_file(filename, mode):
    """
    Open the file using gzip if if is a gzipped file (ending in .gz),
    otherwise, use regular Python read.
    """
    if filename[-3:] == '.gz':
        file = gzip.open(filename, mode=mode)
    else:
        file = open(filename[:-1]+'a',mode=mode)
    return file


def get_b_filename(filename):
    """
    Return the name of the corresponding HYCOM "b" file.
    If it is a gzipped file, replace the .a.gz with .b,
    otherwise, replace .a with .b.
    """
    if filename[-3:] == '.gz':
        bfilename = filename[:-4] + 'b'
    else:
        bfilename = filename[:-1]+'b'
    return bfilename


def thickness2depths(dz):
    """
    Convert layer thickness values to depths.
    return tuple of (z_bottom, z_center, z_top) of each layer.
    """
    z_bottom = 0.0 * dz
    z_center = 0.0 * dz
    z_top = 0.0 * dz
    kdm, jdm, idm = dz.shape

    for k in range(1, kdm):
        z_top[k,:,:] = z_top[k-1,:,:] + dz[k-1,:,:]

    z_bottom[0:kdm-1,:,:] = z_top[1:kdm,:,:]
    z_bottom[kdm-1,:,:] = z_bottom[kdm-2,:,:] + dz[k,:,:]

    z_center = 0.5 * (z_top + z_bottom)

    return (z_bottom, z_center, z_top)


def getTextFile(filename):
    """
    Returns a list where each element contains text from each line
    of given text file.
    """
    return [line.rstrip() for line in open(filename,'r').readlines()]


def getDims(filename):
    """
    Returns HYCOM domain dimensions for a given
    archive or regional.grid .a file.
    NOTE: This does NOT work for regional.depth files.
    """
    f = getTextFile(get_b_filename(filename))
    #
    idmFound = False
    jdmFound = False
    for line in f:
        if 'idm' in line:
            idm = int(line.split()[0])
            idmFound = True
        if 'jdm' in line:
            jdm = int(line.split()[0])
            jdmFound = True
        if idmFound and jdmFound:break
    #
    if 'arch' in filename:
        kdm = int(f[-1].split()[4])
        return (kdm,jdm,idm)
    else:
        return (jdm,idm)


def getFieldIndex(field,filename):
    """
    Function description
    """
    f = getTextFile(get_b_filename(filename))
    if 'arch' in filename:f = f[10:]
    if 'grid' in filename:f = f[3:]
    fieldIndex = []
    for line in f:
        if field == line.split()[0].replace('.','').replace(':',''):
            fieldIndex.append(f.index(line))
    return fieldIndex


def getNumberOfRecords(filename):
    """
    Function description
    """
    f = getTextFile(get_b_filename(filename))
    if 'arch' in filename:
        f = f[10:]; return len(f)
    if 'grid' in filename:
        f = f[3:]; return len(f)
    if 'depth' in filename:
        return 1
    if 'restart' in filename:
        f = f[2:]; return len(f)



def getBathymetry(filename,undef=np.nan):
    """
    Reads a HYCOM bathymetry file (e.g., regional.depth.a)
    and return the bathymetry field.
    Will get dims from regional.grid.a.
    """
    import numpy as np
    jdm,idm = dims
    file = open_a_file(filename, mode='rb')
    ## The data are stored as float32, which has 4 bytes per each value.
    data = file.read(idm*jdm*4)
    field = np.reshape(np.fromstring(data,dtype='float32',count=idm*jdm).byteswap(),(jdm,idm))
    file.close()
    field[field>2**99] = undef
    return field


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
    import numpy as np
    from os.path import getsize

    # Get domain dimensions:
    dims = getDims(filename)

    if dims.__len__() == 3:
        kdm = dims[0]
        jdm = dims[1]
        idm = dims[2]
    if dims.__len__() == 2:
        kdm = 0
        jdm = dims[0]
        idm = dims[1]

    reclen = 4*idm*jdm                                   # Record length in bytes
    ## HYCOM binary data is written out in chunks/"words" of multiples of 4096*4 bytes.
    ## In general, the length of one level of one variable (reclen) will fall between
    ## consecutive multiples of the wordlen. The data is padded to bring the volume
    ## up to the next multiple. The "pad" value below is how many bytes are needed to do this.
    wordlen = 4096*4
    pad = wordlen * np.ceil(reclen / wordlen) - reclen   # Pad size in bytes
    fieldRecords = getFieldIndex(field,filename)         # Get field record indices
    fieldAddresses = np.array(fieldRecords)*(reclen+pad) # Address in bytes

    file = open_a_file(filename,mode='rb') # Open file

    # Read field records:
    if fieldAddresses.size == kdm: # 3-d field
        field = np.zeros((kdm,jdm,idm))
        if layers is None:
            layers = []

        ## Figure out how many layers I need to read from the file.
        if len(layers) > 0:
            kmax = max(np.max(layers),kdm-1)
        else:
            kmax = kdm

        ## Read through layers sequentially.
        for k in range(kmax):
            file.seek(int(fieldAddresses[k]),0) # Move to address
            if len(layers) < 1:
                data = file.read(idm*jdm*4)
                field[k,:,:] = np.reshape(np.fromstring(data,dtype='float32',count=idm*jdm),(jdm,idm)).byteswap()
            else:
                if k in layers:   ## Levels are 1 to kdm. Python indices are zero based.
                    data = file.read(idm*jdm*4)
                    field[k,:,:] = np.reshape(np.fromstring(data,dtype='float32',count=idm*jdm),(jdm,idm)).byteswap()

        ## Keep only tha layers that were specified. (The others would be all zeros.)
        if len(layers) > 0:
            field = field[layers,:,:]

        if not x_range is None:
            field = field[:,:,x_range]

        if not y_range is None:
            field = field[:,y_range,:]

    else: # 2-d field
        file.seek(int(fieldAddresses[0]),0)     # Move to address
        data = file.read(idm*jdm*4)
        field = np.reshape(np.fromstring(data,dtype='float32',count=idm*jdm),(jdm,idm)).byteswap()

        if not x_range is None:
            field = field[:,x_range]

        if not y_range is None:
            field = field[y_range,:]


    #field = field.byteswap() # Convert to little-endian

    file.close()
    field[field>2**99] = undef

    return field


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

    from scipy.interpolate import NearestNDInterpolator

    ## Handle field_list if it is just a string of a single field.
    if not type(field_list) is list:
        field_list = [field_list]
    field_list = ['thknss'] + field_list

    ## Handle points if only one point specified.
    if not type(points[0]) is list:
        points = [points]
    points = np.array(points)

    ## Get regional.grid.a file.
    gridfilename = os.path.dirname(filename) + '/regional.grid.a'

    ## Get lat/lon and bounds for the points.
    min_lon = np.min(points[:,0])
    max_lon = np.max(points[:,0])
    min_lat = np.min(points[:,1])
    max_lat = np.max(points[:,1])
    lon = getField('plon', gridfilename, np.nan)
    lat = getField('plat', gridfilename, np.nan)

    ## Add buffer region of 1 deg, in case all the points specified are too close together
    ## in which case, x_range and/or y_range may end up empty below.
    x_range = [x for x in range(lon.shape[1]) if np.min(lon[:,x]) > min_lon-1.01 and np.max(lon[:,x]) < max_lon+1.01]
    y_range = [x for x in range(lat.shape[0]) if np.min(lat[x,:]) > min_lat-1.01 and np.max(lat[x,:]) < max_lat+1.01]
    lon = lon[y_range,:][:,x_range]
    lat = lat[y_range,:][:,x_range]

    ## Process each field.
    field_profile_list = []
    for field_name in field_list:
        field_data = getField(field_name, filename, undef=undef, layers=None
                    , x_range = x_range, y_range = y_range)

        field_profile = np.zeros([field_data.shape[0],points.shape[0]])

        for kk in range(field_data.shape[0]):
            interp = NearestNDInterpolator((lon.flatten(),lat.flatten()),field_data[kk,:,:].flatten())
            field_profile[kk,:] = interp(points) #[points[:,0],points[:,1]])

        field_profile_list += [field_profile]

    ## Get depth from thickness.
    field_profile_list[0] /= 9806.0

    depth_bottom = 1.0*field_profile_list[0] #/ 9806.0
    for k in range(1, field_data.shape[0]):
        depth_bottom[k,:] = depth_bottom[k-1,:] + depth_bottom[k,:]

    depth = 0.0*depth_bottom
    depth[0,:] = depth_bottom[0,:] / 2.0
    for k in range(1, field_data.shape[0]):
        depth[k,:] = 0.5*(depth_bottom[k-1,:] + depth_bottom[k,:])

    FOUT={}
    FOUT['depth_bottom_of_layer'] = depth_bottom
    FOUT['depth_middle_of_layer'] = depth
    FOUT['lon'] = 0.0*depth
    FOUT['lat'] = 0.0*depth
    for k in range(field_data.shape[0]):
        FOUT['lon'][k,:] = points[:,0]
        FOUT['lat'][k,:] = points[:,1]

    for ii in range(len(field_list)):
        FOUT[field_list[ii]] = field_profile_list[ii]

    return FOUT




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

    import glob
    from scipy.interpolate import NearestNDInterpolator

    ## Handle field_list if it is just a string of a single field.
    if not type(field_list) is list:
        field_list = [field_list]
    field_list = ['thknss'] + field_list

    ## Handle points if only one point specified.
    points = np.array([trajectory['lon'].tolist(),trajectory['lat'].tolist()]).T

    ## Get regional.grid.a file.
    gridfilename = (dir + '/regional.grid.a')

    ## Get lat/lon and bounds for the points.
    min_lon = np.min(points[:,0])
    max_lon = np.max(points[:,0])
    min_lat = np.min(points[:,1])
    max_lat = np.max(points[:,1])
    lon = getField('plon', gridfilename, np.nan)
    lat = getField('plat', gridfilename, np.nan)

    ## Add buffer region of 1 deg, in case all the points specified are too close together
    ## in which case, x_range and/or y_range may end up empty below.
    x_range = [x for x in range(lon.shape[1]) if np.min(lon[:,x]) > min_lon-1.01 and np.max(lon[:,x]) < max_lon+1.01]
    y_range = [x for x in range(lat.shape[0]) if np.min(lat[x,:]) > min_lat-1.01 and np.max(lat[x,:]) < max_lat+1.01]
    lon = lon[y_range,:][:,x_range]
    lat = lat[y_range,:][:,x_range]


    ## Initialize fields.
    field_profile_list = []
    for field_name in field_list:
        field_profile = np.zeros([nz,points.shape[0]])
        field_profile_list += [field_profile]


    ## Process each field at each time.
    for tt in range(len(trajectory['datetime'])):
        filename = glob.glob(trajectory['datetime'][tt].strftime(dir + '/archv.%Y_%j_%H.a*'))[0]
        print(filename)

        ffff=-1
        for field_name in field_list:
            ffff+=1
            field_data = getField(field_name, filename, undef=undef, layers=None
                        , x_range = x_range, y_range = y_range)

            for kk in range(nz):
                interp = NearestNDInterpolator((lon.flatten(),lat.flatten()),field_data[kk,:,:].flatten())
                field_profile_list[ffff][kk,tt] = interp(points[tt,:]) #[points[:,0],points[:,1]])

    ## Get depth from thickness.
    field_profile_list[0] /= 9806.0

    depth_bottom = 1.0*field_profile_list[0] #/ 9806.0
    for k in range(1, field_data.shape[0]):
        depth_bottom[k,:] = depth_bottom[k-1,:] + depth_bottom[k,:]

    depth = 0.0*depth_bottom
    depth[0,:] = depth_bottom[0,:] / 2.0
    for k in range(1, field_data.shape[0]):
        depth[k,:] = 0.5*(depth_bottom[k-1,:] + depth_bottom[k,:])

    FOUT={}
    FOUT['depth_bottom_of_layer'] = depth_bottom
    FOUT['depth_middle_of_layer'] = depth
    FOUT['lon'] = 0.0*depth
    FOUT['lat'] = 0.0*depth
    for k in range(field_data.shape[0]):
        FOUT['lon'][k,:] = points[:,0]
        FOUT['lat'][k,:] = points[:,1]

    for ii in range(len(field_list)):
        FOUT[field_list[ii]] = field_profile_list[ii]

    return FOUT


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

    from netCDF4 import Dataset
    from scipy.interpolate import NearestNDInterpolator

    ## Handle field_list if it is just a string of a single field.
    if not type(field_list) is list:
        field_list = [field_list]

    ## Handle points if only one point specified.
    points = np.array([trajectory['lon'].tolist(),trajectory['lat'].tolist()]).T


    ## Get lat/lon and bounds for the points.
    min_lon = np.min(points[:,0])
    max_lon = np.max(points[:,0])
    min_lat = np.min(points[:,1])
    max_lat = np.max(points[:,1])


    ## Initialize fields.
    field_transect_list = []
    for field_name in field_list:
        field_transect = np.zeros(points.shape[0])
        field_transect_list += [field_transect]


    ## Process each field at each time.
    for tt in range(len(trajectory['datetime'])):
        filename = trajectory['datetime'][tt].strftime(dir + '/wrfout_'+DOM+'_%Y-%m-%d_%H:00:00')
        print(filename)

        DS=Dataset(filename)
        lon = DS['XLONG'][:]
        lat = DS['XLAT'][:]

        ## Add buffer region of 1 deg, in case all the points specified are too close together
        ## in which case, x_range and/or y_range may end up empty below.
        #x_range = [x for x in range(lon.shape[1]) if np.min(lon[:,x]) > min_lon-1.01 and np.max(lon[:,x]) < max_lon+1.01]
        #y_range = [x for x in range(lat.shape[0]) if np.min(lat[x,:]) > min_lat-1.01 and np.max(lat[x,:]) < max_lat+1.01]
        #lon = lon[y_range,:][:,x_range]
        #lat = lat[y_range,:][:,x_range]

        ffff=-1
        for field_name in field_list:
            ffff+=1
            field_data = DS[field_name][:]

            interp = NearestNDInterpolator((lon.flatten(),lat.flatten()),field_data[0,:,:].flatten())
            field_transect_list[ffff][tt] = interp(points[tt]) #[points[:,0],points[:,1]])
        DS.close()

    FOUT={}
    FOUT['lon'] = 0.0*field_transect_list[0]
    FOUT['lat'] = 0.0*field_transect_list[0]
    FOUT['lon'] = points[:,0]
    FOUT['lat'] = points[:,1]

    for ii in range(len(field_list)):
        FOUT[field_list[ii]] = field_transect_list[ii]

    return FOUT



#
########################################################################
#
def ab2nc(filename):
    #
    """
    A function that converts a given hycom binary .a file into an equivalent .nc file.

    Module requirements: numpy,netCDF4,matplotlib.dates

    THIS FUNCTION IS CURRENTLY BROKEN. TODO: Fix this function.
    """
    #
    import numpy as np
    #from pycom import getHycomField
    from netCDF4 import Dataset
    #
    def str2(n):
        if n<10:return '0'+str(n)
        return str(n)
    #
    if filename.rfind('regional.grid.a')>-1:
        #
        # Read regional.grid.b file:
        f=open(filename[:-1]+'b','r')
        file_content=[line.rstrip() for line in f.readlines()]
        f.close()
        idm=int(file_content[0][0:5]) # Get X-dim size
        jdm=int(file_content[1][0:5]) # Get Y-dim size
        dims=(jdm,idm)
        #
        # Get field names:
        fields=[]
        for line in file_content[3:]:
            fields.append(line[0:4])
        #
        ncfile=Dataset(filename[:-1]+'nc','w',format='NETCDF3_CLASSIC') # Open file
        #
        ncfile.createDimension('X',size=idm) # Create x-dim
        ncfile.createDimension('Y',size=jdm) # Create y-dim
        #
        # Write fields into file:
        nc_field=ncfile.createVariable('X',datatype='f4',dimensions=('X')); nc_field[:]=np.arange(idm)
        nc_field=ncfile.createVariable('Y',datatype='f4',dimensions=('Y')); nc_field[:]=np.arange(jdm)
        for field in fields:
            ab_field=getHycomField('2d',fields.index(field)+1,dims,filename,np.NaN)
            nc_field=ncfile.createVariable(field,datatype='f4',dimensions=('Y','X'))
            nc_field[:]=ab_field
        #
        ncfile.close() # Close file
    #
    #--------------------------------------------------------------------------------------------------
    #
    if filename.rfind('arch')>-1:
        #
        from matplotlib.dates import num2date
        #
        # Read archv.b file:
        f=open(filename[:-1]+'b','r')
        file_content=[line.rstrip() for line in f.readlines()]
        f.close()
        idm=int(file_content[7][2:5])    # Get X-dim size
        jdm=int(file_content[8][2:5])    # Get Y-dim size
        kdm=int(file_content[-1][33:35]) # Get Z-dim size
        dims=(kdm,jdm,idm)
        #
        plon=getHycomField('2d',1,dims,'/home/disk/manta8/milan/esmf/application/SCRATCH/output/cpl_sst/regional.grid.a',np.NaN)
        plat=getHycomField('2d',2,dims,'/home/disk/manta8/milan/esmf/application/SCRATCH/output/cpl_sst/regional.grid.a',np.NaN)
        #
        # Compute a current datetime instance:
        day_in_year=int(filename[-8:-5])
        hour=filename[-4:-2]
        year=int(filename[-13:-9])
        day_since_0001_01_01=(year-1)*365+year/4-year/100+year/400+day_in_year
        now=num2date(day_since_0001_01_01)
        date_string=str(now.year)+str2(now.month)+str2(now.day)+'_'+hour
        #
        print('Working on','archv.'+date_string+'.nc')
        ncfile=Dataset('archv.'+date_string+'.nc','w',format='NETCDF3_CLASSIC') # Open file
        #
        ncfile.createDimension('Longitude',size=idm) # Create x-dim
        ncfile.createDimension('Latitude',size=jdm)  # Create y-dim
        ncfile.createDimension('Depth',size=kdm)     # Create z-dim
        #
        # Write 2-d fields into file:
        fields=[]
        for line in file_content[10:22]:
            fields.append(line[0:8].rstrip())
        nc_field=ncfile.createVariable('Longitude',datatype='f4',dimensions=('Longitude'))
        nc_field[:]=np.linspace(np.min(plon[0,:]),np.max(plon[0,:]),idm)
        nc_field=ncfile.createVariable('Latitude',datatype='f4',dimensions=('Latitude'))
        nc_field[:]=plat[:,0]
        nc_field=ncfile.createVariable('Depth',datatype='f4',dimensions=('Depth')); nc_field[:]=np.arange(kdm)
        for field in fields:
            ab_field=getHycomField('2d',fields.index(field)+1,dims,filename,np.NaN)
            nc_field=ncfile.createVariable(field,datatype='f4',dimensions=('Latitude','Longitude'))
            nc_field[:]=ab_field
        #
        # Write 3-d fields into file:
        fields=[]
        for line in file_content[23:29]:
            fields.append(line[0:8].rstrip())
        for field in fields:
            ab_field=getHycomField('3d',fields.index(field)+1,dims,filename,np.NaN)
            nc_field=ncfile.createVariable(field,datatype='f4',dimensions=('Depth','Latitude','Longitude'))
            nc_field[:]=ab_field
        ncfile.close() # Close file
    #
    #--------------------------------------------------------------------------------------------------
    #
    if filename.rfind('forcing')>-1:
        #
        # Read forcing.[field].b file:
        f=open(filename[:-1]+'b','r')
        file_content=[line.rstrip() for line in f.readlines()]
        f.close()
        idm=int(file_content[4][9:12])  # Get X-dim size
        jdm=int(file_content[4][14:17]) # Get Y-dim size
        dims=(jdm,idm)
        print('Working on file: '+filename,dims)
        #
        plon=getHycomField('2d',1,dims,'/home/disk/manta8/sophie/hycom/PACa0.08/SCRATCH/regional.grid.a',np.NaN)
        plat=getHycomField('2d',2,dims,'/home/disk/manta8/sophie/hycom/PACa0.08/SCRATCH/regional.grid.a',np.NaN)
        #
        ncfile=Dataset(filename[:-1]+'nc','w',format='NETCDF3_CLASSIC') # Open file
        #
        ncfile.createDimension('Longitude',size=idm) # Create x-dim
        ncfile.createDimension('Latitude',size=jdm)  # Create y-dim
        ncfile.createDimension('Time',size=0)        # Create t-dim (unlimitted)
        #
        nc_field=ncfile.createVariable(filename[-8:-2],\
                                       datatype='f4',\
                                       dimensions=('Time','Latitude','Longitude'))
        #
        file_content=file_content[5:]
        record=0
        for line in file_content:
            ab_field=getHycomField('2d',record+1,dims,filename,np.NaN)
            print(filename,record,np.min(ab_field),np.max(ab_field))
            nc_field[record,:,:]=ab_field[:,:]
            record=record+1
        ncfile.close() # Close file
        #

########################################################################
############### Mixed Layer Depth Functions ############################
########################################################################

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
    import os.path

    dims2 = getDims(os.path.dirname(filename) + '/regional.grid.b')
    t = getField('temp', filename)
    s = getField('salin', filename)
    dz = getField('thknss', filename) / 9806.0
    z_bottom, z_center, z_top = thickness2depths(dz)

    mldt = np.nan*np.zeros(dims2)
    mlds = np.nan*np.zeros(dims2)
    mld = np.nan*np.zeros(dims2)

    for jj in range(dims2[0]):
        for ii in range(dims2[1]):
            if not np.isnan(t[0,jj,ii]):
                mldt[jj,ii] = mixedLayerDepthT(t[:,jj,ii],z_center[:,jj,ii],delT, ref_depth=10.0)
                mlds[jj,ii] = mixedLayerDepthS(s[:,jj,ii],z_center[:,jj,ii],delS, ref_depth=10.0)
                mld[jj,ii] = min(mldt[jj,ii],mlds[jj,ii])

    return (mld, mldt, mlds)


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

    ref_temp = T[0]
    k_begin = 1

    if ref_depth > 0.1:
        ref_temp = np.interp(ref_depth, d, T)
        k_begin = int(np.ceil(np.interp(ref_depth, d, np.arange(len(d)))))

    Tm=ref_temp-delT
    for k in range(k_begin,len(T),1):
        if np.isfinite(T[k]):
            if T[k]<Tm:
                mld=(abs(T[k-1]-Tm)*d[k] \
                     +abs(Tm-T[k])*d[k-1])\
                     /(T[k-1]-T[k])
                return mld
        else:
            return d[k-1]
    return d[len(T)-1]


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

    ref_sal = S[0]
    k_begin = 1

    if ref_depth > 0.1:
        ref_sal = np.interp(ref_depth, d, S)
        k_begin = int(np.ceil(np.interp(ref_depth, d, np.arange(len(d)))))

    Sm=ref_sal+delS
    for k in range(k_begin,len(S),1):
        if np.isfinite(S[k]):
            if S[k]>Sm:
                mld=(abs(S[k-1]-Sm)*d[k] \
                     +abs(Sm-S[k])*d[k-1])\
                     /(S[k]-S[k-1])
                return mld
        else:
            return d[k-1]
    return d[len(S)-1]


#
########################################################################
#
def str2(number):
    string=str(number)
    if len(string)<2:string='0'+string
    return string
#
