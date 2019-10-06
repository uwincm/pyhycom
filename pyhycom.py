"""
pyhycom.py

A Python interface to HYCOM files.
"""

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
    """
    f = getTextFile(filename[:-1]+'b')
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
    f = getTextFile(filename[:-1]+'b')
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
    f = getTextFile(filename[:-1]+'b')
    if 'arch' in filename:
        f = f[10:]; return len(f)
    if 'grid' in filename:
        f = f[3:]; return len(f)
    if 'depth' in filename:
        return 1
    if 'restart' in filename:
        f = f[2:]; return len(f)



def getBathymetry(filename,dims,undef):
    """
    Reads a HYCOM bathymetry file and return the bathymetry field.
    """
    import numpy as np
    jdm,idm = dims
    file = open(filename[:-1]+'a',mode='rb')
    field = np.reshape(np.fromfile(file,dtype='float32',count=idm*jdm).byteswap(),(jdm,idm))
    file.close()
    field[field>2**99] = undef
    return field


def getField(field,filename,undef,layers=None,x_range=None,y_range=None):
    """
    A function to read hycom raw binary files (regional.grid.a, archv.*.a and forcing.*.a supported),
    and interpret them as numpy arrays.

    ## BK added layers option to get a set of specified layers instead of the full file.
    ## layers is zero based. Leave it as None (or set it to []) to get all layers.
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
    nrecs = getNumberOfRecords(filename)                 # Total number of records
    pad = (getsize(filename)-reclen*nrecs)/nrecs         # Pad size in bytes
    fieldRecords = getFieldIndex(field,filename)         # Get field record indices
    fieldAddresses = np.array(fieldRecords)*(reclen+pad) # Address in bytes

    file = open(filename,mode='rb') # Open file

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
                field[k,:,:] = np.reshape(np.fromfile(file,dtype='float32',count=idm*jdm),(jdm,idm)).byteswap()
            else:
                if k in layers:   ## Levels are 1 to kdm. Python indices are zero based.
                    field[k,:,:] = np.reshape(np.fromfile(file,dtype='float32',count=idm*jdm),(jdm,idm)).byteswap()

        ## Keep only tha layers that were specified. (The others would be all zeros.)
        if len(layers) > 0:
            field = field[layers,:,:]

        if not x_range is None:
            field = field[:,:,x_range]

        if not y_range is None:
            field = field[:,y_range,:]

    else: # 2-d field
        file.seek(int(fieldAddresses[0]),0)     # Move to address
        field = np.reshape(np.fromfile(file,dtype='float32',count=idm*jdm),(jdm,idm)).byteswap()
    #field = field.byteswap() # Convert to little-endian

    file.close()
    field[field>2**99] = undef

    return field
#
########################################################################
#
def ab2nc(filename):
    #
    """
    A function that converts a given hycom binary .a file into an equivalent .nc file.

    Module requirements: numpy,netCDF4,matplotlib.dates
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
#
########################################################################
#
def mixedLayerDepth(T,d,delT):
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
    Tm=T[0]-delT
    for k in range(1,len(T),1):
        if T[k]<Tm:
            mld=(abs(T[k-1]-Tm)*d[k] \
                +abs(Tm-T[k])*d[k-1])\
                /(T[k-1]-T[k])
            return mld
    return d[len(T)-1]
#
########################################################################
#
def str2(number):
    string=str(number)
    if len(string)<2:string='0'+string
    return string
#
