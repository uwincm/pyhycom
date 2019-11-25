import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import datetime as dt
import cmocean

from context import pyhycom


"""
This Python script creates the image "transect.png"
(copied to transect.bak.png)
Test run it to make sure you get the file.

See the "Set Trajectory" section below to edit the path.
The data HYCOM data files can be downloaded from here:

"""

def plot_map_background(plot_area = [-180, 180, -70, 70], ax = plt.gca()
                        , projection = 'cyl', resolution = 'l'
                        , draw_meridians = np.arange(0,360,10), meridians_line_width = 0.25
                        , meridians_labels = [0,0,0,1], meridians_fontsize=10
                        , meridians_dashes = [4,2]
                        , draw_parallels = np.arange(-90,90,10), parallels_line_width = 0.25
                        , parallels_labels = [1,0,0,0], parallels_fontsize=10
                        , parallels_dashes = [4,2]
                        , coastline_line_width = 0.5
                        , countries_line_width = 0.25
                        , x_tick_rotation = 45.0
                        , y_tick_rotation = 45.0):

    map1=Basemap(projection = projection,
                 llcrnrlon = plot_area[0],
                 urcrnrlon = plot_area[1],
                 llcrnrlat = plot_area[2],
                 urcrnrlat = plot_area[3],
                 resolution = resolution, ax = ax)

    # draw lat/lon grid lines degrees.
    meridians = map1.drawmeridians(draw_meridians, linewidth = meridians_line_width
                                   , labels = meridians_labels, fontsize = meridians_fontsize
                                   , dashes = meridians_dashes)
    parallels = map1.drawparallels(draw_parallels, linewidth = parallels_line_width
                                   , labels = parallels_labels, fontsize = parallels_fontsize
                                   , dashes = parallels_dashes)
    map1.drawcoastlines(linewidth = coastline_line_width)
    map1.drawcountries(linewidth = countries_line_width)

    for m in meridians:
        try:
            meridians[m][1][0].set_rotation(x_tick_rotation)
        except:
            pass

    for p in parallels:
        try:
            parallels[p][1][0].set_rotation(y_tick_rotation)
        except:
            pass

    return map1

def draw_color_bar(h, label='', fig = plt.gcf(), ax=plt.gca()):

    # Color bar stuff.
    pos1 = ax.get_position()
    pos11 = [pos1.x0 - 0.03, pos1.y0,  pos1.width, pos1.height]
    ax.set_position(pos11)
    pos2 = [0.91, 0.15, 0.02, 0.5]

    cbax = fig.add_axes(pos2)
    cbar=plt.colorbar(h, cax=cbax)
    cbar.set_label(label)

    return cbar


def calc_trajectory(speed, duration_hours, direction_ccw_from_east
        , prev_trajectory=None, dt_start=None, lon_start=None, lat_start=None):
    ##
    ## A trajectory dictionary has keys 'datetime','lon','lat'
    ##
    factor = 3600.0 / 111000.0   # m/s --> deg / hour at equator. (For lon, adjust for latitude below.)
    direction_radians = direction_ccw_from_east * 3.14159 / 180.0

    if prev_trajectory is None:
        lat_radians = lat_start * 3.14159 / 180.0
        hours = range(int(duration_hours)+1)
        dt_track = [dt_start + dt.timedelta(hours=x) for x in hours]
        lon_track = np.array([lon_start + factor * np.cos(lat_radians) * speed * x * np.cos(direction_radians) for x in hours])
        lat_track = np.array([lat_start + factor * speed * x * np.sin(direction_radians) for x in hours])
    else:
        lat_radians = prev_trajectory['lat'][-1] * 3.14159 / 180.0
        dt_start = prev_trajectory['datetime'][-1]
        hours = range(1,int(duration_hours)+1)
        dt_new_track = [dt_start + dt.timedelta(hours=x) for x in hours]
        dt_track = prev_trajectory['datetime'] + dt_new_track
        lon_track = np.append(prev_trajectory['lon']
                    , np.array([prev_trajectory['lon'][-1] + factor * np.cos(lat_radians) * speed * x * np.cos(direction_radians) for x in hours]))
        lat_track = np.append(prev_trajectory['lat']
                    , np.array([prev_trajectory['lat'][-1] + factor * speed * x * np.sin(direction_radians) for x in hours]))

    ## Prepare output dictionary.
    F = {}
    F['datetime'] = dt_track
    F['lon'] = lon_track
    F['lat'] = lat_track
    return F

################################################################################

## !!!!!! Edit data paths with your system directories. !!!!!!
dir = '/home/orca/bkerns/models/uwincm-pacific-tropics-subtropics-2018/stitch_output'
fn_maps = '/home/orca/bkerns/models/uwincm-pacific-tropics-subtropics-2018/stitch_output/archv.2018_050_00.a'
fn_grid = '/home/orca/bkerns/models/uwincm-pacific-tropics-subtropics-2018/stitch_output/regional.grid.a'
plot_area = [230,250,-6,10]   # [lon1, lon2, lat1, lat2], lon is 0-360 for this HYCOM run.

## !!!!!! Initialize the trajectory you want. !!!!!!!
"""
This example has the trajectory starting at 4.5 N, 120.5W, moving south at 5 m/s for a day,
then moving east at 5 m/s for a day. add as many trajectory legs as you feel like.
(For now, only integer hours are allowed)
Note: For this run, HYCOM longitudes are 0 - 360.
"""

## Trajectory is a dictionary with fields datetime, lon, and lat.
## This code calls the calc_trajectory function above, but you can do it
## however you feel like.
trajectory = {'datetime': [dt.datetime(2018,2,19,0,0,0)]
            , 'lon': [360 - 120.5], 'lat': [4.5]}
##                          spd  hours direction   attach to this trajectory
trajectory = calc_trajectory(5.0, 24, -90.0, prev_trajectory = trajectory)
trajectory = calc_trajectory(5.0, 24, 0.0, prev_trajectory = trajectory)

################################################################################
## Read in the grid.
lon = pyhycom.getField('plon', fn_grid, np.nan)
lat = pyhycom.getField('plat', fn_grid, np.nan)
sst = pyhycom.getField("temp", fn_maps, np.NaN, layers=[0])[0,:,:]
sss = pyhycom.getField("salin", fn_maps, np.NaN, layers=[0])[0,:,:]

##
## Calculate Trajectories and transect profiles
##

## Get trajectories from each file, as needed.
#points = np.array([trajectory['lon'].tolist(),trajectory['lat'].tolist()]).T.tolist()
time_hours = np.array([(x - trajectory['datetime'][0]).total_seconds()/3600.0 for x in trajectory['datetime']])

#F = pyhycom.get_vertical_profiles_at_points(['temp','salin'],dir,trajectory,undef=np.nan)
F = pyhycom.get_vertical_profiles(['temp','salin'],dir,trajectory,undef=np.nan)
time_hours2d, dum = np.meshgrid(time_hours, F['depth_middle_of_layer'][:,0])

################################################################################
## --- The plots ---


# SST

cmap00 = plt.cm.jet
cmap0 = LinearSegmentedColormap.from_list('custom', cmap00(np.linspace(0.10, 1.0, 28)))
cmap = cmap0
levels = MaxNLocator(nbins=28).tick_values(14.0, 28.0)
norm = BoundaryNorm(levels, cmap0.N, clip=True)


fig = plt.figure(figsize=(6.5,5.5))
ax1 = fig.add_subplot(2,2,1)

map1=plot_map_background(plot_area = plot_area, ax=ax1)

H1 = plt.pcolormesh(lon, lat, sst
                           , cmap = cmap
                           , norm = norm)

plt.plot(trajectory['lon'],trajectory['lat'],'k',linewidth=2.0)
plt.text(trajectory['lon'][0],trajectory['lat'][0],'A')
plt.text(trajectory['lon'][-1],trajectory['lat'][-1],'B')

cbar = plt.colorbar(H1)
cbar.set_ticks(np.arange(16, 28, 2))

ax1.set_title('Temperature')

## Temp. Transect.
ax3 = fig.add_subplot(2,2,3)
H3 = plt.contourf(time_hours2d, F['depth_middle_of_layer'],F['temp'], levels=levels
                           , cmap = cmap
                           , norm = norm
                           , extend='both')

cbar = plt.colorbar(H3)
cbar.set_ticks(np.arange(16, 28, 2))

plt.text(0,5,'A')
plt.text(45,5,'B')

ax3.set_ylim([100,0])
ax3.set_xlabel('Time [h]')
ax3.set_title('Temperature')


# SSS (maps)
cmap00 = cmocean.cm.haline
cmap0 = LinearSegmentedColormap.from_list('custom', cmap00(np.linspace(0.0, 1.0, 15)))
cmap_sss = cmap0
levels = MaxNLocator(nbins=15).tick_values(34.0, 35.5)
norm_sss = BoundaryNorm(levels, cmap0.N, clip=True)


ax2 = fig.add_subplot(2,2,2)
map2=plot_map_background(plot_area = plot_area, ax=ax2)
H2 = plt.pcolormesh(lon, lat, sss
                           , cmap = cmap_sss
                           , norm = norm_sss)

plt.plot(trajectory['lon'],trajectory['lat'],'k',linewidth=2.0)
plt.text(trajectory['lon'][0],trajectory['lat'][0],'A')
plt.text(trajectory['lon'][-1],trajectory['lat'][-1],'B')

cbar= plt.colorbar(H2)
cbar.set_ticks(np.arange(32,39, 1))

ax2.set_title('Salinity')

## Temp. Transect.
ax4 = fig.add_subplot(2,2,4)
H4 = plt.contourf(time_hours2d, F['depth_middle_of_layer'],F['salin'], levels=levels
                           , cmap = cmap_sss
                           , norm = norm_sss
                           , extend='both')

cbar= plt.colorbar(H4)
cbar.set_ticks(np.arange(32,39, 1))

plt.text(0,5,'A')
plt.text(45,5,'B')

ax4.set_ylim([100,0])
ax4.set_xlabel('Time [h]')
ax4.set_title('Salinity')


plt.tight_layout()


plt.savefig('transect.png',dpi=100,bbox_inches='tight')
