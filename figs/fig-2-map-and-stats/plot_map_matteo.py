import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile as shp

def mapformat_CA(zoomout=False): # epsg 3310, 3488, or 26941
  
  m = Basemap(llcrnrlon=-125.6, llcrnrlat=27.7, urcrnrlon=-113.2,
              urcrnrlat=50.2, projection='cyl', resolution='l', area_thresh=25000.0)
  m.drawmapboundary(fill_color='steelblue', zorder=-99)

  m.arcgisimage(service='World_Shaded_Relief', xpixels=8000, dpi=75, verbose= True)
  m.drawstates(zorder=6, color='gray')
  m.drawcountries(zorder=6, color='gray')
  m.drawcoastlines(color='gray')

  return m

m = mapformat_CA(zoomout=True)



# plt.savefig('map.svg')
plt.show()
