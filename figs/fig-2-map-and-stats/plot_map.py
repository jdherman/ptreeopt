import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile as shp

def mapformat_CA(zoomout=False): # epsg 3310, 3488, or 26941
  
  if zoomout:
    m = Basemap(llcrnrlon=-125.6, llcrnrlat=27.7, urcrnrlon=-113.2,
                urcrnrlat=50.2, projection='cyl', resolution='l', area_thresh=25000.0)
    m.drawmapboundary(fill_color='steelblue', zorder=-99)

  else:
    m = Basemap(llcrnrlon=-122, llcrnrlat=38.4, urcrnrlon=-119.7,
                urcrnrlat=39.7, projection='cyl', resolution='h', area_thresh=1.0)

  m.arcgisimage(service='World_Shaded_Relief', xpixels=8000, dpi=75, verbose= True)
  # m.readshapefile('data/na_riv_30s/cropped', 'rivers', drawbounds=False)
  # m.fillcontinents(color='None', lake_color='steelblue', zorder=5)
  m.drawstates(zorder=6, color='gray')
  m.drawcountries(zorder=6, color='gray')
  m.drawcoastlines(color='gray')

  return m

m = mapformat_CA(zoomout=True)

patches = []

# # already cropped to hucs 18020128 and 18020129 (north/south american)
# sf = shp.Reader('data/hucs/americanriver.shp')
# for shape in sf.shapeRecords():
#   patches.append(Polygon(np.array(shape.shape.points), True))
# plt.gca().add_collection(PatchCollection(patches, facecolor= 'k', edgecolor='None', alpha=0.2, zorder=2))

# for info,shape in zip(m.rivers_info, m.rivers):
#   size = info['UP_CELLS']
#   if size > 100:
#     x,y = zip(*shape)
#     m.plot(x,y, marker=None, color='steelblue', linewidth=np.log10(size)-2, zorder=5)

# if zoomed out ...
patches.append(Polygon(np.array([(-122.0,38.4),(-122.0,39.7),(-119.7,39.7),(-119.7,38.4)]), True))
plt.gca().add_collection(PatchCollection(patches, facecolor='silver', edgecolor='0.2', alpha=0.7, zorder=2, linewidth=1.0))

plt.savefig('zoomout.svg')
plt.savefig('zoomout.png')
# # plt.show()
