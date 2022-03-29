# GOES utils
Download and crop GOES-16/17 images from AWS

# Example

```python
from apertools import parsers
import matplotlib.pyplot as plt

# User the parser class to get frame info 
s1 = parsers.Sentinel("S1A_IW_SLC__1SDV_20150316T005052_20150316T005122_005050_006570_AF56.SAFE")
# Get the start_time of the scene, download the nearest GOES scene to the Sentinel time
file_paths, _ = utils.download_nearest(s1.start_time)

# use lat/lon bounds (left, bot, right, top)
bounds = s1.get_swath_bbox()
# Subset based the frame bounds, warp from geostationary projection to latlon
arr = utils.warp_subset(fname=file_paths[0], bounds=bounds)

plt.imshow(arr)
```
