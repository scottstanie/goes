import os
import re
from datetime import datetime

# import metpy  # noqa: F401
import numpy as np
import rioxarray
import xarray as xr

# import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import boto3
from botocore import UNSIGNED
from botocore.config import Config

"""
https://docs.opendata.aws/noaa-goes16/cics-readme.html
Possible layers to use:
    ABI-L1b-RadC - Advanced Baseline Imager Level 1b CONUS

    ABI-L2-CMIPC - Advanced Baseline Imager Level 2 Cloud and Moisture Imagery CONUS
    ABI-L2-CTPC - Advanced Baseline Imager Level 2 Cloud Top Pressure CONUS
    ABI-L2-LVMPC - Advanced Baseline Imager Level 2 Legacy Vertical Moisture Profile CONUS
    ABI-L2-LVTPC - Advanced Baseline Imager Level 2 Legacy Vertical Temperature Profile CONUS
    ABI-L2-ACHAC - Advanced Baseline Imager Level 2 Cloud Top Height CONUS
    ABI-L2-ACHTF - Advanced Baseline Imager Level 2 Cloud Top Temperature Full Disk
    ABI-L2-CODC - Advanced Baseline Imager Level 2 Cloud Optical Depth CONUS
    ABI-L2-ACTPC - Advanced Baseline Imager Level 2 Cloud Top Phase CONUS
    ABI-L2-MCMIPC - Advanced Baseline Imager Level 2 Cloud and Moisture Imagery CONUS
    ABI-L2-TPWC - Advanced Baseline Imager Level 2 Total Precipitable Water CONUS

    ABI-L2-CPSC - Advanced Baseline Imager Level 2 Cloud Particle Size CONUS
    ABI-L2-LSTC - Advanced Baseline Imager Level 2 Land Surface Temperature CONUS

    ABI-L1b-RadF - Advanced Baseline Imager Level 1b Full Disk
"""

# $ aws s3 ls s3://noaa-goes16/ABI-L1b-RadC/2019/204/10/  --no-sign-request |head
# 2019-07-23 05:04:27  OR_ABI-L1b-RadC-M6C01_G16_s20192041001395_e20192041004167_c20192041004217.nc
# 2019-07-23 05:09:32  OR_ABI-L1b-RadC-M6C01_G16_s20192041006395_e20192041009167_c20192041009214.nc


PRODUCT_L1_MESO = "ABI-L1b-RadM"  # MESOSCALE
PRODUCT_CLOUD = "ABI-L2-MCMIPC"
PRODUCT_L1_CONUS = "ABI-L1b-RadC"  # CONUS
PLATFORM = "goes16"
BUCKET = f"noaa-{PLATFORM}"


def form_s3_search(dt, product=PRODUCT_L1_CONUS, platform=PLATFORM, use_hour=True):

    if use_hour:
        year_doy_hour = dt.strftime("%Y/%j/%H")
    else:
        year_doy_hour = dt.strftime("%Y/%j")
    # start_search = dt.strftime("%Y%j%H%m")
    # start_search = dt.strftime("%Y%j%H")

    # example:
    # //noaa-goes16/ABI-L1b-RadC/2000/001/12/OR_ABI-L1b-RadC-M3C01_G16_s20000011200000_e20000011200000_c20170671748180.nc
    # mode = "*"
    # M3: is mode 3 (scan operation),
    # M4 is mode 4 (only full disk scans every five minutes – no mesoscale or CONUS)
    # M6 is "...a new 10-minute flex mode": https://www.goes-r.gov/users/abiScanModeInfo.html
    # mode = "M3"
    # channel = "*"
    # channel = "C01" # is channel or band 01, There will be sixteen bands, 01-16
    # prefix = f"{product}/{year_doy_hour}/OR_{product}-{mode}{channel}_G{platform[-2:]}_s{start_search}"
    prefix = f"{product}/{year_doy_hour}/"
    return prefix


def search_s3(
    search_prefix=None,
    dt=None,
    s3=None,
    product=PRODUCT_L1_CONUS,
    platform=PLATFORM,
    channel=None,
):
    if search_prefix is None:
        search_prefix = form_s3_search(dt, product=product, platform=platform)
    # Connect to s3 via boto
    print("Searching:", search_prefix)

    if s3 is None:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # bucket = f"noaa-{platform}"
    paginator = s3.get_paginator("list_objects")
    page_iterator = paginator.paginate(
        Bucket=BUCKET,
        Prefix=search_prefix,
    )
    results = [obj for page in page_iterator for obj in page.get("Contents", [])]
    print(f"Found {len(results)} results")
    if channel:
        channel_str = f"C{channel:02d}"
        results = [r for r in results if channel_str in r["Key"]]
        print(f"Found {len(results)} results for channel {channel_str} ")
    return results


"""Results:
[{'Key': 'ABI-L2-MCMIPC/2020/180/00/OR_ABI-L2-MCMIPC-M6_G16_s20201800001177_e20201800003550_c20201800004102.nc',
  'LastModified': datetime.datetime(2020, 6, 28, 0, 26, 47, tzinfo=tzutc()),
  'ETag': '"2bc5d3470875d03b490c2b6e3a491be9-8"',
  'Size': 62795922,
  'StorageClass': 'INTELLIGENT_TIERING',
  'Owner': {'DisplayName': 'sandbox',
   'ID': '07cd0b2bd0f30623096b9275946be8ed8f210ec3ec83f15b416f8296c4e7e947'}},
   ..."""


def get_start_times(s3_results):
    keys = [r["Key"] for r in s3_results]
    start_times = [parse_goes_filename(f.split("/")[-1])["start_time"] for f in keys]
    return start_times


def get_timediffs(dt, s3_results):
    keys = [r["Key"] for r in s3_results]
    start_times = get_start_times(s3_results)
    time_diffs = [(dt - s).total_seconds() for s in start_times]
    return time_diffs, keys


def filter_results_by_dt(s3_results, dt_start, dt_end):
    start_times = get_start_times(s3_results)
    return [res for res, dt in zip(s3_results, start_times) if dt_start < dt < dt_end]


def download_nearest(dt, product=PRODUCT_L1_CONUS, channel=1, **kwargs):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_results = search_s3(dt=dt, s3=s3, product=product, channel=channel)
    time_diffs, keys = get_timediffs(dt, s3_results)
    min_dt, min_key = min(zip(time_diffs, keys), key=lambda x: abs(x[0]))
    fname = min_key.split("/")[-1]
    _download_one_key(min_key, s3)
    return fname, min_key, min_dt


def _download_one_key(key, s3, overwrite=False, verbose=True):
    fname = key.split("/")[-1]
    if os.path.exists(fname) and not overwrite:
        print(f"Skipping {fname}, already exists")
        return
    else:
        if verbose:
            print(f"Downloading {fname}")
        s3.download_file(
            BUCKET,
            Key=key,
            Filename=fname,
        )


def download_range(dt_start, dt_end, product=PRODUCT_L1_CONUS, channel=1, **kwargs):
    import pandas as pd

    hourly_dt = pd.date_range(dt_start, dt_end, freq="H")
    print(f"Searching from {dt_start} to {dt_end}")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_results = []
    for dt in hourly_dt:
        s3_results.extend(search_s3(dt=dt, s3=s3, product=product, channel=channel))

    s3_results = filter_results_by_dt(s3_results, dt_start, dt_end)
    # TODO: need to generalize these filters?
    s3_results = [
        r for r in s3_results if parse_goes_filename(r["Key"])["Product"] == "RadM2"
    ]
    print(f"Downloading {len(s3_results)} files")
    for r in s3_results:
        _download_one_key(r["Key"], s3, verbose=True)


"""TODO
warp_subset( 'NETCDF:OR_ABI-L1b-RadC-M6C02_G16_s20200600056143_e20200600058516_c20200600058546.nc:Rad', 'test2020229_rad2.tif', bounds=(-104.33, 31.6, -103.7, 31.9))
    """


"""
In [23]: utils.search_s3(prefix)
Out[23]:
[{'Key': 'ABI-L2-MCMIPC/2019/204/00/OR_ABI-L2-MCMIPC-M6_G16_s20192040001394_e20192040004167_c20192040004279.nc',
  'LastModified': datetime.datetime(2019, 7, 23, 0, 4, 52, tzinfo=tzutc()),
  'ETag': '"ad0517035e1c7d99724999d6f27ff880-8"',
  'Size': 60042281,
  'StorageClass': 'INTELLIGENT_TIERING',
  'Owner': {'DisplayName': 'sandbox',
   'ID': '07cd0b2bd0f30623096b9275946be8ed8f210ec3ec83f15b416f8296c4e7e947'}},
 {'Key': 'ABI-L2-MCMIPC/2019/204/00/OR_ABI-L2-MCMIPC-M6_G16_s20192040006394_e20192040009167_c20192040009282.nc',
 ...
 """


def parse_goes_filename(fname):
    """Parses attributes of file available in filename

    e.g.
    OR_ABI-L1b-RadC-M3C01_G16_s20190621802131_e20190621804504_c20190621804546.nc
    OR: Operational System Real-Time Data
    ABI-L2: Advanced Baseline Imager Level 2+ (other option is level 1, L1a, L1b)
    CMIPF: product. Cloud and Moisture Image Product – Full Disk
    M3 / M4: ABI Mode 3, ABI Mode 4, Mode 6=10 minute flex
    C09: Channel Number (Band 9 in this example)
    G16: GOES-16
    sYYYYJJJHHMMSSs: Observation Start
    eYYYYJJJHHMMSSs: Observation End
    cYYYYJJJHHMMSSs: File Creation

    L1 example:
    OR_ABI-L2-MCMIPC-M6_G16_s20201800021177_e20201800023556_c20201800024106.nc

    """
    fname_noslash = fname.split("/")[-1]

    fname_pattern = re.compile(
        r"OR_ABI-"
        r"(?P<level>L[12b]+)-"
        r"(?P<product>\w+)-"
        r"(?P<mode>M[3-6AM])"
        r"(?P<channel>C\d+)?_"  # channel is optional
        r"(?P<platform>G\d+)_"
        r"s(?P<start_time>\w+)_"
        r"e(?P<end_time>\w+)_"
        r"c(?P<creation_time>\w+).nc"
    )
    time_pattern = "%Y%j%H%M%S%f"
    m = re.match(fname_pattern, fname_noslash)
    if not m:
        raise ValueError(f"{fname_noslash} does not match GOES format")

    time_pattern = "%Y%j%H%M%S%f"
    match_dict = m.groupdict()
    for k in ("start_time", "end_time", "creation_time"):
        match_dict[k] = datetime.strptime(match_dict[k], time_pattern)

    return match_dict


def bbox(ds, x="x", y="y"):
    """Get (left, bot, right, top) of xarray dataset with `x`, `y` coordinates"""
    bbox = ds[x].min(), ds[y].min(), ds[x].max(), ds[y].max()
    # will be a tuple of DataArrays. We just want the values
    return tuple(c.item() for c in bbox)


# Easy reprojection:
# with rioxarray.open_rasterio(f) as src:
# src_lonlat = src.rio.reproject("EPSG:4326")


def warp_subset(
    fname,
    outname,
    bounds=(-105, 30, -101, 33),
    resolution=(0.001666666667, 0.001666666667),
    resampling="bilinear",
    dset=None,
):
    import gdal

    # if fname.endswith('.nc') and dset is not None:
    # src_name

    dsg = gdal.Open(fname)
    srcSRS = dsg.GetProjectionRef()
    dstSRS = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    xRes, yRes = resolution
    gdal.Warp(
        outname,
        dsg,
        xRes=xRes,
        yRes=yRes,
        srcSRS=srcSRS,
        dstSRS=dstSRS,
        multithread=True,
        outputBounds=bounds,
    )
    dsg = None
    return


def subset(
    fname,
    dset="Rad",
    proj="latlon",
    bounds=(-105, 30, -101, 33),
    resolution=(0.001666666667, 0.001666666667),
    resampling=1,
):
    # TODO: is warped vrt any better?
    left, bot, right, top = bounds

    if proj == "latlon":
        proj_str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    elif proj == "utm":
        proj_str = "+proj=utm +datum=WGS84 +zone=13"
    with rioxarray.open_rasterio(fname) as src:
        xds_lonlat = src.rio.reproject(
            proj_str,
            resolution=resolution,
            resampling=resampling,
            # num_threads=20, # option seems to have disappeared
        )
        subset_ds = xds_lonlat[dset][0].sel(x=slice(left, right), y=slice(top, bot))
        # subset_ds.plot.imshow()
        return subset_ds
