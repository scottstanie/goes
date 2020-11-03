from datetime import datetime

# import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# import metpy  # noqa: F401
import numpy as np
import rioxarray
import xarray as xr
import re

import boto3
from botocore import UNSIGNED
from botocore.config import Config

"""
Possible layers to use:

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

"""

# $ aws s3 ls s3://noaa-goes16/ABI-L1b-RadC/2019/204/10/  --no-sign-request |head
# 2019-07-23 05:04:27  OR_ABI-L1b-RadC-M6C01_G16_s20192041001395_e20192041004167_c20192041004217.nc
# 2019-07-23 05:09:32  OR_ABI-L1b-RadC-M6C01_G16_s20192041006395_e20192041009167_c20192041009214.nc


def form_s3_search(dt, product="ABI-L2-MCMIPC", platform="goes16"):
    year_doy_hour = dt.strftime("%Y/%j/%H")
    # start_search = dt.strftime("%Y%j%H%m")
    start_search = dt.strftime("%Y%j%H")

    # example:
    # //noaa-goes16/ABI-L1b-RadC/2000/001/12/OR_ABI-L1b-RadC-M3C01_G16_s20000011200000_e20000011200000_c20170671748180.nc
    mode = "*"
    mode = "M3"
    channel = "*"
    # channel = "C01"
    # prefix = f"{product}/{year_doy_hour}/OR_{product}-{mode}{channel}_G{platform[-2:]}_s{start_search}"
    prefix = f"{product}/{year_doy_hour}/"
    return prefix


def search_s3(search_prefix, platform="goes16"):
    # Connect to s3 via boto
    bucket = f"noaa-{platform}"

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    paginator = s3.get_paginator("list_objects")
    page_iterator = paginator.paginate(
        Bucket=bucket,
        Prefix=search_prefix,
    )
    return [obj for page in page_iterator for obj in page["Contents"]]


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
    M3 / M4: ABI Mode 3 or ABI Mode 4
    C09: Channel Number (Band 9 in this example)
    G16: GOES-16
    sYYYYJJJHHMMSSs: Observation Start
    eYYYYJJJHHMMSSs: Observation End
    cYYYYJJJHHMMSSs: File Creation

    """

    fname_pattern = re.compile(
        r"OR_ABI-"
        r"(?P<level>L[12b]+)-"
        r"(?P<product>\w+)-"
        r"(?P<mode>M[34])"
        r"(?P<channel>C\d+)_"
        r"(?P<platform>G\d+)_"
        r"s(?P<start_time>\w+)_"
        r"e(?P<end_time>\w+)_"
        r"c(?P<creation_time>\w+).nc"
    )
    time_pattern = "%Y%j%H%M%S%f"
    m = re.match(fname_pattern, fname)
    if not m:
        raise ValueError(f"{fname} does not match GOES format")

    time_pattern = "%Y%j%H%M%S%f"
    match_dict = m.groupdict()
    for k in ("start_time", "end_time", "creation_time"):
        match_dict[k] = datetime.strptime(match_dict[k], time_pattern)

    return match_dict


def subset_and_plot(fname, dset="Rad", bounds=(-105, 30, -101, 33)):
    # TODO: is warped vrt any better?
    left, bot, right, top = bounds
    with rioxarray.open_rasterio(fname) as src:
        xds_lonlat = src.rio.reproject(
            "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        )
        subset_ds = xds_lonlat[dset][0].sel(x=slice(left, right), y=slice(top, bot))
        subset_ds.plot.imshow()
        return subset_ds
