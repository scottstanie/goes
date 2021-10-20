"""Functions for downloading and warping/reprojecting GOES images
"""
import os
import datetime
from pathlib import Path

# import metpy  # noqa: F401
import pandas as pd

# import cartopy.crs as ccrs
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from parsing import parse_goes_filename

# TODO:
# 2. Go from sentinel 1 filename -> start_time -> download nearest
# 3. seearch all sentinel files, download all matching RadC

"""
https://docs.opendata.aws/noaa-goes16/cics-readme.html
Possible layers to use:
    ABI-L1b-RadC - Advanced Baseline Imager Level 1b CONUS
    ABI-L1b-RadF - Advanced Baseline Imager Level 1b Full Disk

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
ALL_PRODUCTS = pd.read_csv("product_list.csv")

# $ aws s3 ls s3://noaa-goes16/ABI-L1b-RadC/2019/204/10/  --no-sign-request |head
# 2019-07-23 05:04:27  OR_ABI-L1b-RadC-M6C01_G16_s20192041001395_e20192041004167_c20192041004217.nc
# 2019-07-23 05:09:32  OR_ABI-L1b-RadC-M6C01_G16_s20192041006395_e20192041009167_c20192041009214.nc


PRODUCT_L1_MESO = "ABI-L1b-RadM"  # MESOSCALE
PRODUCT_CLOUD = "ABI-L2-MCMIPC"
PRODUCT_L1_CONUS = "ABI-L1b-RadC"  # CONUS
PLATFORM_EAST = "goes16"
PLATFORM_WEST = "goes17"
BUCKET_EAST = f"noaa-{PLATFORM_EAST}"
BUCKET_WEST = f"noaa-{PLATFORM_WEST}"


def _bucket(platform):
    return BUCKET_EAST if platform == PLATFORM_EAST else BUCKET_WEST


def download_nearest(
    dt,
    n=1,
    product=PRODUCT_L1_CONUS,
    channels=[1],
    outdir="data",
    platform=PLATFORM_EAST,
    **kwargs,
):
    """Download the nearest `n` closest products to the given datetime `dt`"""
    dt = _check_tz(dt)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_results = search_s3(
        dt=dt, s3=s3, product=product, platform=platform, channels=channels
    )

    closest_n = []
    if channels:
        for channel in channels:
            # Find the nearest separetely for each channel
            time_diffs, keys = get_timediffs(
                dt, filter_results_by_channel(s3_results, [channel])
            )

            cur_closest = sorted(zip(time_diffs, keys), key=lambda x: abs(x[0]))[:n]
            closest_n.extend(cur_closest)
    else:
        time_diffs, keys = get_timediffs(dt, s3_results)
        cur_closest = sorted(zip(time_diffs, keys), key=lambda x: abs(x[0]))[:n]
        closest_n.extend(cur_closest)

    # Now re-sort so that the products are in ascending time order
    closest_n = sorted(closest_n)

    file_paths = []
    for tdiff, key in closest_n:
        print(
            f"Downloading {key}, {abs(tdiff)} sec {'before' if tdiff < 0 else 'after'} {dt}"
        )
        file_path = _download_one_key(
            key, s3, outdir=outdir, platform=platform, verbose=True
        )
        file_paths.append(file_path)
    return file_paths, closest_n


def _check_tz(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def download_range(
    dt_start,
    dt_end,
    product=PRODUCT_L1_CONUS,
    channels=[1],
    platform=PLATFORM_EAST,
    outdir="data",
    **kwargs,
):
    import pandas as pd

    dt_start = _check_tz(dt_start)
    dt_end = _check_tz(dt_end)

    hourly_dt = pd.date_range(dt_start, dt_end, freq="H")
    print(f"Searching from {dt_start} to {dt_end}")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_results = []
    for dt in hourly_dt:
        s3_results.extend(
            search_s3(
                dt=dt, s3=s3, product=product, channels=channels, platform=platform
            )
        )

    s3_results = filter_results_by_dt(s3_results, dt_start, dt_end)
    if product == PRODUCT_L1_MESO:
        # TODO: any way in advance to know the bounds of these mesoscale products?
        # TODO: need to generalize these filters?
        prod = "RadM1"
        from collections import Counter

        print(Counter([parse_goes_filename(r["Key"])["product"] for r in s3_results]))
        print(f"Filtering based on 'product' = {prod}")
        s3_results = [
            r for r in s3_results if parse_goes_filename(r["Key"])["product"] == prod
        ]
    print(f"Downloading {len(s3_results)} files")
    for r in s3_results:
        _download_one_key(r["Key"], s3, outdir=outdir, platform=platform, verbose=True)


# example:
# //noaa-goes16/ABI-L1b-RadC/2000/001/12/OR_ABI-L1b-RadC-\
# M3C01_G16_s20000011200000_e20000011200000_c20170671748180.nc
# mode = "*"
# M3: is mode 3 (scan operation),
# M4 is mode 4 (only full disk scans every five minutes – no mesoscale or CONUS)
# M6 is "...a new 10-minute flex mode": https://www.goes-r.gov/users/abiScanModeInfo.html
# mode = "M3"
# channel = "*"
# channel = "C01" # is channel or band 01, There will be sixteen bands, 01-16
# prefix = f"{product}/{year_doy_hour}/OR_{product}-{mode}{channel}_G{platform[-2:]}_s{start_search}"


def search_s3(
    dt=None,
    product=PRODUCT_L1_CONUS,
    platform=PLATFORM_EAST,
    channels=None,
    s3=None,
    use_hour=True,
):
    year_doy_hour = dt.strftime("%Y/%j/%H") if use_hour else dt.strftime("%Y/%j")
    search_prefix = f"{product}/{year_doy_hour}/"
    # Connect to s3 via boto
    print("Searching:", search_prefix)

    if s3 is None:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    paginator = s3.get_paginator("list_objects")
    page_iterator = paginator.paginate(
        Bucket=_bucket(platform),
        Prefix=search_prefix,
    )
    print(f"bucket, {_bucket(platform)}")
    results = [obj for page in page_iterator for obj in page.get("Contents", [])]
    print(f"Found {len(results)} results")
    if channels:
        results = filter_results_by_channel(results, channels)
        print(f"Found {len(results)} results for channels {channels} ")
    return results


"""Results:
[{'Key': 'ABI-L2-MCMIPC/2020/180/00/OR_ABI-L2-\
MCMIPC-M6_G16_s20201800001177_e20201800003550_c20201800004102.nc',
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
    time_diffs = [(s - dt).total_seconds() for s in start_times]
    return time_diffs, keys


def filter_results_by_dt(s3_results, dt_start, dt_end):
    start_times = get_start_times(s3_results)
    return [res for res, dt in zip(s3_results, start_times) if dt_start <= dt <= dt_end]


def filter_results_by_channel(s3_results, channels):
    channel_strs = [f"C{channel:02d}" for channel in channels]
    s3_results = [r for r in s3_results if any(c in r["Key"] for c in channel_strs)]
    return s3_results


def _download_one_key(
    key, s3, outdir=".", platform=PLATFORM_EAST, overwrite=False, verbose=True
):
    if not os.path.exists(outdir):
        print(f"Making directory {outdir}")
        os.mkdir(outdir)

    fname = key.split("/")[-1]
    file_path = str((Path(outdir) / Path(fname)).absolute())
    if os.path.exists(file_path) and not overwrite:
        print(f"Skipping {file_path}, already exists")
        return file_path
    else:
        if verbose:
            print(f"Downloading {fname}")
        s3.download_file(_bucket(platform), Key=key, Filename=file_path)
    return file_path


"""
In [23]: utils.search_s3(prefix)
Out[23]:
[{'Key': 'ABI-L2-MCMIPC/2019/204/00/OR_ABI-L2-M..._c20192040004279.nc',
  'LastModified': datetime.datetime(2019, 7, 23, 0, 4, 52, tzinfo=tzutc()),
  'ETag': '"ad0517035e1c7d99724999d6f27ff880-8"',
  'Size': 60042281,
  'StorageClass': 'INTELLIGENT_TIERING',
  'Owner': {'DisplayName': 'sandbox',
   'ID': '07cd0b2bd0f30623096b9275946be8ed8f210ec3ec83f15b416f8296c4e7e947'}},
 {'Key': 'ABI-L2-MCMIPC/2019/204/00/OR_ABI-L2-..._c20192040009282.nc',
 ...
 """


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
    outname="",
    bounds=None,
    # bounds=(-105, 30, -101, 33),
    resolution=(1 / 600, 1 / 600),
    resampling="bilinear",
    dset="Rad",
):
    import gdal

    if fname.endswith(".nc") and dset is not None:
        src_name = f"NETCDF:{fname}:{dset}"
    else:
        src_name = fname
    dsg = gdal.Open(src_name)
    srcSRS = dsg.GetProjectionRef()
    dstSRS = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    xRes, yRes = resolution
    # print(f"Warping to {bounds}")

    # If no output specified, use an in-memory VRT. Otherwise, dont override extension
    out_format = "VRT" if not outname else None

    out_ds = gdal.Warp(
        outname,
        dsg,
        format=out_format,
        xRes=xRes,
        yRes=yRes,
        srcSRS=srcSRS,
        dstSRS=dstSRS,
        multithread=True,
        resampleAlg=resampling,
        outputBounds=bounds,
    )
    out_arr = out_ds.ReadAsArray()
    # Make sure to close the files
    out_ds, dsg = None, None
    return out_arr


def convert_dt(dt, to_zone="America/Chicago", from_zone="UTC"):
    from dateutil import tz

    from_zone = tz.gettz(from_zone)
    # TODO: get somehow from the image?
    to_zone = tz.gettz(to_zone)
    # Tell the datetime object that it's in UTC time zone, then convert
    return dt.replace(tzinfo=from_zone).astimezone(to_zone)


def interp_goes(dt, file1, file2):
    import xarray as xr

    t1 = parse_goes_filename(file1)["start_time"]
    t2 = parse_goes_filename(file2)["start_time"]
    alpha = (dt - t1).seconds / (t2 - t1).seconds
    # Make sure that dt is between t1 and t2
    assert 0 <= alpha <= 1
    # an alpha = 0 means the desired time is right on image1 (dt == t1)
    # alpha = 1 means dt == t2
    # interpolated will be (1 - alpha) * image1 + alpha * image2
    image1 = xr.open_dataset(file1)
    image2 = xr.open_dataset(file2)
    return (1 - alpha) * image1 + alpha * image2


# NOTE: for now just use the above
def subset(
    fname,
    dset="Rad",
    proj="lonlat",
    bounds=(-105, 30, -101, 33),
    resolution=(0.001666666667, 0.001666666667),
    resampling=1,
):
    import rioxarray

    # TODO: why the eff is this 100x slower than the gdal.Warp version?
    # TODO: is warped vrt any better?
    left, bot, right, top = bounds

    if proj == "lonlat":
        # proj_str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        proj_str = "EPSG:4326"
    elif proj == "utm":
        proj_str = "+proj=utm +datum=WGS84 +zone=13"
    else:
        proj_str = proj
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
