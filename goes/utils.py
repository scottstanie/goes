"""Functions for downloading and warping/reprojecting GOES images
"""
import os
import datetime
from pathlib import Path

# import metpy  # noqa: F401
import pandas as pd
import numpy as np

# import cartopy.crs as ccrs
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from .parsing import parse_goes_filename

# TODO:
# 2. Go from sentinel 1 filename -> start_time -> download nearest
# 3. search all sentinel files, download all matching RadC

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
ALL_PRODUCTS = pd.read_csv(Path(__file__).parent / "product_list.csv")

# $ aws s3 ls s3://noaa-goes16/ABI-L1b-RadC/2019/204/10/  --no-sign-request |head
# 2019-07-23 05:04:27  OR_ABI-L1b-RadC-M6C01_G16_s20192041001395_e20192041004167_c20192041004217.nc
# 2019-07-23 05:09:32  OR_ABI-L1b-RadC-M6C01_G16_s20192041006395_e20192041009167_c20192041009214.nc


PRODUCT_L1_MESO = "ABI-L1b-RadM"  # MESOSCALE
PRODUCT_CLOUD = "ABI-L2-MCMIPC"
PRODUCT_L1_CONUS = "ABI-L1b-RadC"  # CONUS
PRODUCT_RGB_CONUS = PRODUCT_CLOUD # alias to remind that this is what to use for RGB
PLATFORM_EAST = "goes16"
PLATFORM_WEST = "goes17"
BUCKET_EAST = f"noaa-{PLATFORM_EAST}"
BUCKET_WEST = f"noaa-{PLATFORM_WEST}"

# Common channels: https://www.goes-r.gov/mission/ABI-bands-quick-info.html
CHANNEL_B = 1
CHANNEL_R = 2
CHANNEL_G = 3
CHANNEL_IR = 13
# Note: unit of the clean IR channel is brightness temperature, NOT reflectance.
L2_DSET_BLUE = "CMI_C01"
L2_DSET_RED = "CMI_C02"
L2_DSET_GREEN = "CMI_C03"
L2_DSET_IR = "CMI_C13"

# Latlon resolution (degrees) of approximately 1km grid
RES_1KM = 0.008333


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

# https://noaa-goes16.s3.amazonaws.com/ABI-L2-MCMIPC/2019/132/00/
# https://noaa-goes16.s3.amazonaws.com/ABI-L2-MCMIPC/2019/132/00/OR_ABI-L2-MCMIPC-M6_G16_s20191320051288_e20191320054060_c20191320054168.nc

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
    # The level 2 products seem to not have Channel in filename
    if channels and "ABI-L2" not in product:
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
    resolution=(RES_1KM, RES_1KM),
    resampling="bilinear",
    dset="Rad",
):
    from osgeo import gdal

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


def create_rgb(
    dt,
    bounds,
    # resolution=(1 / 600, 1 / 600),
    resolution=(RES_1KM, RES_1KM),
    n=1,
    outdir="data",
    gamma=1 / 2.2,
    contrast=None,
    add_ir=False,
    out_shape=None,
):
    from skimage.transform import resize
    channels = [CHANNEL_B, CHANNEL_R, CHANNEL_G]
    if add_ir:
        channels.append(CHANNEL_IR)

    file_paths, _ = download_nearest(
        dt,
        n=1,
        channels=None,
        # product=PRODUCT_L1_CONUS,
        product=PRODUCT_RGB_CONUS
        ,
        outdir=outdir,
    )
    print(f"{file_paths = }")
    B = warp_subset(file_paths[0], bounds=bounds, resolution=resolution, dset=L2_DSET_BLUE)
    R = warp_subset(file_paths[0], bounds=bounds, resolution=resolution, dset=L2_DSET_RED)
    G = warp_subset(file_paths[0], bounds=bounds, resolution=resolution, dset=L2_DSET_GREEN)
    if out_shape is not None:
        R = resize(R, out_shape)
        G = resize(G, out_shape)
        B = resize(B, out_shape)

    is_night = False
    nunique = len(np.unique(R))
    if nunique < 5:
        print(f"WARNING: R has {nunique} values. {dt} may be at night")
        is_night = True

    IR = (
        warp_subset(file_paths[0], bounds=bounds, resolution=resolution, dset=L2_DSET_IR)
        if add_ir
        else None
    )
    if out_shape is not None:
        IR = resize(IR, out_shape)
    if is_night:
        # Set all 3 to IR channel
        print(IR.max(), IR.min())
        cleanIR = _prep_ir(IR)
        print(cleanIR.max(), cleanIR.min())
        return np.dstack([cleanIR, cleanIR, cleanIR])

    return combine_rgb(R, G, B, gamma=gamma, contrast=contrast, IR=IR)


def combine_rgb(R, G, B, gamma=1 / 2.2, contrast=None, IR=None, stretch_max=True):
    """Combine the Red, "Veggie" (G), and Blue bands to make true-color image

    These are bands 1, 3, 2 of the ABI imager for R, G, B

    Reference:
    True color guide:
    http://cimss.ssec.wisc.edu/goes/OCLOFactSheetPDFs/ABIQuickGuide_CIMSSRGB_v2.pdf
    tutorial:
    https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html
    """
    # from skimage.exposure import rescale_intensity, adjust_gamma
    # from skimage.util import img_as_float

    # Rg = adjust_gamma(rescale_intensity(img_as_float(R)), gamma)
    # Gg = adjust_gamma(rescale_intensity(img_as_float(G)), gamma)
    # Bg = adjust_gamma(rescale_intensity(img_as_float(B)), gamma)
    print("Max, min of R, G, B")
    print(R.max(), R.min())
    print(G.max(), G.min())
    print(B.max(), B.min())
    if stretch_max:
        m = np.max([R.max(), G.max(), B.max()]).astype(float)
    else:
        #   Int16 valid_range 0, 4095;
        m = 4095
    Rg = np.power(np.clip(R / m, 0, 1), gamma)
    Gg = np.power(np.clip(G / m, 0, 1), gamma)
    Bg = np.power(np.clip(B / m, 0, 1), gamma)

    G_true = 0.45 * Rg + 0.1 * Gg + 0.45 * Bg
    # G_true = Gg

    RGB = np.dstack([Rg, G_true, Bg])
    if contrast:
        RGB = contrast_correction(RGB, contrast=contrast)

    if IR is not None:
        return combine_ir(RGB, _prep_ir(IR))
    else:
        return RGB


def _prep_ir(IR, minval=None, maxval=None):
    # Normalize the channel between a range.
    # cleanIR = (cleanIR-minimumValue)/(maximumValue-minimumValue)
    # cleanIR = (IR - 90) / (313 - 90)
    if minval is None:
        minval = IR.min()
    if maxval is None:
        maxval = IR.max()
    cleanIR = (IR - minval) / (maxval - minval)

    # Apply range limits to make sure values are between 0 and 1
    cleanIR = np.clip(cleanIR, 0, 1)

    # Invert colors so that cold clouds are white
    cleanIR = 1 - cleanIR

    # Lessen the brightness of the coldest clouds so they don't appear so bright
    # when we overlay it on the true color image.
    cleanIR = cleanIR / 1.4
    return cleanIR


def contrast_correction(color, contrast=105):
    """
    Modify the contrast of an RGB
    See:
    https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/

    Input:
        color    - an array representing the R, G, and/or B channel
        contrast - contrast correction level

    Usage:
        RGB_contrast = contrast_correction(RGB, contrast_amount)
    """
    if not contrast:
        return color
    F = (259 * (contrast + 255)) / (255.0 * 259 - contrast)
    color = F * (color - 0.5) + 0.5
    color = np.clip(color, 0, 1)  # Force value limits 0 through 1.
    return color


def combine_ir(rgb, IR):
    """Add in clean IR to the contrast-corrected True Color image"""
    if IR is None:
        return rgb

    return np.dstack(
        [
            np.maximum(rgb[:, :, 0], IR),
            np.maximum(rgb[:, :, 1], IR),
            np.maximum(rgb[:, :, 2], IR),
        ]
    )
