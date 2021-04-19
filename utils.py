import os
from pathlib import Path
import re
from datetime import datetime

# import metpy  # noqa: F401
import numpy as np
import rioxarray
import pandas as pd

# import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import boto3
from botocore import UNSIGNED
from botocore.config import Config

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
PLATFORM = "goes16"
BUCKET = f"noaa-{PLATFORM}"


def download_nearest(
    dt, n=1, product=PRODUCT_L1_CONUS, channels=[1], outdir="data", **kwargs
):
    """Download the nearest `n` closest products to the given datetime `dt`"""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_results = search_s3(dt=dt, s3=s3, product=product, channels=channels)

    closest_n = []
    for channel in channels:
        # Find the nearest separetely for each channel
        time_diffs, keys = get_timediffs(
            dt, filter_results_by_channel(s3_results, [channel])
        )

        cur_closest = sorted(zip(time_diffs, keys), key=lambda x: abs(x[0]))[:n]
        closest_n.extend(cur_closest)

    # Now re-sort so that the products are in ascending time order
    closest_n = sorted(closest_n)

    file_paths = []
    for tdiff, key in closest_n:
        print(
            f"Downloading {key}, occurs {abs(tdiff)} seconds {'before' if tdiff < 0 else 'after'} {dt}"
        )
        file_path = _download_one_key(key, s3, outdir=outdir, verbose=True)
        file_paths.append(file_path)
    return file_paths, closest_n


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
    channels=None,
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
    if channels:
        results = filter_results_by_channel(results, channels)
        print(f"Found {len(results)} results for channels {channels} ")
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
    time_diffs = [(s - dt).total_seconds() for s in start_times]
    return time_diffs, keys


def filter_results_by_dt(s3_results, dt_start, dt_end):
    start_times = get_start_times(s3_results)
    return [res for res, dt in zip(s3_results, start_times) if dt_start < dt < dt_end]


def filter_results_by_channel(s3_results, channels):
    channel_strs = [f"C{channel:02d}" for channel in channels]
    s3_results = [r for r in s3_results if any(c in r["Key"] for c in channel_strs)]
    return s3_results


def _download_one_key(key, s3, outdir=".", overwrite=False, verbose=True):
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
        s3.download_file(BUCKET, Key=key, Filename=file_path)
    return file_path


def download_range(
    dt_start, dt_end, product=PRODUCT_L1_CONUS, channels=[1], outdir="data", **kwargs
):
    import pandas as pd

    hourly_dt = pd.date_range(dt_start, dt_end, freq="H")
    print(f"Searching from {dt_start} to {dt_end}")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3_results = []
    for dt in hourly_dt:
        s3_results.extend(search_s3(dt=dt, s3=s3, product=product, channels=channels))

    s3_results = filter_results_by_dt(s3_results, dt_start, dt_end)
    # TODO: need to generalize these filters?
    s3_results = [
        r for r in s3_results if parse_goes_filename(r["Key"])["product"] == "RadM2"
    ]
    print(f"Downloading {len(s3_results)} files")
    for r in s3_results:
        _download_one_key(r["Key"], s3, outdir=outdir, verbose=True)


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


def parse_goes_filename(fname):
    """Parses attributes of file available in filename

    The main categories are separated by underscores:
    ops environment, Data Set Name (DSN), platform, start, end, created


    The DSN has multiple sub-fields, e.g.: DSN for the
    Lighting Detection product is “GLM-L2-LCFA”

    e.g.
    OR_ABI-L2-MCMIPF-M6_G16_s20201800021177_e20201800023556_c20201800024106.nc
    OR: Operational System Real-Time Data
    ABI-L2: Advanced Baseline Imager Level 2+ (other option is level 1, L1a, L1b)
    CMIPF: product. Cloud and Moisture Image Product – Full Disk
    M3/M4/M6: ABI Mode 3, 4 or 6, (Mode 6=10 minute flex)
    C09: Channel Number (Band 9 in this example)
    G16: GOES-16
    sYYYYJJJHHMMSSs: Observation Start
    eYYYYJJJHHMMSSs: Observation End
    cYYYYJJJHHMMSSs: File Creation

    L1 example:
    OR_ABI-L1b-RadC-M3C01_G16_s20190621802131_e20190621804504_c20190621804546.nc
    Lightning mapper:
    OR_GLM-L2-LCFA_G16_s20190720000000_e20190720000200_c20190720000226.nc

    Reference:
    https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf
    Appendix A
    """
    fname_noslash = fname.split("/")[-1]

    fname_pattern = re.compile(
        r"OR_"
        r"(?P<DSN>.*)_"
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

    if "ABI" in match_dict["DSN"]:
        dsn_pattern = re.compile(
            r"(?P<instrument>ABI|GLM)-"  # fix this
            r"(?P<level>L[12b]+)-"
            r"(?P<product>\w+)-"
            r"(?P<mode>M[3-6AM])"
            r"(?P<channel>C\d+)?"  # channel is optional
        )
        m2 = re.match(dsn_pattern, match_dict["DSN"])
        match_dict.update(m2.groupdict())
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
    outname="",
    bounds=(-105, 30, -101, 33),
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


def plot_series(file_paths, bounds, dset="CMI", title_format="time"):
    metas = [parse_goes_filename(f) for f in file_paths]
    image_list = [warp_subset(f, bounds=bounds, dset=dset) for f in file_paths]

    nfiles = len(image_list)
    if nfiles > 3:
        ntiles = int(np.ceil(np.sqrt(nfiles)))
        layout = (ntiles, ntiles)
    else:
        layout = (1, nfiles)

    fig, axes = plt.subplots(*layout, sharex=True, sharey=True)
    for ax, cm, m in zip(axes.ravel(), image_list, metas):
        axim = ax.imshow(cm, cmap="RdBu_r")
        fig.colorbar(axim, ax=ax)
        if title_format == "time":
            title = m["start_time"].strftime("%H:%M")
        elif title_format == "channel":
            title = m["channel"]
        else:
            raise ValueError("Unknown title_format")
        ax.set_title(title)

    fig.suptitle(m["start_time"].strftime("%Y-%m-%d"))
    fig.tight_layout()


def create_rgb(
    dt,
    bounds,
    resolution=(1 / 600, 1 / 600),
    n=1,
    outdir="data",
    gamma=0.7,
):
    file_paths, _ = download_nearest(
        dt,
        n=1,
        product=PRODUCT_L1_CONUS,
        channels=[1, 2, 3],
    )
    R = warp_subset(file_paths[0], bounds=bounds, resolution=resolution)
    G = warp_subset(file_paths[2], bounds=bounds, resolution=resolution)
    B = warp_subset(file_paths[1], bounds=bounds, resolution=resolution)
    return combine_rgb(R, G, B, gamma=gamma)


def combine_rgb(R, G, B, gamma=0.7):
    """Combine the Red, "Veggie" (G), and Blue bands to make true-color image

    These are bands 1, 3, 2 of the ABI imager for R, G, B

    Reference:
    True color guide:
    http://cimss.ssec.wisc.edu/goes/OCLOFactSheetPDFs/ABIQuickGuide_CIMSSRGB_v2.pdf
    tutorial:
    https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html
    """
    from skimage.exposure import rescale_intensity, adjust_gamma
    from skimage.util import img_as_float

    Rg = adjust_gamma(rescale_intensity(img_as_float(R)), gamma)
    Gg = adjust_gamma(rescale_intensity(img_as_float(G)), gamma)
    Bg = adjust_gamma(rescale_intensity(img_as_float(B)), gamma)
    G_true = 0.45 * Rg + 0.1 * Gg + 0.45 * Bg

    return np.dstack([Rg, G_true, Bg])


def convert_utc_to_local(dt, to_zone="America/Chicago"):
    from dateutil import tz

    from_zone = tz.gettz("UTC")
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


# # NOTE: for now just use the above
# def subset(
#     fname,
#     dset="Rad",
#     proj="lonlat",
#     bounds=(-105, 30, -101, 33),
#     resolution=(0.001666666667, 0.001666666667),
#     resampling=1,
# ):
#     # TODO: why the eff is this 100x slower than the gdal.Warp version?
#     # TODO: is warped vrt any better?
#     left, bot, right, top = bounds

#     if proj == "lonlat":
#         # proj_str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
#         proj_str = "EPSG:4326"
#     elif proj == "utm":
#         proj_str = "+proj=utm +datum=WGS84 +zone=13"
#     else:
#         proj_str = proj
#     with rioxarray.open_rasterio(fname) as src:
#         xds_lonlat = src.rio.reproject(
#             proj_str,
#             resolution=resolution,
#             resampling=resampling,
#             # num_threads=20, # option seems to have disappeared
#         )
#         subset_ds = xds_lonlat[dset][0].sel(x=slice(left, right), y=slice(top, bot))
#         # subset_ds.plot.imshow()
#         return subset_ds
