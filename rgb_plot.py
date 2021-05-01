"""
Create true color RGB from GOES Level 1 radiances. Plot with cartopy

Source: https://unidata.github.io/python-training/gallery/mapping_goes16_truecolor/
"""
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import metpy  # noqa: F401
import numpy as np
import xarray


# TODO: dedupe? or delete this?
def make_rgb():
    file_ = (
        "https://ramadda.scigw.unidata.ucar.edu/repository/opendap"
        "/4ef52e10-a7da-4405-bff4-e48f68bb6ba2/entry.das#fillmismatch"
    )
    # file_ = (
    #     "OR_ABI-L1b-RadC-M3C01_G16_s20190621802131_e20190621804504_c20190621804546.nc"
    # )

    C = xarray.open_dataset(file_)
    # Scan's start time, converted to datetime object
    scan_start = datetime.strptime(C.time_coverage_start, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Scan's end time, converted to datetime object
    scan_end = datetime.strptime(C.time_coverage_end, "%Y-%m-%dT%H:%M:%S.%fZ")

    # File creation time, convert to datetime object
    file_created = datetime.strptime(C.date_created, "%Y-%m-%dT%H:%M:%S.%fZ")

    # The 't' variable is the scan's midpoint time
    midpoint = str(C["t"].data)[:-8]
    scan_mid = datetime.strptime(midpoint, "%Y-%m-%dT%H:%M:%S.%f")

    print("Scan Start    : {}".format(scan_start))
    print("Scan midpoint : {}".format(scan_mid))
    print("Scan End      : {}".format(scan_end))
    print("File Created  : {}".format(file_created))
    print("Scan Duration : {:.2f} minutes".format((scan_end - scan_start).seconds / 60))
    # Load the three channels into appropriate R, G, and B variables
    R = C["CMI_C02"].data
    G = C["CMI_C03"].data
    B = C["CMI_C01"].data
    # Apply range limits for each channel. RGB values must be between 0 and 1
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    # Apply a gamma correction to the image to correct ABI detector brightness
    gamma = 2.2
    R = np.power(R, 1 / gamma)
    G = np.power(G, 1 / gamma)
    B = np.power(B, 1 / gamma)
    # Calculate the "True" Green
    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.clip(G_true, 0, 1)  # apply limits again, just in case.
    # The RGB array with the raw veggie band
    # RGB_veggie = np.dstack([R, G, B])

    # The RGB array for the true color image
    RGB = np.dstack([R, G_true, B])
    return C, RGB


def plot_rgb(C, RGB, extent=[-105, -101, 30, 33]):
    # We'll use the `CMI_C02` variable as a 'hook' to get the CF metadata.
    dat = C.metpy.parse_cf("CMI_C02")
    scan_start = datetime.strptime(C.time_coverage_start, "%Y-%m-%dT%H:%M:%S.%fZ")

    geos = dat.metpy.cartopy_crs

    # We also need the x (north/south) and y (east/west) axis sweep of the ABI data
    x = dat.x
    y = dat.y
    fig = plt.figure(figsize=(8, 8))

    pc = ccrs.PlateCarree()

    ax = fig.add_subplot(1, 1, 1, projection=pc)
    # utah_extent = [-114.75, -108.25, 36, 43]
    ax.set_extent(extent, crs=pc)

    ax.imshow(
        RGB,
        origin="upper",
        extent=(x.min(), x.max(), y.min(), y.max()),
        transform=geos,
        interpolation="none",
    )

    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(ccrs.cartopy.feature.STATES)

    plt.title("GOES-16 True Color", loc="left", fontweight="bold", fontsize=15)
    plt.title("{}".format(scan_start.strftime("%d %B %Y %H:%M UTC ")), loc="right")
    return fig, ax


# TODO: did i just double-write this function and forget about it?
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
