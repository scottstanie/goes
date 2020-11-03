# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:39:45 2019

@author: Guy Serbin
"""

import os, sys, glob, argparse
from osgeo import gdal, osr

# from scipy.misc import imresize  # deprecated
from skimage.transform import resize as resize

parser = argparse.ArgumentParser(
    description="Script to create CONUS true color image from GOES 16 L1b data."
)
parser.add_argument(
    "-i", "--indir", type=str, default=".", help="Input directory name."
)
parser.add_argument(
    "-o", "--outdir", type=str, default=".", help="Output directory name."
)
parser.add_argument(
    "-p",
    "--proj",
    type=int,
    default=3857,
    help="Output projection, must be EPSG number.",
)
args = parser.parse_args()

if not args.indir:
    print("ERROR: --indir not set. exiting.")
    sys.exit()
elif not os.path.isdir(args.indir):
    print("ERROR: --indir not set to a valid directory path. exiting.")
    sys.exit()

if not args.outdir:
    print("WARNING: --outdir not set. Output will be written to --indir.")
    args.outdir = args.indir

o_srs = osr.SpatialReference()
o_srs.ImportFromEPSG(args.proj)


# based upon code ripped from https://riptutorial.com/gdal/example/25859/read-a-netcdf-file---nc--with-python-gdal

# Path of netCDF file
netcdf_red = glob.glob(os.path.join(args.indir, "OR_ABI-L1b-RadC-M3C02_G16_s*.nc"))[0]
netcdf_green = glob.glob(os.path.join(args.indir, "OR_ABI-L1b-RadC-M3C03_G16_s*.nc"))[0]
netcdf_blue = glob.glob(os.path.join(args.indir, "OR_ABI-L1b-RadC-M3C01_G16_s*.nc"))[0]
baselist = os.path.basename(netcdf_blue).split("_")

outputfilename = os.path.join(
    args.outdir, "OR_ABI-L1b-RadC-M3TrueColor_1_G16_{}.tif".format(baselist[3])
)
print("Output file will be: {}".format(outputfilename))
tempfile = os.path.join(args.outdir, "temp.tif")

# Specify the layer name to read
layer_name = "Rad"

# Open netcdf file.nc with gdal
print("Opening red band file: {}".format(netcdf_red))
dsR = gdal.Open("NETCDF:{0}:{1}".format(netcdf_red, layer_name))
print("Opening green band file: {}".format(netcdf_green))
dsG = gdal.Open("NETCDF:{0}:{1}".format(netcdf_green, layer_name))
print("Opening blue band file: {}".format(netcdf_blue))
dsB = gdal.Open("NETCDF:{0}:{1}".format(netcdf_blue, layer_name))
red_srs = osr.SpatialReference()
red_srs.ImportFromWkt(dsR.GetProjectionRef())
i_srs = osr.SpatialReference()
i_srs.ImportFromWkt(dsG.GetProjectionRef())
GeoT = dsG.GetGeoTransform()
print(i_srs.ExportToWkt())
red_transform = osr.CoordinateTransformation(red_srs, o_srs)
transform = osr.CoordinateTransformation(i_srs, o_srs)

# Read full data from netcdf

print("Reading green band into memory.")
green = dsG.ReadAsArray(0, 0, dsG.RasterXSize, dsG.RasterYSize)
print("Reading blue band into memory.")
blue = dsB.ReadAsArray(0, 0, dsB.RasterXSize, dsB.RasterYSize)
print("Reading red band into memory.")
red = dsR.ReadAsArray(0, 0, dsR.RasterXSize, dsR.RasterYSize)
print("Resizing red band to match green and blue bands.")
red = resize(red, blue.shape)
red[red < 0] = 0
green[green < 0] = 0
blue[blue < 0] = 0

# Stack data and output
print("Stacking data.")
driver = gdal.GetDriverByName("GTiff")
stack = driver.Create(
    "/vsimem/stack.tif", dsB.RasterXSize, dsB.RasterYSize, 3, gdal.GDT_Int16
)
stack.SetProjection(i_srs.ExportToWkt())
stack.SetGeoTransform(GeoT)
stack.GetRasterBand(1).WriteArray(red)
stack.GetRasterBand(2).WriteArray(green)
stack.GetRasterBand(3).WriteArray(blue)
print("Warping data to new projection.")
warped = gdal.Warp("/vsimem/warped.tif", stack, dstSRS=o_srs, outputType=gdal.GDT_Int16)

print("Writing output to disk.")

outRaster = gdal.Translate(outputfilename, "/vsimem/warped.tif")

outRaster = None
red = None
green = None
blue = None
tmp_ds = None
dsR = None
dsG = None
dsB = None

print("Processing complete.")
