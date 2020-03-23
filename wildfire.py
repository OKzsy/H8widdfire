#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
# @Time    : 2020/3/20 14:52
# @Author  : zhaoss
# @FileName: wildfire.py
# @Email   : zhaoshaoshuai@hnnydsj.com
Description:


Parameters


"""

import os
import sys
import glob
import time
import fnmatch
import numpy as np
from osgeo import gdal, ogr, osr, gdalconst

try:
    progress = gdal.TermProgress_nocb
except:
    progress = gdal.TermProgress


def searchfiles(dirpath, partfileinfo='*', recursive=False):
    """列出符合条件的文件（包含路径），默认不进行递归查询，当recursive为True时同时查询子文件夹"""
    # 定义结果输出列表
    filelist = []
    # 列出根目录下包含文件夹在内的所有文件目录
    pathlist = glob.glob(os.path.join(os.path.sep, dirpath, "*"))
    # 逐文件进行判断
    for mpath in pathlist:
        if os.path.isdir(mpath):
            # 默认不判断子文件夹
            if recursive:
                filelist += searchfiles(mpath, partfileinfo, recursive)
        elif fnmatch.fnmatch(os.path.basename(mpath), partfileinfo):
            filelist.append(mpath)
        # 如果mpath为子文件夹，则进行递归调用，判断子文件夹下的文件
    return filelist


def getsubsetdata(datasets, subdataset_name):
    """根据提供子数据集的名字提取子集数据"""
    for subdataset in datasets:
        # 挑选指定波段
        short_name = subdataset[0].split(':')[-1]
        if short_name == subdataset_name:
            band_data = gdal.Open(subdataset[0]).ReadAsArray()
            break
    return band_data


def transformTogeotiff(file, outdir):
    visual_band_list = ['albedo_01', 'albedo_02', 'albedo_03', 'albedo_04']
    nir_band_list = ['tbb_07', 'tbb_14']
    # 获取nc数据的文件名
    basename = os.path.splitext(os.path.basename(file))[0]
    # 创建可见光影像文件名
    visual_file = os.path.join(outdir, basename) + '.tif'
    # 创建热红外通道文件名
    nir_file = os.path.join(outdir, basename) + '.tif'
    # 打开nc文件
    ds = gdal.Open(file)
    # 获取投影参数信息
    meta = ds.GetMetadata_Dict()
    up_latitude = float(meta['NC_GLOBAL#upper_left_latitude'])
    up_longitude = float(meta['NC_GLOBAL#upper_left_longitude'])
    xsize = int(meta['NC_GLOBAL#line_number'])
    ysize = int(meta['NC_GLOBAL#pixel_number'])
    resolution = float(meta['NC_GLOBAL#grid_interval'])
    # 使用GLT方式创建投影坐标信息
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    prj = srs.ExportToWkt()
    geos = (up_longitude, resolution, 0, up_latitude, 0, -resolution)
    subdatasets = ds.GetSubDatasets()
    # 根据指定名字提取子集数据，并做相应转换
    # 提取太阳天顶角
    subset_name = "SOZ"
    soz_data = getsubsetdata(subdatasets, subset_name)
    # 定标为真实角度，并转化为弧度，同时求余弦值
    soz_data = 1.0 / np.cos(soz_data * 0.01 * np.pi / 180)
    # 创建可视化图像
    tif_driver = gdal.GetDriverByName('GTiff')
    visual_out_ds = tif_driver.Create(visual_file, xsize, ysize, len(visual_band_list), gdal.GDT_Int16)
    visual_out_ds.SetGeoTransform(geos)
    visual_out_ds.SetProjection(prj)
    band_id = 0
    for isubset in visual_band_list:
        band_id += 1
        visual_band_data = getsubsetdata(subdatasets, isubset)
        # 定标为真实albedo值，并校正为真实反射率(首先缩小10000倍为真实反照率，然后放大10000倍存储）
        visual_band_data = (visual_band_data * soz_data).astype(np.int16)
        visual_out_ds.GetRasterBand(band_id).WriteArray(visual_band_data)
        visual_band_data = None
    visual_out_ds = None
    return None


def main(indir, outdir):
    # search files
    files = searchfiles(indir, partfileinfo='*.nc')
    visual_band_list = ['albedo_01', 'albedo_02', 'albedo_03', 'albedo_04']
    nir_band_list = ['tbb_07', 'tbb_14']
    for ifile in files:
        transformTogeotiff(ifile, outdir)
    return None


if __name__ == '__main__':
    # 支持中文路径
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 支持中文属性字段
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    # 注册所有ogr驱动
    ogr.RegisterAll()
    # 注册所有gdal驱动
    gdal.AllRegister()
    start_time = time.time()
    H8_dir_path = r"E:\H8"
    out_dir_path = r"E:\H8\out"
    main(H8_dir_path, out_dir_path)
    end_time = time.time()
    print("time: %.4f secs." % (end_time - start_time))
