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
import argparse
import fnmatch
import numpy as np
import warnings
from datetime import datetime
from osgeo import gdal, ogr, osr, gdalconst

warnings.filterwarnings('ignore')
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
            tmp_ds = gdal.Open(subdataset[0])
            band_data = tmp_ds.ReadAsArray()
            tmp_ds = None
            break
    return band_data


def transformTogeotiff(file, outdir):
    visual_band_list = ['albedo_01', 'albedo_02', 'albedo_03', 'albedo_04', 'albedo_05']
    nir_band_list = ['tbb_07', 'tbb_14', 'tbb_15']
    # 获取nc数据的文件名
    basename = os.path.splitext(os.path.basename(file))[0]
    # 创建可见光影像文件名
    visual_file = os.path.join(outdir, basename) + '_vis.tif'
    # 创建热红外通道文件名
    nir_file = os.path.join(outdir, basename) + '_nir.tif'
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
    # 创建热红外图像，用于火点检测
    nir_out_ds = tif_driver.Create(nir_file, xsize, ysize, len(nir_band_list), gdal.GDT_Float32)
    nir_out_ds.SetGeoTransform(geos)
    nir_out_ds.SetProjection(prj)
    band_id = 0
    for isubset in nir_band_list:
        band_id += 1
        nir_band_data = getsubsetdata(subdatasets, isubset)
        # 定标为真实亮温值
        nir_band_data = nir_band_data * 0.01 + 273.15
        nir_out_ds.GetRasterBand(band_id).WriteArray(nir_band_data)
        nir_band_data = None
    nir_out_ds = None
    return visual_file, nir_file


def generat_mask(vis_file_path, nir_file_path, outdir):
    """利用可见光近红外波段数据生成云，水，冰雪掩模"""
    vis_ds = gdal.Open(vis_file_path)
    xsize = vis_ds.RasterXSize
    ysize = vis_ds.RasterYSize
    geos = vis_ds.GetGeoTransform()
    prj = vis_ds.GetProjection()
    bands_data = vis_ds.ReadAsArray()
    # 计算冰雪掩模
    ndsi = (bands_data[0, :, :] - bands_data[4, :, :]) / (bands_data[0, :, :] + bands_data[4, :, :])
    mask = ndsi > 0.13
    ndsi = None
    # 计算水体掩模
    ndwi = (bands_data[3, :, :] - bands_data[2, :, :]) / (bands_data[3, :, :] + bands_data[2, :, :])
    mask = np.bitwise_or(ndwi < 0, mask)
    ndwi = None
    # 计算云掩模
    # 读取亮温数据
    BT_ds = gdal.Open(nir_file_path)
    bt_data = BT_ds.ReadAsArray()
    part1 = np.bitwise_or(bt_data[1, :, :] < 265, bt_data[2, :, :] < 265)
    part2 = np.bitwise_and(bt_data[1, :, :] < 285, (bt_data[0, :, :] - bt_data[1, :, :]) > 22)
    part3 = bands_data[2, :, :] > 5600
    cld_mask = np.bitwise_or(np.bitwise_or(part1, part2), part3)
    part1 = part2 = part3 = None
    bt_data = BT_ds = None
    mask = np.bitwise_or(cld_mask, mask).astype(np.byte) * 255
    # 写出掩模
    basename = os.path.splitext(os.path.basename(vis_file_path))[0]
    mask_path = os.path.join(outdir, basename) + "_cld.tif"
    tif_driver = gdal.GetDriverByName('GTiff')
    out_ds = tif_driver.Create(mask_path, xsize, ysize, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geos)
    out_ds.SetProjection(prj)
    out_ds.GetRasterBand(1).WriteArray(mask)
    out_ds = None
    return 1


def shp2raster(raster_ds, shp_layer, ext):
    # 将行列整数浮点化
    ext = np.array(ext) * 1.0
    # 获取栅格数据的基本信息
    raster_prj = raster_ds.GetProjection()
    raster_geo = raster_ds.GetGeoTransform()
    # 根据最小重叠矩形的范围进行矢量栅格化
    ulx, uly = gdal.ApplyGeoTransform(raster_geo, ext[0], ext[1])
    x_size = ext[2] - ext[0]
    y_size = ext[3] - ext[1]
    # 创建mask
    mask_ds = gdal.GetDriverByName('MEM').Create('', int(x_size), int(y_size), 1, gdal.GDT_Byte)
    mask_ds.SetProjection(raster_prj)
    mask_geo = [ulx, raster_geo[1], 0, uly, 0, raster_geo[5]]
    mask_ds.SetGeoTransform(mask_geo)
    # 矢量栅格化
    gdal.RasterizeLayer(mask_ds, [1], shp_layer, burn_values=[1])

    return mask_ds


def min_rect(raster_ds, shp_layer):
    # 获取栅格的大小
    x_size = raster_ds.RasterXSize
    y_size = raster_ds.RasterYSize
    # 获取是矢量的范围
    extent = shp_layer.GetExtent()
    # 获取栅格的放射变换参数
    raster_geo = raster_ds.GetGeoTransform()
    # 计算逆放射变换系数
    raster_inv_geo = gdal.InvGeoTransform(raster_geo)
    # 计算在raster上的行列号
    # 左上
    off_ulx, off_uly = map(round, gdal.ApplyGeoTransform(raster_inv_geo, extent[0], extent[3]))
    # 右下
    off_drx, off_dry = map(round, gdal.ApplyGeoTransform(raster_inv_geo, extent[1], extent[2]))
    # 判断是否有重叠区域
    if off_ulx >= x_size or off_uly >= y_size or off_drx <= 0 or off_dry <= 0:
        sys.exit("Have no overlap")
    # 限定重叠范围在栅格影像上
    # 列
    offset_column = np.array([off_ulx, off_drx])
    offset_column = np.maximum((np.minimum(offset_column, x_size - 1)), 0)
    # 行
    offset_line = np.array([off_uly, off_dry])
    offset_line = np.maximum((np.minimum(offset_line, y_size - 1)), 0)

    return [offset_column[0], offset_line[0], offset_column[1], offset_line[1]]


def mask_raster(raster_ds, mask_ds, ext):
    # 将行列整数浮点化
    ext = np.array(ext) * 1.0
    # 获取栅格数据的基本信息
    raster_prj = raster_ds.GetProjection()
    raster_geo = raster_ds.GetGeoTransform()
    bandCount = raster_ds.RasterCount
    dataType = raster_ds.GetRasterBand(1).DataType
    # 根据最小重叠矩形的范围进行矢量栅格化
    ulx, uly = gdal.ApplyGeoTransform(raster_geo, ext[0], ext[1])
    x_size = ext[2] - ext[0]
    y_size = ext[3] - ext[1]
    # 创建输出影像
    result_ds = gdal.GetDriverByName('MEM').Create('', int(x_size), int(y_size), bandCount, dataType)
    result_ds.SetProjection(raster_prj)
    result_geo = [ulx, raster_geo[1], 0, uly, 0, raster_geo[5]]
    result_ds.SetGeoTransform(result_geo)
    # 获取掩模
    mask = mask_ds.GetRasterBand(1).ReadAsArray()
    # 对原始影像进行掩模并输出
    for band in range(bandCount):
        banddata = raster_ds.GetRasterBand(band + 1).ReadAsArray(int(ext[0]), int(ext[1]), int(x_size), int(y_size))
        banddata = np.choose(mask, (0, banddata))
        result_ds.GetRasterBand(band + 1).WriteArray(banddata)
    return result_ds


def clip(raster, shp, out):
    # 打开栅格和矢量影像
    raster_ds = gdal.Open(raster)
    shp_ds = ogr.Open(shp)
    shp_l = shp_ds.GetLayer()
    # 计算矢量和栅格的最小重叠矩形
    offset = min_rect(raster_ds, shp_l)
    # 矢量栅格化
    mask_ds = shp2raster(raster_ds, shp_l, offset)
    # 进行裁剪
    res = mask_raster(raster_ds, mask_ds, offset)
    # 删除原来影像，写入裁剪后的影像
    raster_ds = None
    shp_ds = None
    os.remove(raster)
    gdal.GetDriverByName("GTiff").CreateCopy(out, res, strict=1)
    return None


def action(indir, outdir, shp_file):
    # 搜索文件
    files = searchfiles(indir, partfileinfo='*.nc')
    for ifile in files:
        visual_file, nir_file = transformTogeotiff(ifile, outdir)
        # 预留裁剪模块，只处理感兴趣区域
        if shp_file is not None:
            clip(visual_file, shp_file, visual_file)
            clip(nir_file, shp_file, nir_file)
        # 生成云，水，冰雪掩模
        cld = generat_mask(visual_file, nir_file, outdir)
        # 火点检测
        pass
    return None


# def main(argv):
def main():
    # parser = argparse.ArgumentParser(prog=argv[0])
    # parser.add_argument('-src', '--srcdir', dest='srcdir', required=True)
    # parser.add_argument('-dst', '--dstdir', dest='dstdir', required=True)
    # parser.add_argument('-v', '--vector', dest='vector', default=None)
    # args = parser.parse_args(argv[1:])
    # if not os.path.exists(args.dstdir):
    #     os.makedirs(args.dstdir)
    # 支持中文路径
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 支持中文属性字段
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    # 注册所有ogr驱动
    ogr.RegisterAll()
    # 注册所有gdal驱动
    gdal.AllRegister()
    start_time = time.time()
    # H8_dir_path = args.srcdir
    # out_dir_path = args.dstdir
    # shp = args.vector
    H8_dir_path = r"F:\kuihua8"
    out_dir_path = r"F:\kuihua8\out"
    shp = r"F:\kuihua8\guojie\bou1_4p.shp"
    action(H8_dir_path, out_dir_path, shp_file=shp)
    end_time = time.time()
    print("time: %.4f secs." % (end_time - start_time))


if __name__ == '__main__':
    try:
        # sys.exit(main(sys.argv))
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(-1)
