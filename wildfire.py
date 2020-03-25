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
import math
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
            tmp_ds = gdal.Open(subdataset[0])
            band_data = tmp_ds.ReadAsArray()
            tmp_ds = None
            break
    return band_data


def transformTogeotiff(file, outdir):
    visual_band_list = ['albedo_01', 'albedo_02', 'albedo_03', 'albedo_04', 'albedo_05']
    nir_band_list = ['tbb_07', 'tbb_14']
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


def Extend(xs, ys, matrix):
    """
    根据滤波模板的大小，对原始影像矩阵进行外扩。
    :param xs: 滤波模板的xsize，要求为奇数
    :param ys: 滤波模板的ysize，要求为奇数
    :param matrix: 原始影像矩阵
    :return: 依据模板大小扩展后的矩阵
    """
    xs_fill = int((xs - 1) / 2)
    ys_fill = int((ys - 1) / 2)
    # 使用镜像填充
    extended_val = np.pad(matrix, ((ys_fill, ys_fill), (xs_fill, xs_fill)), "reflect")
    matrix = None
    return extended_val


def img_filtering(xs, ys, ori_xsize, ori_ysize, kernel, ext_img):
    """

    :param xs: 卷积核大小：列
    :param ys: 卷积核大小：行
    :param kernel: 卷积核
    :param ext_img: 经扩展后的影像
    :return: 滤波后的影像
    """
    # 使用切片后影像的波段书
    # 创建切片后存储矩阵
    # 使用滑动窗口
    filtered_img = np.zeros((math.ceil(ori_ysize / ys), math.ceil(ori_xsize / xs)), dtype=np.float16)
    for irow in range(ys):
        for icol in range(xs):
            filtered_img += ext_img[irow: irow + ori_ysize: ys, icol: icol + ori_xsize: xs] * kernel[irow, icol]
    return filtered_img


def generat_mask(vis_file_path):
    """利用可见光近红外波段数据生成云，水，冰雪掩模"""
    vis_ds = gdal.Open(vis_file_path)
    xsize = vis_ds.RasterXSize
    ysize = vis_ds.RasterYSize
    geos = vis_ds.GetGeoTransform()
    prj = vis_ds.GetProjection()
    bands_data = vis_ds.ReadAsArray()
    # 创建掩模矩阵
    mask = np.ones((ysize, xsize), dtype=np.byte)
    # # 计算冰雪掩模
    # ndsi = (bands_data[0, :, :] - bands_data[4, :, :]) / (bands_data[0, :, :] + bands_data[4, :, :])
    # ndsi_index = np.where(ndsi > 0.13)
    # mask[ndsi_index] = 0
    # ndsi = None
    # ndsi_index = None
    # # 计算水体掩模
    # ndwi = (bands_data[3, :, :] - bands_data[2, :, :]) / (bands_data[3, :, :] + bands_data[2, :, :])
    # ndwi_index = np.where(ndwi < 0)
    # mask[ndwi_index] = 0
    # ndwi = None
    # ndwi_index = None
    # 计算云掩模
    # 获取0.51微米波段反射率
    ref_green = bands_data[1, :, :] * 0.0001
    # ref_green = np.arange(9).reshape(3, 3) + 1
    # 考虑边缘像素，首先对影像边缘进行处理
    win_xs = 3
    win_ys = 3
    # xsize = 3
    # ysize = 3
    ext_ref_green = Extend(win_xs, win_ys, ref_green)
    kernel = np.ones((win_ys, win_xs)) / 9
    part1 = img_filtering(win_xs, win_ys, xsize, ysize, kernel, ext_ref_green * ext_ref_green)
    part2 = img_filtering(win_xs, win_ys, xsize, ysize, kernel, ext_ref_green)
    part2 **= 2
    variance = np.maximum(part1 - part2, 0)
    ref_green_std = np.sqrt(variance)
    ref_green_std_index = np.where(ref_green_std > 0.01)
    tmp_mask = np.ones_like(ref_green_std, dtype=np.byte)
    tmp_mask[ref_green_std_index] = 0
    # 对原始掩模进行处理
    ext_mask = Extend(win_xs, win_ys, mask)
    for irow in range(win_ys):
        for icol in range(win_xs):
            ext_mask[irow: irow + ysize: win_ys, icol: icol + xsize: win_xs] = \
                np.bitwise_and(ext_mask[irow: irow + ysize: win_ys, icol: icol + xsize: win_xs], tmp_mask)
    mask = ext_mask[win_ys // 2: -(win_ys // 2), win_xs // 2: -(win_xs // 2)]
    # ref_green = ext_ref_green = part1 = part2 = ref_green_std = ref_green_std_index = None
    # # 获取0.47微米波段反射率
    # ref_blue = bands_data[0, :, :] * 0.0001
    # ref_blue_index = np.where(ref_blue > 0.25)
    # mask[ref_blue_index] = 0
    # ref_blue = ref_blue_index = None
    # 临时写出掩模
    basename = os.path.splitext(os.path.basename(vis_file_path))[0]
    tmp_dir = r"F:\kuihua8\out"
    mask_path = os.path.join(tmp_dir, basename) + "_cld.tif"
    tif_driver = gdal.GetDriverByName('GTiff')
    out_ds = tif_driver.Create(mask_path, xsize, ysize, 1, gdal.GDT_Byte)
    # out_ds = tif_driver.Create(mask_path, xsize, ysize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geos)
    out_ds.SetProjection(prj)
    out_ds.GetRasterBand(1).WriteArray(mask)
    # out_ds.GetRasterBand(1).WriteArray(ref_green_std)
    out_ds = None
    return 1


def main(indir, outdir, shp_file=None):
    # search files
    files = searchfiles(indir, partfileinfo='*20200315_0300*.nc')
    for ifile in files:
        visual_file, nir_file = transformTogeotiff(ifile, outdir)
        # 预留裁剪模块，只处理感兴趣区域
        if shp_file is not None:
            pass
        # 生成云，水，冰雪掩模
        cld = generat_mask(visual_file)
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
    H8_dir_path = r"F:\kuihua8"
    out_dir_path = r"F:\kuihua8\out"
    shp = None
    main(H8_dir_path, out_dir_path, shp_file=shp)
    end_time = time.time()
    print("time: %.4f secs." % (end_time - start_time))
