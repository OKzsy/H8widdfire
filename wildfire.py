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
    visual_band_list = ['albedo_01', 'albedo_02',
                        'albedo_03', 'albedo_04', 'albedo_05']
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
    visual_out_ds = tif_driver.Create(
        visual_file, xsize, ysize, len(visual_band_list), gdal.GDT_Int16)
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
    nir_out_ds = tif_driver.Create(
        nir_file, xsize, ysize, len(nir_band_list) + 1, gdal.GDT_Float32)
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
    nir_out_ds.GetRasterBand(band_id + 1).WriteArray(soz_data)
    nir_out_ds = None
    return visual_file, nir_file


def generat_mask(vis_file_path, nir_file_path, mask_path):
    """利用可见光近红外波段数据生成云，水，冰雪掩模"""
    vis_ds = gdal.Open(vis_file_path)
    xsize = vis_ds.RasterXSize
    ysize = vis_ds.RasterYSize
    geos = vis_ds.GetGeoTransform()
    prj = vis_ds.GetProjection()
    bands_data = vis_ds.ReadAsArray()
    # 计算冰雪掩模
    ndsi = (bands_data[0, :, :] - bands_data[4, :, :]) / \
           (bands_data[0, :, :] + bands_data[4, :, :])
    mask = ndsi > 0.13
    ndsi = None
    # 计算水体掩模
    ndwi = (bands_data[3, :, :] - bands_data[2, :, :]) / \
           (bands_data[3, :, :] + bands_data[2, :, :])
    mask = np.bitwise_or(ndwi < 0, mask)
    ndwi = None
    # 计算云掩模
    # 读取亮温数据
    BT_ds = gdal.Open(nir_file_path)
    bt_data = BT_ds.ReadAsArray()
    part1 = np.bitwise_or(bt_data[1, :, :] < 265, bt_data[2, :, :] < 265)
    part2 = np.bitwise_and(bt_data[1, :, :] < 285,
                           (bt_data[0, :, :] - bt_data[1, :, :]) > 22)
    part3 = bands_data[2, :, :] > 5600
    cld_mask = np.bitwise_or(np.bitwise_or(part1, part2), part3)
    part1 = part2 = part3 = None
    bt_data = BT_ds = None
    mask = np.bitwise_or(cld_mask, mask).astype(np.byte) * 255
    # 写出掩模
    tif_driver = gdal.GetDriverByName('GTiff')
    out_ds = tif_driver.Create(mask_path, xsize, ysize, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geos)
    out_ds.SetProjection(prj)
    out_ds.GetRasterBand(1).WriteArray(mask)
    out_ds = None
    return mask_path


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
    mask_ds = gdal.GetDriverByName('MEM').Create(
        '', int(x_size), int(y_size), 1, gdal.GDT_Byte)
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
    off_ulx, off_uly = map(round, gdal.ApplyGeoTransform(
        raster_inv_geo, extent[0], extent[3]))
    # 右下
    off_drx, off_dry = map(round, gdal.ApplyGeoTransform(
        raster_inv_geo, extent[1], extent[2]))
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
    result_ds = gdal.GetDriverByName('MEM').Create(
        '', int(x_size), int(y_size), bandCount, dataType)
    result_ds.SetProjection(raster_prj)
    result_geo = [ulx, raster_geo[1], 0, uly, 0, raster_geo[5]]
    result_ds.SetGeoTransform(result_geo)
    # 获取掩模
    mask = mask_ds.GetRasterBand(1).ReadAsArray()
    # 对原始影像进行掩模并输出
    for band in range(bandCount):
        banddata = raster_ds.GetRasterBand(
            band + 1).ReadAsArray(int(ext[0]), int(ext[1]), int(x_size),
                                  int(y_size))
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
    raster_ds = None
    shp_ds = None
    # 如果裁剪后的新影像文件名和原始一致，则删除原来影像，写入裁剪后的影像，否则不删除
    if os.path.exists(out):
        os.remove(out)
    gdal.GetDriverByName("GTiff").CreateCopy(out, res, strict=1)
    return None


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


def img_mean(xs, ys, ori_xsize, ori_ysize, kernel, ext_img):
    """
    :param xs: 卷积核大小：列
    :param ys: 卷积核大小：行
    :param kernel: 卷积核
    :param ext_img: 经扩展后的影像
    :return: 滤波后的影像
    """
    # 使用切片后影像的波段书
    # 创建切片后存储矩阵
    filtered_img = np.zeros((ori_ysize, ori_xsize), dtype=np.float16)
    for irow in range(ys):
        for icol in range(xs):
            filtered_img += ext_img[irow: irow + ori_ysize, icol: icol + ori_xsize] * kernel[irow * xs + icol]
    return filtered_img


def detect_dynamic(point_x, point_y, rows, cols, data, valid_back, back_fire):
    IT04 = data[0, point_x, point_y]
    IT11 = data[1, point_x, point_y]
    IT41 = IT04 - IT11
    anchor_x = point_x
    anchor_y = point_y
    for win_size in range(3, 11, 2):
        radius = win_size // 2
        star_x = -radius if (anchor_x - radius) >= 0 else -anchor_x
        star_y = -radius if (anchor_y - radius) >= 0 else -anchor_y
        end_x = radius if (rows - anchor_x - radius) >= 1 else (rows - anchor_x - 1)
        end_y = radius if (cols - anchor_y - radius) >= 1 else (cols - anchor_y - 1)
        # 计算梯度
        xx = list(range(star_x + anchor_x, end_x + 1 + anchor_x))
        yy = list(range(star_y + anchor_y, end_y + 1 + anchor_y))
        grad_index = np.meshgrid(yy, xx)
        # 获取计算梯度区域数据
        valid_matrix = valid_back[grad_index[1], grad_index[0]]
        back_fire_matrix = back_fire[grad_index[1], grad_index[0]]
        thermal_matrix = data[0:2, grad_index[1], grad_index[0]]
        # 计算该窗口的统计信息
        bt04 = thermal_matrix[0, :, :]
        bk_bt04 = thermal_matrix[0, :, :] * back_fire_matrix
        bt04_index = np.where(bt04 != 0)
        # 判断是否满足基本条件
        all_num = bt04_index[0].shape[0]
        clear_num = np.where(valid_matrix == 1)[0].shape[0]
        if not ((clear_num / all_num >= 0.25) and (clear_num >= 8)):
            continue
        bt04 = bt04 * valid_matrix
        bt11 = thermal_matrix[1, :, :] * valid_matrix
        bt41 = bt04 - bt11
        BT04 = bt04[np.where(bt04 != 0)]
        BT41 = bt41[np.where(bt41 != 0)]
        IM04 = np.mean(BT04)
        IS04 = np.sum(np.abs(BT04 - IM04)) / clear_num
        IM41 = np.mean(BT41)
        IS41 = np.sum(np.abs(BT41 - IM41)) / clear_num
        bk_BT04 = bk_bt04[np.where(bk_bt04 != 0)]
        bk_IM04 = np.mean(bk_BT04)
        bk_IS04 = np.sum(np.abs(bk_BT04 - bk_IM04)) / np.where(bk_bt04 != 0)[0].shape[0]
        part1 = (IT41 - IM41) > (3.5 * IS41)
        part2 = (IT41 - IM41) > 6
        part3 = (IT04 - IM04) > (3 * IS04)
        part4 = (IT11 - IM41) > IS41 - 4
        part5 = bk_IS04 > 5
        if (part1 and part2 and part3 and (part4 or part5)):
            return True
    return False


def detectFire(thermal_file, cloud_mask, plant_mask):
    """
    利用热红外影像进行火点检测，同时对检测到的火点利用各种掩膜进行筛查
    :param thermal_file:
    :param cloud_mask:
    :param plant_mask:
    :return:
    """
    # 打开各类数据集,并获取数据
    thermal_ds = gdal.Open(thermal_file)
    thermal_data = thermal_ds.ReadAsArray()
    xsize = thermal_ds.RasterXSize
    ysize = thermal_ds.RasterYSize
    cloud_data = gdal.Open(cloud_mask).ReadAsArray()
    cloud_data = (cloud_data / 255).astype(dtype=np.byte)
    plant_data = gdal.Open(plant_mask).ReadAsArray()
    """
    火点检测思想为首先进行绝对火点检测，包括一类绝对火点和二类绝对火点。然后使用火点
    绝对火点掩膜、云掩膜剔除不需要进一步检测的像元点，对于剩余的像元点进行背景火点检测。
    """
    # 进行火点检测
    thermal_data = thermal_data * (1 - cloud_data) * plant_data
    # 进行第一类绝对火点检测
    Compensation_value = 15 * thermal_data[3, :, :]
    fire_mask = (thermal_data[0, :, :] - 2 * Compensation_value) > 330
    # 采用modis方法进行火点检测
    # 首先去除绝对火点
    bt04 = thermal_data[0, :, :]
    bt11 = thermal_data[1, :, :]
    # 获取潜在火点
    part1 = bt04 > 300
    part2 = (bt04 - bt11) > 8
    fire_pot = np.bitwise_and(np.bitwise_and(np.bitwise_and(part1, part2), 1 - cloud_data), plant_data)
    # 获取背景火点
    part1 = bt04 > 325
    part2 = (bt04 - bt11) > 20
    bk_fire = np.bitwise_and(part1, part2)
    # 获取有效背景
    valid_bk = np.bitwise_and(np.bitwise_and(1 - bk_fire, 1 - cloud_data), plant_data)
    # 进行动态窗口检测
    for irow in range(ysize):
        for icol in range(xsize):
            value = fire_pot[irow, icol]
            if not value:
                continue
            # 动态窗口进行判断
            fire_res = detect_dynamic(irow, icol, ysize, xsize, thermal_data, valid_bk, bk_fire)
            fire_pot[irow, icol] = fire_res
    fire_mask = np.bitwise_or(fire_mask, fire_pot)
    fire_mask = fire_mask.astype(dtype=np.byte)
    dirpath = os.path.dirname(thermal_file)
    basename = os.path.splitext(os.path.basename(thermal_file))[0]
    fire_mask_path = os.path.join(dirpath, basename) + '_fire_msk.tif'
    tif_driver = gdal.GetDriverByName('GTiff')
    out_ds = tif_driver.Create(fire_mask_path, xsize, ysize, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(thermal_ds.GetGeoTransform())
    out_ds.SetProjection(thermal_ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(fire_mask)
    out_ds = None
    thermal_ds = None
    return 1


def action(indir, outdir, veg_msk, shp_file):
    # 搜索文件,判断是否为文件夹
    if os.path.isdir(indir):
        files = searchfiles(indir, partfileinfo='*.nc')
    else:
        files = [indir]
    for ifile in files:
        basename = os.path.splitext(os.path.basename(ifile))[0]
        visual_file, nir_file = transformTogeotiff(ifile, outdir)
        # 预留裁剪模块，只处理感兴趣区域
        if shp_file is not None:
            clip(visual_file, shp_file, visual_file)
            clip(nir_file, shp_file, nir_file)
            # 利用shp文件裁剪植被掩模
            sub_veg_msk = os.path.join(outdir, basename) + '_veg_msk.tif'
            clip(veg_msk, shp_file, sub_veg_msk)
        # 生成云，水，冰雪掩模
        cloud_mask_path = os.path.join(outdir, basename) + '_cld_msk.tif'
        cld_msk = generat_mask(visual_file, nir_file, cloud_mask_path)
        # 火点检测
        res = detectFire(nir_file, cld_msk, sub_veg_msk)
    return None


def main(argv):
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument('srcdir', help="The folder path that contains the file or the file path")
    parser.add_argument('dstdir', help="The result folder path")
    parser.add_argument('vector', help="Vector file path for area of interest")
    parser.add_argument('plant', help="Vegetation mask file path")
    args = parser.parse_args(argv[1:])
    if not os.path.exists(args.dstdir):
        os.makedirs(args.dstdir)
    # 支持中文路径
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 支持中文属性字段
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    # 注册所有ogr驱动
    ogr.RegisterAll()
    # 注册所有gdal驱动
    gdal.AllRegister()
    start_time = time.time()
    H8_dir_path = args.srcdir
    out_dir_path = args.dstdir
    shp = args.vector
    vegetation_mask = args.plant
    action(H8_dir_path, out_dir_path, veg_msk=vegetation_mask, shp_file=shp)
    end_time = time.time()
    print("time: %.4f secs." % (end_time - start_time))


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
