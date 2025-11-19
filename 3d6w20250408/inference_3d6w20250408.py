import numpy as np
import os, cv2, math, sys
from utils.baseconfig import ERRMSG
from .inference_common import nms_for_cls, get_main_tar, get_center_tar, get_center_tar_list, get_tar_in_box, get_nearest_tar, iou_calc1, iof_calc
from .inference_common import _get_line_by_houf, show_and_save_result, screen_by_class_score, _get_angle_by_lines, postprocess_nms_screen, get_core_of_array

CLASSES = ['011_qjxx_hjxx_aqd', '011_qjxx_hjxx_aqm', '011_qjxx_hjxx_aqs', '011_qjxx_hjxx_bsp', '011_qjxx_hjxx_dbc', '011_qjxx_hjxx_dc', '011_qjxx_hjxx_dk', '011_qjxx_hjxx_dx', '011_qjxx_hjxx_dzkz', '011_qjxx_hjxx_dzwl', '011_qjxx_hjxx_dzyf', '011_qjxx_hjxx_gr', '011_qjxx_hjxx_gzf', '011_qjxx_hjxx_gzfkz', '011_qjxx_hjxx_hgwl', '011_qjxx_hjxx_jswl', '011_qjxx_hjxx_jyst', '011_qjxx_hjxx_kdybhcs', '011_qjxx_hjxx_kh', '011_qjxx_hjxx_s', '011_qjxx_hjxx_slgwl', '011_qjxx_hjxx_slwl', '011_qjxx_hjxx_wjj', '011_qjxx_hjxx_wzgkd', '011_qjxx_hjxx_wzwl', '011_qjxx_hjxx_xr', '011_qjxx_hjxx_ydb', '011_qjxx_hjxx_yzgkd', '011_qjxx_ryxx_dmzyry-ydcghh', '011_qjxx_ryxx_dmzyry-ydmcghh', '011_qjxx_ryxx_dmzyry-ypdsst', '011_qjxx_ryxx_gczyry-aqddggy', '011_qjxx_ryxx_gczyry-ptyrfc', '011_qjxx_ryxx_gczyry-wgaqs', '011_zywz_gkzyfgz_tszywrft', '011_zywz_xczyaqjs_ky（hc）aqwl', '011_zywz_xczyaqjs_ky（sk）aqwl', '011_zywz_xczyldfh_xczywdaqm', '011_zyxc_gczy_gczy']
# 自动生成xml文件时的标签
cls_score_dict = {'011_qjxx_hjxx_aqd':0.60, '011_qjxx_hjxx_aqm':0.60, '011_qjxx_hjxx_aqs':0.60, '011_qjxx_hjxx_bsp':0.60, '011_qjxx_hjxx_dbc':0.60, '011_qjxx_hjxx_dc':0.60, '011_qjxx_hjxx_dk':0.60, '011_qjxx_hjxx_dx':0.60, '011_qjxx_hjxx_dzkz':0.60, '011_qjxx_hjxx_dzwl':0.60, '011_qjxx_hjxx_dzyf':0.60, '011_qjxx_hjxx_gr':0.60, '011_qjxx_hjxx_gzf':0.60, '011_qjxx_hjxx_gzfkz':0.60, '011_qjxx_hjxx_hgwl':0.60, '011_qjxx_hjxx_jswl':0.60, '011_qjxx_hjxx_jyst':0.60, '011_qjxx_hjxx_kdybhcs':0.60, '011_qjxx_hjxx_kh':0.60, '011_qjxx_hjxx_s':0.60, '011_qjxx_hjxx_slgwl':0.60, '011_qjxx_hjxx_slwl':0.60, '011_qjxx_hjxx_wjj':0.60, '011_qjxx_hjxx_wzgkd':0.60, '011_qjxx_hjxx_wzwl':0.60, '011_qjxx_hjxx_xr':0.60, '011_qjxx_hjxx_ydb':0.60, '011_qjxx_hjxx_yzgkd':0.60, '011_qjxx_ryxx_dmzyry-ydcghh':0.60, '011_qjxx_ryxx_dmzyry-ydmcghh':0.60, '011_qjxx_ryxx_dmzyry-ypdsst':0.60, '011_qjxx_ryxx_gczyry-aqddggy':0.60, '011_qjxx_ryxx_gczyry-ptyrfc':0.60, '011_qjxx_ryxx_gczyry-wgaqs':0.60, '011_zywz_gkzyfgz_tszywrft':0.60, '011_zywz_xczyaqjs_ky（hc）aqwl':0.60, '011_zywz_xczyaqjs_ky（sk）aqwl':0.60, '011_zywz_xczyldfh_xczywdaqm':0.60, '011_zyxc_gczy_gczy':0.60}


merge_nms_xjj = []
iou_thres_xjj = [0.25,]

# 模型识别完的后处理
def inference_postprocess(result, imgpath, detect_cls=[], cls_score={}, infer_stage=4):
    if isinstance(imgpath, str):
        img = cv2.imdecode(np.fromfile(imgpath,dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        img = imgpath
    if infer_stage == 0: # 返回所有缺陷类别(不含小金具类别)，阈值取0.05
        return result, [], ''
    cls_score_dict.update(cls_score)
    result = postprocess_nms_screen(result, CLASSES, cls_score_dict, merge_nms_xjj, iou_thres_xjj)
    return result, [], ''


