import numpy as np
import os, cv2, math, sys
from utils.baseconfig import ERRMSG
from .inference_common import nms_for_cls, get_main_tar, get_center_tar, get_center_tar_list, get_tar_in_box, get_nearest_tar, iou_calc1, iof_calc
from .inference_common import _get_line_by_houf, show_and_save_result, screen_by_class_score, _get_angle_by_lines, postprocess_nms_screen, get_core_of_array

CLASSES = ['标准器具铭牌出厂编号正常031_bzqjmc_ccbh_zc', '标准器具铭牌型号正常031_bzqjmc_xh_zc', '电能表电表条形码正常031_dnb_dbtxm_zc', '电能表电表示数0数字031_dnb_dbss_sz0', '电能表电表示数1数字031_dnb_dbss_sz1', '电能表电表示数2数字031_dnb_dbss_sz2', '电能表电表示数3数字031_dnb_dbss_sz3', '电能表电表示数4数字031_dnb_dbss_sz4', '电能表电表示数5数字031_dnb_dbss_sz5', '电能表电表示数6数字031_dnb_dbss_sz6', '电能表电表示数7数字031_dnb_dbss_sz7', '电能表电表示数8数字031_dnb_dbss_sz8', '电能表电表示数9数字031_dnb_dbss_sz9', '电能表电表示数小数点031_dnb_dbss_xsd', '电能表电表示数峰值031_dnb_dbss_fz', '电能表电表示数总值031_dnb_dbss_zz', '电能表电表示数谷值031_dnb_dbss_gz', '电能表电量示数正常031_dnb_dlss_zc']
# 自动生成xml文件时的标签
cls_score_dict = {'标准器具铭牌出厂编号正常031_bzqjmc_ccbh_zc':0.60, '标准器具铭牌型号正常031_bzqjmc_xh_zc':0.60, '电能表电表条形码正常031_dnb_dbtxm_zc':0.60, '电能表电表示数0数字031_dnb_dbss_sz0':0.60, '电能表电表示数1数字031_dnb_dbss_sz1':0.60, '电能表电表示数2数字031_dnb_dbss_sz2':0.60, '电能表电表示数3数字031_dnb_dbss_sz3':0.60, '电能表电表示数4数字031_dnb_dbss_sz4':0.60, '电能表电表示数5数字031_dnb_dbss_sz5':0.60, '电能表电表示数6数字031_dnb_dbss_sz6':0.60, '电能表电表示数7数字031_dnb_dbss_sz7':0.60, '电能表电表示数8数字031_dnb_dbss_sz8':0.60, '电能表电表示数9数字031_dnb_dbss_sz9':0.80, '电能表电表示数小数点031_dnb_dbss_xsd':0.60, '电能表电表示数峰值031_dnb_dbss_fz':0.60, '电能表电表示数总值031_dnb_dbss_zz':0.60, '电能表电表示数谷值031_dnb_dbss_gz':0.60, '电能表电量示数正常031_dnb_dlss_zc':0.60}


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


