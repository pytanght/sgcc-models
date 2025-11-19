import numpy as np
import os, cv2, math, sys
from utils.baseconfig import ERRMSG
from .inference_common import nms_for_cls, get_main_tar, get_center_tar, get_center_tar_list, get_tar_in_box, get_nearest_tar, iou_calc1, iof_calc
from .inference_common import _get_line_by_houf, show_and_save_result, screen_by_class_score, _get_angle_by_lines, postprocess_nms_screen, get_core_of_array

CLASSES = ['站房设备数据储能指示亮023_zf_sbsj_cnzsl', '站房设备数据分闸指示亮023_zf_sbsj_fzzsl', '站房设备数据分闸指示灭023_zf_sbsj_fzzsm', '站房设备数据压板分023_zf_sbsj_ybf', '站房设备数据压板合023_zf_sbsj_ybh', '站房设备数据合闸指示亮023_zf_sbsj_hzzsl', '站房设备数据合闸指示灭023_zf_sbsj_hzzsm', '站房设备数据报警灯灭023_zf_sbsj_bjm', '站房设备数据柜体023_zf_sbsj_guiti', '站房设备数据远方023_zf_sbsj_yf', '站房设备数据铭牌023_zf_sbsj_mingpai']
# 自动生成xml文件时的标签
cls_score_dict = {'站房设备数据储能指示亮023_zf_sbsj_cnzsl':0.60, '站房设备数据分闸指示亮023_zf_sbsj_fzzsl':0.60, '站房设备数据分闸指示灭023_zf_sbsj_fzzsm':0.60, '站房设备数据压板分023_zf_sbsj_ybf':0.60, '站房设备数据压板合023_zf_sbsj_ybh':0.60, '站房设备数据合闸指示亮023_zf_sbsj_hzzsl':0.60, '站房设备数据合闸指示灭023_zf_sbsj_hzzsm':0.60, '站房设备数据报警灯灭023_zf_sbsj_bjm':0.60, '站房设备数据柜体023_zf_sbsj_guiti':0.60, '站房设备数据远方023_zf_sbsj_yf':0.60, '站房设备数据铭牌023_zf_sbsj_mingpai':0.60}


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
    result = postprocess_nms_screen(result, CLASSES, cls_score_dict, merge_nms_xjj, iou_thres_xjj)
    return result, [], ''


