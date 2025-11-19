import numpy as np
import os, cv2, math, sys
from utils.baseconfig import ERRMSG
from .inference_common import nms_for_cls, get_main_tar, get_center_tar, get_center_tar_list, get_tar_in_box, get_nearest_tar, iou_calc1, iof_calc
from .inference_common import _get_line_by_houf, show_and_save_result, screen_by_class_score, _get_angle_by_lines, postprocess_nms_screen, get_core_of_array

CLASSES = ['SF6ylb', 'aqmzc', 'bj_bpmh', 'bj_bpps', 'bj_wkps', 'bjdsyc_sx', 'bjdsyc_ywc', 'bjdsyc_ywj', 'bjdsyc_zz', 'bmwh', 'cysb_cyg', 'cysb_lqq', 'cysb_qtjdq', 'cysb_qyb', 'cysb_sgz', 'cysb_tg', 'ddjt', 'drq', 'drqgd', 'ecjxh', 'fhz_f', 'fhz_h', 'fhz_ztyc', 'gcc_mh', 'gcc_ps', 'gzzc', 'hxq_gjbs', 'hxq_gjtps', 'hxq_gjzc', 'hxq_yfps', 'hzyw', 'jdyxx', 'jdyxxsd', 'jsxs_ddjt', 'jsxs_ddyx', 'jsxs_ecjxh', 'jsxs_jdyxx', 'jyh', 'jyhbx', 'jyz_pl', 'kgg_ybf', 'kgg_ybh', 'kk_f', 'kk_h', 'mbhp', 'pzq', 'pzqcd', 'sly_bjbmyw', 'sly_dmyw', 'wcaqm', 'ws_ywyc', 'ws_ywzc', 'xldlb', 'xmbhyc', 'xmbhzc', 'ylb', 'ylsff', 'yw_gkxfw', 'yw_nc', 'ywb', 'ywc', 'ywj', 'ywzt_yfyc', 'ywzt_yfzc', 'yx', 'yxdgsg', 'zsd_l', 'zsd_m']

# 自动生成xml文件时的标签
cls_score_dict = {'SF6ylb':0.30, 'aqmzc':0.30, 'bj_bpmh':0.30, 'bj_bpps':0.30, 'bj_wkps':0.30, 'bjdsyc_sx':0.30, 'bjdsyc_ywc':0.30, 'bjdsyc_ywj':0.30, 'bjdsyc_zz':0.30, 'bmwh':0.30, 'cysb_cyg':0.30, 'cysb_lqq':0.30, 'cysb_qtjdq':0.30, 'cysb_qyb':0.30, 'cysb_sgz':0.30, 'cysb_tg':0.30, 'ddjt':0.30, 'drq':0.30, 'drqgd':0.30, 'ecjxh':0.30, 'fhz_f':0.30, 'fhz_h':0.30, 'fhz_ztyc':0.30, 'gcc_mh':0.30, 'gcc_ps':0.30, 'gzzc':0.30, 'hxq_gjbs':0.30, 'hxq_gjtps':0.30, 'hxq_gjzc':0.30, 'hxq_yfps':0.30, 'hzyw':0.30, 'jdyxx':0.30, 'jdyxxsd':0.30, 'jsxs_ddjt':0.30, 'jsxs_ddyx':0.30, 'jsxs_ecjxh':0.30, 'jsxs_jdyxx':0.30, 'jyh':0.30, 'jyhbx':0.30, 'jyz_pl':0.30, 'kgg_ybf':0.30, 'kgg_ybh':0.30, 'kk_f':0.30, 'kk_h':0.30, 'mbhp':0.30, 'pzq':0.30, 'pzqcd':0.30, 'sly_bjbmyw':0.30, 'sly_dmyw':0.30, 'wcaqm':0.30, 'ws_ywyc':0.30, 'ws_ywzc':0.30, 'xldlb':0.30, 'xmbhyc':0.30, 'xmbhzc':0.30, 'ylb':0.30, 'ylsff':0.30, 'yw_gkxfw':0.30, 'yw_nc':0.30, 'ywb':0.30, 'ywc':0.30, 'ywj':0.30, 'ywzt_yfyc':0.30, 'ywzt_yfzc':0.30, 'yx':0.30, 'yxdgsg':0.30, 'zsd_l':0.30, 'zsd_m':0.30}

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


