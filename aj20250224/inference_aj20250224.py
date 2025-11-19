import numpy as np
import os, cv2, math, sys
from utils.baseconfig import ERRMSG
from .inference_common import nms_for_cls, get_main_tar, get_center_tar, get_center_tar_list, get_tar_in_box, get_nearest_tar, iou_calc1, iof_calc
from .inference_common import _get_line_by_houf, show_and_save_result, screen_by_class_score, _get_angle_by_lines, postprocess_nms_screen, get_core_of_array

CLASSES = ['上跨cross_up', '下跨cross_down', '乙炔或氧气瓶垂直放置oxygen_vertically', '乙炔或氧气瓶水平放置oxygen_horizontally', '人员倒地down', '佩戴纱手套pdsst', '佩戴绝缘手套insulating_gloves', '假打电话fakephone', '冬装衣服winter_clothes', '冬装裤子winter_trousers', '冲锋钻charge_drill', '切割cutting', '动火fire', '单人扛梯single_ladder', '卷起来的工作裤roll_workclothes', '卷起来的非工作裤roll_noworkclothes', '双人扛梯double_ladder', '只有扶梯人，看不到登高人height_zero', '围栏fence', '安全带safteybelt', '安全帽hat', '工人work_men', '工作服上衣workclothes_clothes', '工作服裤子workclothes_trousers', '手hand', '手机phone', '扣环safteybelt_clasp', '护坡protection', '护目镜goggles', '抽烟smoking', '挂钩hook', '无井盖的孔洞holes', '无护坡unprotection', '有人扶梯的登高height_two', '未戴安全帽person', '柱子上的安全绳safteybelt_ring', '标示牌identification_plate', '水泥杆登高noheight_men', '灭火器extinguisher', '煤气罐gas_tank', '玩手机play_mobile', '电焊electric_welding', '登高无人扶梯height_one', '短裤shorts', '起重车辆载人（叉车载人）forklifts_have', '起重车辆载人（无人）forklifts_head', '起重车辆载人（有货物无人）forklifts_standing', '路人stranger_men', '防护面罩protective_mask', '非工作服上衣noworkclothes_clothes', '非工作服裤子noworkclothes_trousers', '马甲vest', '验电手的正常位置righting', '验电笔groundrod', '验电超过护环overring', '（卷起来）短袖roll_shirts']
# 自动生成xml文件时的标签
cls_score_dict = {'上跨cross_up':0.60, '下跨cross_down':0.60, '乙炔或氧气瓶垂直放置oxygen_vertically':0.60, '乙炔或氧气瓶水平放置oxygen_horizontally':0.60, '人员倒地down':0.60, '佩戴纱手套pdsst':0.60, '佩戴绝缘手套insulating_gloves':0.60, '假打电话fakephone':0.60, '冬装衣服winter_clothes':0.60, '冬装裤子winter_trousers':0.60, '冲锋钻charge_drill':0.60, '切割cutting':0.60, '动火fire':0.60, '单人扛梯single_ladder':0.60, '卷起来的工作裤roll_workclothes':0.60, '卷起来的非工作裤roll_noworkclothes':0.60, '双人扛梯double_ladder':0.60, '只有扶梯人，看不到登高人height_zero':0.60, '围栏fence':0.60, '安全带safteybelt':0.60, '安全帽hat':0.60, '工人work_men':0.60, '工作服上衣workclothes_clothes':0.60, '工作服裤子workclothes_trousers':0.60, '手hand':0.60, '手机phone':0.60, '扣环safteybelt_clasp':0.60, '护坡protection':0.60, '护目镜goggles':0.60, '抽烟smoking':0.60, '挂钩hook':0.60, '无井盖的孔洞holes':0.60, '无护坡unprotection':0.60, '有人扶梯的登高height_two':0.60, '未戴安全帽person':0.60, '柱子上的安全绳safteybelt_ring':0.60, '标示牌identification_plate':0.60, '水泥杆登高noheight_men':0.60, '灭火器extinguisher':0.60, '煤气罐gas_tank':0.60, '玩手机play_mobile':0.60, '电焊electric_welding':0.60, '登高无人扶梯height_one':0.60, '短裤shorts':0.60, '起重车辆载人（叉车载人）forklifts_have':0.60, '起重车辆载人（无人）forklifts_head':0.60, '起重车辆载人（有货物无人）forklifts_standing':0.60, '路人stranger_men':0.60, '防护面罩protective_mask':0.60, '非工作服上衣noworkclothes_clothes':0.60, '非工作服裤子noworkclothes_trousers':0.60, '马甲vest':0.60, '验电手的正常位置righting':0.60, '验电笔groundrod':0.60, '验电超过护环overring':0.60, '（卷起来）短袖roll_shirts':0.60}


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


