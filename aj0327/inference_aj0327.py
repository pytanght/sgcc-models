import numpy as np
import os, cv2, math, sys
from utils.baseconfig import ERRMSG
from .inference_common import nms_for_cls, get_main_tar, get_center_tar, get_center_tar_list, get_tar_in_box, get_nearest_tar, iou_calc1, iof_calc
from .inference_common import _get_line_by_houf, show_and_save_result, screen_by_class_score, _get_angle_by_lines, postprocess_nms_screen, get_core_of_array

CLASSES = ['上跨cross_up','下跨cross_down','佩戴绝缘手套insulating_gloves','冬装衣服winter_clothes','冬装裤子winter_trousers','单人扛梯single_ladder','只有扶梯人，看不到登高人height_zero','围栏fence','安全带safteybelt','安全帽hat','工人work_men','工作服上衣workclothes_clothes','工作服裤子workclothes_trousers','手hand','手机phone','扣环safteybelt_clasp','抽烟smoking','挂钩hook','无井盖的孔洞holes','有人扶梯的登高height_two','未戴安全帽person','柱子上的安全绳safteybelt_ring','标示牌identification_plate','水泥杆登高noheight_men','灭火器extinguisher','玩手机play_mobile','登高无人扶梯height_one','短裤shorts','路人stranger_men','非工作服上衣noworkclothes_clothes','非工作服裤子noworkclothes_trousers','马甲vest','验电手的正常位置righting','验电笔groundrod','（卷起来）短袖roll_shirts']
# 自动生成xml文件时的标签
cls_score_dict = {'上跨cross_up':0.05,'下跨cross_down':0.15,'佩戴绝缘手套insulating_gloves':0.30,'冬装衣服winter_clothes':0.20,'冬装裤子winter_trousers':0.55,'切割cutting':0.00,'单人扛梯single_ladder':0.35,'卷起来的工作裤roll_workclothes':0.00,'只有扶梯人，看不到登高人height_zero':0.00,'围栏fence':0.20,'安全带safteybelt':0.05,'安全帽hat':0.65,'工人work_men':0.25,'工作服上衣workclothes_clothes':0.30,'工作服裤子workclothes_trousers':0.45,'手hand':0.25,'手机phone':0.10,'扣环safteybelt_clasp':0.00,'抽烟smoking':0.35,'无井盖的孔洞holes':0.45,'有人扶梯的登高height_two':0.00,'未戴安全帽person':0.45,'标示牌identification_plate':0.45,'水泥杆登高noheight_men':0.00,'灭火器extinguisher':0.25,'玩手机play_mobile':0.05,'登高无人扶梯height_one':0.20,'短裤shorts':0.60,'起重车辆载人（叉车载人）forklifts_have':0.75,'起重车辆载人（无人）forklifts_head':0.00,'起重车辆载人（有货物无人）forklifts_standing':0.70,'路人stranger_men':0.25,'非工作服上衣noworkclothes_clothes':0.55,'非工作服裤子noworkclothes_trousers':0.50,'马甲vest':0.75,'验电手的正常位置righting':0.25,'验电笔groundrod':0.05,'验电超过护环overring':0.10,'（卷起来）短袖roll_shirts':0.45}


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


