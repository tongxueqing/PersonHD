import imp
import pickle
import tqdm
import pandas as pd
import json
import numpy as np
MISSING_VALUE = -1

PARTS_SEL = [0, 1, 14, 15, 16, 17]


target_annotation = "/data2/xueqing_tong/dataset/cropped_front/resize256/label/pose_label_test_256.pkl"
pred_annotations = ['/data2/xueqing_tong/codes/SPGNet/checkpoints/PoseTransfer_personHD_2e5_front/output_8.csv','/data2/xueqing_tong/codes/Pose-Transfer/results_personHD_2e5_epoch/person_PATN_front_2e5/test_4/images/fake_p2.csv','/data2/xueqing_tong/codes/MUST-GAN/results_personHD_test/MUST-GAN-personHD_2e5_front/test_10/images/fake_p2.csv',"/data2/xueqing_tong/codes/SPGNet/checkpoints/PoseTransfer_personHD_finetune_jitter_2e5_front/output_40.csv","/data2/xueqing_tong/codes/SPGNet/checkpoints/PoseTransfer_personHD_2e5_front/output_8.csv"]


# target_annotation ='/data2/xueqing_tong/dataset/cropped_side/image_cropped_side_test_256.pkl'
# pred_annotation = './checkpoints/PoseTransfer_personHD_2e5_side/output_8.csv'
# pred_annotation ='/data2/xueqing_tong/codes/Pose-Transfer/results_personHD_2e5_epoch/person_PATN_side_2e5/test_4/images/fake_p2.csv'
# pred_annotation ='/data2/xueqing_tong/codes/MUST-GAN/results_personHD_test/MUST-GAN-personHD_side_2e5/test_10/images/fake_p2.csv'
# pred_annotation="/data2/xueqing_tong/codes/SPGNet/checkpoints/PoseTransfer_personHD_finetune_jitter_2e5_side/output_40.csv"

'''
  hz: head size
  alpha: norm factor
  px, py: predict coords
  tx, ty: target coords
'''
def isRight(px, py, tx, ty, hz, alpha):
    if px == -1 or py == -1 or tx == -1 or ty == -1:
        return 0

    if abs(px - tx) < hz[0] * alpha and abs(py - ty) < hz[1] * alpha:
        return 1
    else:
        return 0


def how_many_right_seq(px, py, tx, ty, hz, alpha):
    nRight = 0
    for i in range(len(px)):
        nRight = nRight + isRight(px[i], py[i], tx[i], ty[i], hz, alpha)

    return nRight


def ValidPoints(tx):
    nValid = 0
    for item in tx:
        if item != -1:
            nValid = nValid + 1
    return nValid


def get_head_wh(x_coords, y_coords):
    final_w, final_h = -1, -1
    component_count = 0
    save_componets = []
    for component in PARTS_SEL:
        if x_coords[component] == MISSING_VALUE or y_coords[component] == MISSING_VALUE:
            continue
        else:
            component_count += 1
            save_componets.append([x_coords[component], y_coords[component]])
    if component_count >= 2:
        x_cords = []
        y_cords = []
        for component in save_componets:
            x_cords.append(component[0])
            y_cords.append(component[1])
        xmin = min(x_cords)
        xmax = max(x_cords)
        ymin = min(y_cords)
        ymax = max(y_cords)
        final_w = xmax - xmin
        final_h = ymax - ymin
    return final_w, final_h


# tAnno = pd.read_csv(target_annotation, sep=':')
with open(target_annotation,'rb') as f:
    tAnno = pickle.load(f)

d1=dict()
for pred_annotation in pred_annotations:
    pAnno = pd.read_csv(pred_annotation, sep=':')

    pRows = pAnno.shape[0]

    nAll = 0
    nCorrect = 0
    s=0
    sample=[]
    alpha = 0.5
    for i in tqdm.tqdm(range(pRows)):
        pValues = pAnno.iloc[i].values
        pname = pValues[0]
        pycords = json.loads(pValues[1])  # list of numbers
        pxcords = json.loads(pValues[2])

        tname = pname

        ####
        if '___' in tname:
            tname =tname.split('___')[1].split('.')[0]
        else:
            tname =tname.split('__')[1].split('.')[0]
        # if tname.count('_')==5:
        #     ns = tname.split('_')
        #     tname = ns[0]+ns[1]+'_'+ns[2]+'_'+ns[3]+ns[4]+'_'+ns[5]
        
        tValues = tAnno[tname]
        if type(tValues)==list:
            tValues=np.array(tValues)
        tycords = tValues[:,1]  # list of numbers
        txcords = tValues[:,0]


        xBox, yBox = get_head_wh(txcords, tycords)
        if xBox == -1 or yBox == -1:
            continue

        head_size = (xBox, yBox)
        nAll = nAll + ValidPoints(tycords)
        nCorrect = nCorrect + how_many_right_seq(pxcords, pycords, txcords, tycords, head_size, alpha)

        ratio=how_many_right_seq(pxcords, pycords, txcords, tycords, head_size, alpha)/(ValidPoints(tycords)+0.001)
        if ratio<0.6 and ValidPoints(tycords)>0 :
            print(tname)
            s+=1
            sample.append(tname)
    d1[pred_annotation]=sample
# print('wrong sample:',s)
 
# print('%d/%d %f' % (nCorrect, nAll, nCorrect * 1.0 / nAll)) 
for i in range(len(pred_annotation)):
    for j in range(i+1,len(pred_annotation)):
        
        x1=d1[pred_annotations[i]]
        x2=d1[pred_annotations[j]]
        print(pred_annotations[i])
        print(pred_annotations[j])
        print('len:',len(x1),len(x2))
        print("&:",len(set(x1)&set(x2)))
        print("|:",len(set(x1)|set(x2)))

