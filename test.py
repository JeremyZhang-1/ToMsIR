# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""


# -*- coding: utf-8 -*-
"""
test_demo.py —— 修改版
修改记录:
1. GetClass 类别列表补全为 8 类，顺序与训练时对齐
2. GetClass 内部对 out_t 做 squeeze，避免 2D argmax 结果不稳定
3. 导入改为 model0202_fixed
4. load_checkpoint 增加文件存在性检查，避免直接崩溃
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import os
from model0225 import *   # 修改3: 与修正后的模型文件对齐


os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_checkpoint(checkpoint_dir):
    ckpt_path = checkpoint_dir + 'checkpoint.pth_0227.tar'

    # 修改4: 增加文件存在性检查
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model_info = torch.load(ckpt_path)
    net        = MainNet()
    device_ids = [0]
    model      = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(model_info['state_dict'])
    optimizer  = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    cur_epoch  = model_info['epoch']

    return model, optimizer, cur_epoch


def GetClass(out):
    # 修改1: 8类列表，顺序与训练时 types=0~7 严格对齐
    # 训练时: 0=clear,1=haze,2=rain,3=snow,4=rainsnow,5=hazerain,6=hazesnow,7=hazerainsnow
    DegradedTypes = [
        'clear',                    # 0
        'haze',                     # 1
        'rain',                     # 2
        'snow',                     # 3
        'rain + snow',              # 4
        'haze + rain',              # 5
        'haze + snow',              # 6
        'haze + rain + snow',       # 7
    ]

    # 修改2: squeeze 确保是 1D 数组再取 argmax，避免 2D 时结果不稳定
    out_np       = out.cpu().detach().numpy()
    out_np       = out_np.squeeze()          # (1,8) -> (8,)
    pred_idx     = int(np.argmax(out_np))
    DegradedType = DegradedTypes[pred_idx]

    return DegradedType


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])


if __name__ == '__main__':
    checkpoint_dir = './checkpoint/'
    test_dir = './DAWN'
    result_dir = './result_DAWN'  
    testfiles      = os.listdir(test_dir)

    os.makedirs(result_dir, exist_ok=True)   # 结果目录不存在时自动创建

    print('> Loading checkpoint ...')
    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir)
    print(f'> Loaded epoch: {cur_epoch}')

    model.eval()   # eval() 放在循环外，避免重复调用
    for f in range(len(testfiles)):
        with torch.no_grad():
            img_path = os.path.join(test_dir, testfiles[f])
            img_c    = cv2.imread(img_path)

            if img_c is None:
                print(f'Warning: cannot read {img_path}, skipped.')
                continue

            img_c     = img_c / 255.0
            h, w, c   = img_c.shape
            img_l     = hwc_to_chw(np.array(img_c).astype('float32'))
            input_var = torch.from_numpy(img_l.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()

            s              = time.time()
            out_r, out_t   = model(input_var)
            e              = time.time()

            degraded_type  = GetClass(out_t)
            out_r          = chw_to_hwc(out_r.squeeze().cpu().detach().numpy())

            print('Input: %s | Size: (%d,%d) | Time: %.4fs | Type: %s'
                  % (testfiles[f], h, w, e - s, degraded_type))

            save_path = os.path.join(result_dir, testfiles[f][:-4] + '_UMWR.png')
            cv2.imwrite(save_path, np.clip(out_r * 255, 0.0, 255.0))
	  
				
			
			

