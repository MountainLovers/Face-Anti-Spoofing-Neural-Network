# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf, gen_anchor
import argparse
import torch.backends.cudnn as cudnn
import os
from PIL import Image

STD_SIZE = 120
OULU_DATABASE_PATH = '/mnt/hdd.user/datasets/FAS/Oulu-NPU'
SMALL_OULU_DATABASE_PATH = '/workspace/Face-Anti-Spoofing-Neural-Network/SMALLOULU'
SUBDIRS = {"train": "Train_files", "test": "Test_files", "dev": "Dev_files"}

def resize_depth(imgdepth):
    img=np.array(imgdepth)
    res=np.zeros((32,32,1),dtype=float)
    for x in range(32):
        realx = 8*x
        for y in range(32):
            realy=8*y
            res[x,y,0]=np.mean(img[realx:realx+8,realy:realy+8])
    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    args = parser.parse_args()

    folder = np.load('/workspace/Face-Anti-Spoofing-Neural-Network/data_processing/folder_small2.npz')
    
    label = np.load('/workspace/Face-Anti-Spoofing-Neural-Network/data_processing/label_small2.npz')

    Anchors = {}
    Labels_D = {}
    
    # 1. 保存预训练模型
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. 人脸检测功能
    face_detector = dlib.get_frontal_face_detector()

    # 3. 检测
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    for item in folder:
        # print(type(str(folder[item])))
        src_path = str(folder[item])
        print("pic: {}, label: {}".format(src_path, label[item]))
        img_ori = cv2.imread(src_path)
        rects = face_detector(img_ori, 1)
        if len(rects) != 0:
        
            for rect in rects:
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)
                img = crop_img(img_ori, roi_box)
                img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                
            vertices_lst = [] 
            vertices = predict_dense(param, roi_box)
            vertices_lst.append(vertices)
                    
            Anchor = gen_anchor(param=param,kernel_size=args.paf_size)
            Anchors[item]=Anchor
            
            depth_path = src_path[:-4] + "_depth.png"
            depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1) 
            if(int(item)%100==0):
                print(item)
            if label[item]==1: #real face
                print(depth_path)
                Labels_D[item]=resize_depth(depths_img)
                cv2.imwrite(depth_path, depths_img)
            else: #spoof face
                Labels_D[item]=np.zeros((32,32,1),dtype=float)
        else:
            #Case our cropping didn't work
            Anchors[item] = np.zeros((2,4096),dtype=float)
            print('fausse image:'+item)
            Labels_D[item] = np.zeros((32,32,1),dtype=float)
                    
    np.savez("anchors.npz",**Anchors)
    np.savez("labels_D.npz",**Labels_D)
    np.savez("folder_dp2.npz",**folder)
