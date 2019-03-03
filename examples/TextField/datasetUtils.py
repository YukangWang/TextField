import os
import cv2
import numpy as np
import math
import scipy.io as sio

def ReadGt_synth(fileroot, filename):
    with open(fileroot+filename+'.gt') as f:
        lst = f.readlines()
    return lst

def GetPts_synth(lst, idx):
    pts_lst = lst[idx+1].split('\n')[0].split(' ')
    pts = np.zeros([4,2], dtype=np.int32)
    for pts_num in range(4):
        pts[pts_num][0] = round(float(pts_lst[2*pts_num+1]))
        pts[pts_num][1] = round(float(pts_lst[2*pts_num+2]))
    return pts

def ReadGt_ctw(fileroot, filename):
    with open(fileroot+filename+'.txt') as f:
        lst = f.readlines()
    return lst

def GetPts_ctw(lst, idx):
    pts_lst = lst[idx].split(',')
    pts = np.zeros([14,2], dtype=np.int32)
    for pts_num in range(14):
        pts[pts_num][0] = int(pts_lst[0])+int(pts_lst[2*pts_num+4])
        pts[pts_num][1] = int(pts_lst[1])+int(pts_lst[2*pts_num+5])
    return pts

def ReadGt_total(fileroot, filename):
    polygt = sio.loadmat(fileroot+'poly_gt_'+filename+'.mat')['polygt']
    return polygt

def GetPts_total(polygt, idx):
    pts_lst = []
    if polygt[idx][5].shape[0] == 0:
        hard = 1
    else:
        hard = int(polygt[idx][5][0]=='#')
    for pts_num in range(polygt[idx][1].shape[1]):
        pts_lst.append([polygt[idx][1][0][pts_num],polygt[idx][3][0][pts_num]])
    pts = np.array(pts_lst, dtype=np.int32)
    return pts, hard

def ReadGt_ic15(fileroot,filename):
    with open(fileroot+filename+'.txt') as f:
        lst = f.readlines()
    lst = [x.strip() for x in lst]
    return lst

def GetPts_ic15(lst, idx):
    pts_lst = lst[idx].replace('\xef\xbb\xbf','').split(',')
    x1 = int(pts_lst[0])
    y1 = int(pts_lst[1])
    x2 = int(pts_lst[2])
    y2 = int(pts_lst[3])
    x3 = int(pts_lst[4])
    y3 = int(pts_lst[5])
    x4 = int(pts_lst[6])
    y4 = int(pts_lst[7])
    hard = int(pts_lst[8]=='###')
    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    return pts, hard

def ReadGt_td(fileroot, filename):
    with open(fileroot+filename+'.gt') as f:
        lst = f.readlines()
    lst = [x.strip() for x in lst]
    return lst

def GetPts_td(lst, idx):
    pts_lst = lst[idx].split()
    hard = int(pts_lst[1])
    x = int(pts_lst[2])
    y = int(pts_lst[3])
    w = int(pts_lst[4])
    h = int(pts_lst[5])
    theta = float(pts_lst[6])
    x1 = math.cos(theta) * (-0.5 * w) - math.sin(theta) * (-0.5 * h) + x + 0.5 * w
    y1 = math.sin(theta) * (-0.5 * w) + math.cos(theta) * (-0.5 * h) + y + 0.5 * h
    x2 = math.cos(theta) * (0.5 * w) - math.sin(theta) * (-0.5 * h) + x + 0.5 * w
    y2 = math.sin(theta) * (0.5 * w) + math.cos(theta) * (-0.5 * h) + y + 0.5 * h
    x3 = math.cos(theta) * (0.5 * w) - math.sin(theta) * (0.5 * h) + x + 0.5 * w
    y3 = math.sin(theta) * (0.5 * w) + math.cos(theta) * (0.5 * h) + y + 0.5 * h
    x4 = math.cos(theta) * (-0.5 * w) - math.sin(theta) * (0.5 * h) + x + 0.5 * w
    y4 = math.sin(theta) * (-0.5 * w) + math.cos(theta) * (0.5 * h) + y + 0.5 * h
    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    return pts, hard