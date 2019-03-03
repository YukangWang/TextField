import sys
import os
import cv2
import numpy as np

seg_dir = sys.argv[1]
bbox_dir = sys.argv[2]
min_area = sys.argv[3]

seg_lst = os.listdir(seg_dir)
for num in range(len(seg_lst)):
    seg = cv2.imread(seg_dir+seg_lst[num], -1)
    if np.amax(seg) == 0:
        f = open(bbox_dir+seg_lst[num][:-4]+'.txt','w')
        f.close()
        continue
    filtered_seg = 0
    for idx in range(np.amax(seg)):
        seg_mask = (seg == (idx+1)).astype(np.uint8)
        if np.sum(seg_mask) <= int(min_area):
            filtered_seg += 1
            continue
    if np.amax(seg) == filtered_seg:
        f = open(bbox_dir+seg_lst[num][:-4]+'.txt','w')
        f.close()
        continue

    f = open(bbox_dir+seg_lst[num][:-4]+'.txt','w')
    for idx in range(np.amax(seg)):
        seg_mask = (seg == (idx+1)).astype(np.uint8)
        if np.sum(seg_mask) <= int(min_area):
            filtered_seg += 1
            continue
        contours, hierarchy = cv2.findContours(seg_mask, 1, 2)
        maxc, maxc_idx = 0, 0
        for i in range(len(contours)):
            if len(contours[i]) > maxc:
                maxc = len(contours[i])
                maxc_idx = i
        cnt = contours[maxc_idx]
        bbox = np.transpose(cnt, (1,0,2))
        bbox = bbox[0].astype(np.float64)
        bbox = bbox[:,::-1]
        if bbox.shape[0]*bbox.shape[1] < 8:
            continue
        bbox = bbox.reshape(1, bbox.shape[0]*bbox.shape[1])
        np.savetxt(f,bbox,fmt='%d',delimiter=',')
    print num
