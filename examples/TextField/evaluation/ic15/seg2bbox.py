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
        f = open(bbox_dir+'res_'+seg_lst[num][:-4]+'.txt','w')
        f.close()
        continue
    filtered_seg = 0
    for idx in range(np.amax(seg)):
        seg_mask = (seg == (idx+1)).astype(np.uint8)
        if np.sum(seg_mask) <= int(min_area):
            filtered_seg += 1
            continue
    if np.amax(seg) == filtered_seg:
        f = open(bbox_dir+'res_'+seg_lst[num][:-4]+'.txt','w')
        f.close()
        continue

    bbox = np.zeros((np.amax(seg)-filtered_seg, 8))
    bbox_idx = 0
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
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        bbox[bbox_idx] = box.reshape(1,8)
        bbox_idx += 1
        np.savetxt(bbox_dir+'res_'+seg_lst[num][:-4]+'.txt',bbox,fmt='%d',delimiter=',')
    print num