import caffe
import numpy as np
import cv2
import random
import os
import math
from datasetUtils import *

def random_crop(mask, scale=(0.1, 1.0), ratio=(0.3, 3.0)):
    for attempt in range(10):
        height = mask.shape[0]
        width = mask.shape[1]
        area = height*width
        target_area = np.random.uniform(*scale) * area
        aspect_ratio = np.random.uniform(*ratio)
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        if np.random.random() < 0.5:
            h, w = w, h
        if h < height and w < width:
            i = np.random.randint(0, height-h)
            j = np.random.randint(0, width-w)
            return i, j, h, w
    m = min(height, width)
    i = (height-m) // 2
    j = (width-m) // 2
    return i, j, h, w

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # data layer config
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.dataset = params['dataset']
        self.patch_size = int(params['patch_size'])
        self.seed = params['seed']
        self.mean = np.array(params['mean'])
        self.random = True

        # three tops: data, label and weight
        if len(top) != 3:
            raise Exception("Need to define three tops: data, label and weight.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # set directory for each dataset here
        if self.dataset == 'synth':
            self.fnLst = open(self.data_dir+'SynthText/list_train.txt').readlines()
        elif self.dataset == 'ctw':
            self.fnLst = os.listdir(self.data_dir+'ctw1500/train/text_image/')
        elif self.dataset == 'total':
            self.fnLst = os.listdir(self.data_dir+'totaltext/Images/Train/')
        elif self.dataset == 'ic15':
            self.fnLst = os.listdir(self.data_dir+'icdar2015/train_images/')
        elif self.dataset == 'td':
            self.fnLst = os.listdir(self.data_dir+'MSRA-TD500/train/text_image/')
        else:
            raise Exception("Invalid dataset.")

        # randomization: seed and pick
        self.idx = 0
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.fnLst)-1)

    def reshape(self, bottom, top):
        # load image, label and weight
        if self.dataset == 'synth':
            self.data, self.label, self.weight = self.loadsynth(self.fnLst[self.idx])
        elif self.dataset == 'ctw':
            self.data, self.label, self.weight = self.loadctw(self.fnLst[self.idx])
        elif self.dataset == 'total':
            self.data, self.label, self.weight = self.loadtotal(self.fnLst[self.idx])
        elif self.dataset == 'ic15':
            self.data, self.label, self.weight = self.loadic15(self.fnLst[self.idx])
        elif self.dataset == 'td':
            self.data, self.label, self.weight = self.loadtd(self.fnLst[self.idx])
        else:
            raise Exception("Invalid dataset.")

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(1, *self.weight.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.weight

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.fnLst)-1)
        else:
            self.idx += 1
            if self.idx == len(self.fnLst):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def loadsynth(self, idx):
        # generate mask
        image = cv2.imread('{}/SynthText/{}'.format(self.data_dir, idx[:-1][2:]),1)
        height = image.shape[0]
        width = image.shape[1]
        mask = np.zeros(image.shape, dtype=np.uint8)
        bbox = ReadGt_synth('{}/SynthText/'.format(self.data_dir), idx[:-5][2:])
        for bboxi in range(int(bbox[0])):
            pts = GetPts_synth(bbox, bboxi)
            cv2.fillPoly(mask,[pts],(255,255,255),1)

        # random crop parameters
        attempt = 0
        min_overlap = [0.1, 0.3, 0.5, 0.7]
        while attempt < 10:
            patch_h0, patch_w0, patch_h, patch_w = random_crop(mask)
            overlap = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w] > 0
            random_idx = np.random.randint(0, 4)
            if np.sum(overlap) >= np.sum(mask>0)*min_overlap[random_idx]:
                break
            attempt += 1
        if attempt == 10:
            patch_h = min(height, width)
            patch_w = patch_h
            patch_h0 = (height-patch_h) // 2
            patch_w0 = (width-patch_w) // 2

        # random crop & resize
        image = image[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)

        # random rotate
        prob = np.random.uniform(0,1)
        if prob <= 0.2:
            rtimes = 1
        elif prob >= 0.8:
            rtimes = 3
        else:
            rtimes = 0
        for rcount in range(rtimes):
            image = np.rot90(image)
        # cv2.imwrite('train_input/{}'.format(str(self.idx)+'.jpg'),image)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute vec
        accumulation = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        for bboxi in range(int(bbox[0])):
            pts = GetPts_synth(bbox, bboxi)
            seg = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(seg,[pts],(255,255,255),1)
            segPatch = seg[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
            for rcount in range(rtimes):
                segPatch = np.rot90(segPatch)
            if segPatch.max()!=255: continue
            segPatch = cv2.resize(segPatch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(segPatch,(0,0),(self.patch_size-1,self.patch_size-1),(0,0,0),1)

            img = cv2.cvtColor(segPatch,cv2.COLOR_BGR2GRAY)
            img = (img > 128).astype(np.uint8)
            dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
            index = np.copy(labels)
            index[img > 0] = 0
            place = np.argwhere(index > 0)
            nearCord = place[labels-1,:]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, self.patch_size, self.patch_size))
            nearPixel[0,:,:] = x
            nearPixel[1,:,:] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel
            dist = np.sqrt(np.sum(diff**2, axis = 0))

            direction = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
            direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
            direction[2,img > 0] = bboxi+1

            accumulation[0,img > 0] = 0
            accumulation[1,img > 0] = 0
            accumulation[2,img > 0] = 0
            accumulation = accumulation + direction
        vec = np.stack((accumulation[0], accumulation[1]))

        # compute weight
        weight = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        posRegion = accumulation[2]>0
        posCount = np.sum(posRegion)
        if posCount != 0:
            bboxRemain = 0
            for bboxi in range(int(bbox[0])):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                bboxRemain = bboxRemain+1
            bboxAve = float(posCount)/bboxRemain
            for bboxi in range(int(bbox[0])):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                pixAve = bboxAve/overlapCount_bboxi
                weight = weight*(~overlap_bboxi) + pixAve*overlap_bboxi
        weight = weight[np.newaxis, ...]

        return image, vec, weight

    def loadctw(self, idx):
        # generate mask
        image = cv2.imread('{}/ctw1500/train/text_image/{}'.format(self.data_dir, idx), 1)
        height = image.shape[0]
        width = image.shape[1]
        mask = np.zeros(image.shape, dtype=np.uint8)
        bbox = ReadGt_ctw('{}/ctw1500/train/text_label_curve/'.format(self.data_dir), idx[:-4])
        for bboxi in range(len(bbox)):
            pts = GetPts_ctw(bbox, bboxi)
            cv2.fillPoly(mask,[pts],(255,255,255),1)

        # random crop parameters
        attempt = 0
        min_overlap = [0.1, 0.3, 0.5, 0.7]
        while attempt < 10:
            patch_h0, patch_w0, patch_h, patch_w = random_crop(mask)
            overlap = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w] > 0
            random_idx = np.random.randint(0, 4)
            if np.sum(overlap) >= np.sum(mask>0)*min_overlap[random_idx]:
                break
            attempt += 1
        if attempt == 10:
            patch_h = min(height, width)
            patch_w = patch_h
            patch_h0 = (height-patch_h) // 2
            patch_w0 = (width-patch_w) // 2

        # random crop & resize
        image = image[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)

        # random rotate
        prob = np.random.uniform(0,1)
        if prob <= 0.2:
            rtimes = 1
        elif prob >= 0.8:
            rtimes = 3
        else:
            rtimes = 0
        for rcount in range(rtimes):
            image = np.rot90(image)
        # cv2.imwrite('train_input/{}'.format(idx),image)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute vec
        accumulation = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        for bboxi in range(len(bbox)):
            pts = GetPts_ctw(bbox, bboxi)
            seg = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(seg,[pts],(255,255,255),1)
            segPatch = seg[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
            for rcount in range(rtimes):
                segPatch = np.rot90(segPatch)
            if segPatch.max()!=255: continue
            segPatch = cv2.resize(segPatch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(segPatch,(0,0),(self.patch_size-1,self.patch_size-1),(0,0,0),1)

            img = cv2.cvtColor(segPatch,cv2.COLOR_BGR2GRAY)
            img = (img > 128).astype(np.uint8)
            dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
            index = np.copy(labels)
            index[img > 0] = 0
            place = np.argwhere(index > 0)
            nearCord = place[labels-1,:]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, self.patch_size, self.patch_size))
            nearPixel[0,:,:] = x
            nearPixel[1,:,:] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel
            dist = np.sqrt(np.sum(diff**2, axis = 0))

            direction = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
            direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
            direction[2,img > 0] = bboxi+1

            accumulation[0,img > 0] = 0
            accumulation[1,img > 0] = 0
            accumulation[2,img > 0] = 0
            accumulation = accumulation + direction
        vec = np.stack((accumulation[0], accumulation[1]))

        # compute weight
        weight = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        posRegion = accumulation[2]>0
        posCount = np.sum(posRegion)
        if posCount != 0:
            bboxRemain = 0
            for bboxi in range(len(bbox)):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                bboxRemain = bboxRemain+1
            bboxAve = float(posCount)/bboxRemain
            for bboxi in range(len(bbox)):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                pixAve = bboxAve/overlapCount_bboxi
                weight = weight*(~overlap_bboxi) + pixAve*overlap_bboxi
        weight = weight[np.newaxis, ...]
        
        return image, vec, weight

    def loadtotal(self, idx):
        # generate mask
        image = cv2.imread('{}/totaltext/Images/Train/{}'.format(self.data_dir, idx), 1)
        height = image.shape[0]
        width = image.shape[1]
        mask = np.zeros(image.shape, dtype=np.uint8)
        bbox = ReadGt_total('{}/totaltext/Groundtruth/Polygon/Train/'.format(self.data_dir), idx[:-4])
        for bboxi in range(bbox.shape[0]):
            pts, hard = GetPts_total(bbox, bboxi)
            if hard == 1: continue
            cv2.fillPoly(mask,[pts],(255,255,255),1)

        # random crop parameters
        attempt = 0
        min_overlap = [0.1, 0.3, 0.5, 0.7]
        while attempt < 10:
            patch_h0, patch_w0, patch_h, patch_w = random_crop(mask)
            overlap = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w] > 0
            random_idx = np.random.randint(0, 4)
            if np.sum(overlap) >= np.sum(mask>0)*min_overlap[random_idx]:
                break
            attempt += 1
        if attempt == 10:
            patch_h = min(height, width)
            patch_w = patch_h
            patch_h0 = (height-patch_h) // 2
            patch_w0 = (width-patch_w) // 2

        # random crop & resize
        image = image[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)

        # random rotate
        prob = np.random.uniform(0,1)
        if prob <= 0.2:
            rtimes = 1
        elif prob >= 0.8:
            rtimes = 3
        else:
            rtimes = 0
        for rcount in range(rtimes):
            image = np.rot90(image)
        # cv2.imwrite('train_input/{}'.format(idx),image)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute vec
        accumulation = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        for bboxi in range(bbox.shape[0]):
            pts, hard = GetPts_total(bbox, bboxi)
            seg = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(seg,[pts],(255,255,255),1)
            segPatch = seg[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
            for rcount in range(rtimes):
                segPatch = np.rot90(segPatch)
            if segPatch.max()!=255: continue
            segPatch = cv2.resize(segPatch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(segPatch,(0,0),(self.patch_size-1,self.patch_size-1),(0,0,0),1)

            img = cv2.cvtColor(segPatch,cv2.COLOR_BGR2GRAY)
            img = (img > 128).astype(np.uint8)
            dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
            index = np.copy(labels)
            index[img > 0] = 0
            place = np.argwhere(index > 0)
            nearCord = place[labels-1,:]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, self.patch_size, self.patch_size))
            nearPixel[0,:,:] = x
            nearPixel[1,:,:] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel
            dist = np.sqrt(np.sum(diff**2, axis = 0))

            direction = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
            direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
            if hard == 0:
                direction[2,img > 0] = bboxi+1

            accumulation[0,img > 0] = 0
            accumulation[1,img > 0] = 0
            accumulation[2,img > 0] = 0
            accumulation = accumulation + direction
        vec = np.stack((accumulation[0], accumulation[1]))

        # compute weight
        weight = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        posRegion = accumulation[2]>0
        posCount = np.sum(posRegion)
        if posCount != 0:
            bboxRemain = 0
            for bboxi in range(bbox.shape[0]):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                bboxRemain = bboxRemain+1
            bboxAve = float(posCount)/bboxRemain
            for bboxi in range(bbox.shape[0]):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                pixAve = bboxAve/overlapCount_bboxi
                weight = weight*(~overlap_bboxi) + pixAve*overlap_bboxi
        weight = weight[np.newaxis, ...]

        return image, vec, weight

    def loadic15(self, idx):
        # generate mask
        image = cv2.imread('{}/icdar2015/train_images/{}'.format(self.data_dir, idx), 1)
        height = image.shape[0]
        width = image.shape[1]
        mask = np.zeros(image.shape, dtype=np.uint8)
        bbox = ReadGt_ic15('{}/icdar2015/train_gts/'.format(self.data_dir), idx)
        for bboxi in range(len(bbox)):
            pts, hard = GetPts_ic15(bbox, bboxi)
            if hard == 1: continue
            cv2.fillPoly(mask,[pts],(255,255,255),1)

        # random crop parameters
        attempt = 0
        min_overlap = [0.1, 0.3, 0.5, 0.7]
        while attempt < 10:
            patch_h0, patch_w0, patch_h, patch_w = random_crop(mask)
            overlap = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w] > 0
            random_idx = np.random.randint(0, 4)
            if np.sum(overlap) >= np.sum(mask>0)*min_overlap[random_idx]:
                break
            attempt += 1
        if attempt == 10:
            patch_h = min(height, width)
            patch_w = patch_h
            patch_h0 = (height-patch_h) // 2
            patch_w0 = (width-patch_w) // 2

        # random crop & resize
        image = image[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)

        # random rotate
        prob = np.random.uniform(0,1)
        if prob <= 0.2:
            rtimes = 1
        elif prob >= 0.8:
            rtimes = 3
        else:
            rtimes = 0
        for rcount in range(rtimes):
            image = np.rot90(image)
        # cv2.imwrite('train_input/{}'.format(idx),image)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute vec
        accumulation = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        for bboxi in range(len(bbox)):
            pts, hard = GetPts_ic15(bbox, bboxi)
            seg = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(seg,[pts],(255,255,255),1)
            segPatch = seg[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
            for rcount in range(rtimes):
                segPatch = np.rot90(segPatch)
            if segPatch.max()!=255: continue
            segPatch = cv2.resize(segPatch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(segPatch,(0,0),(self.patch_size-1,self.patch_size-1),(0,0,0),1)

            img = cv2.cvtColor(segPatch,cv2.COLOR_BGR2GRAY)
            img = (img > 128).astype(np.uint8)
            dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
            index = np.copy(labels)
            index[img > 0] = 0
            place = np.argwhere(index > 0)
            nearCord = place[labels-1,:]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, self.patch_size, self.patch_size))
            nearPixel[0,:,:] = x
            nearPixel[1,:,:] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel
            dist = np.sqrt(np.sum(diff**2, axis = 0))

            direction = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
            direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
            if hard == 0:
                direction[2,img > 0] = bboxi+1

            accumulation[0,img > 0] = 0
            accumulation[1,img > 0] = 0
            accumulation[2,img > 0] = 0
            accumulation = accumulation + direction
        vec = np.stack((accumulation[0], accumulation[1]))

        # compute weight
        weight = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        posRegion = accumulation[2]>0
        posCount = np.sum(posRegion)
        if posCount != 0:
            bboxRemain = 0
            for bboxi in range(len(bbox)):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                bboxRemain = bboxRemain+1
            bboxAve = float(posCount)/bboxRemain
            for bboxi in range(len(bbox)):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                pixAve = bboxAve/overlapCount_bboxi
                weight = weight*(~overlap_bboxi) + pixAve*overlap_bboxi
        weight = weight[np.newaxis, ...]

        return image, vec, weight

    def loadtd(self, idx):
        # generate mask
        image = cv2.imread('{}/MSRA-TD500/train/text_image/{}'.format(self.data_dir, idx), 1)
        height = image.shape[0]
        width = image.shape[1]
        mask = np.zeros(image.shape, dtype=np.uint8)
        bbox = ReadGt_td('{}/MSRA-TD500/train/gt/'.format(self.data_dir), idx[:-4])
        for bboxi in range(len(bbox)):
            pts, hard = GetPts_td(bbox, bboxi)
            if hard == 1: continue
            cv2.fillPoly(mask,[pts],(255,255,255),1)

        # random crop parameters
        attempt = 0
        min_overlap = [0.1, 0.3, 0.5, 0.7]
        while attempt < 10:
            patch_h0, patch_w0, patch_h, patch_w = random_crop(mask)
            overlap = mask[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w] > 0
            random_idx = np.random.randint(0, 4)
            if np.sum(overlap) >= np.sum(mask>0)*min_overlap[random_idx]:
                break
            attempt += 1
        if attempt == 10:
            patch_h = min(height, width)
            patch_w = patch_h
            patch_h0 = (height-patch_h) // 2
            patch_w0 = (width-patch_w) // 2

        # random crop & resize
        image = image[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)

        # random rotate
        prob = np.random.uniform(0,1)
        if prob <= 0.2:
            rtimes = 1
        elif prob >= 0.8:
            rtimes = 3
        else:
            rtimes = 0
        for rcount in range(rtimes):
            image = np.rot90(image)
        # cv2.imwrite('train_input/{}'.format(idx),image)

        # normalization
        image = image.astype(np.float32)
        image -= self.mean
        image = image.transpose((2,0,1))

        # compute vec
        accumulation = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        for bboxi in range(len(bbox)):
            pts, hard = GetPts_td(bbox, bboxi)
            seg = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(seg,[pts],(255,255,255),1)
            segPatch = seg[patch_h0:patch_h0+patch_h, patch_w0:patch_w0+patch_w, :]
            for rcount in range(rtimes):
                segPatch = np.rot90(segPatch)
            if segPatch.max()!=255: continue
            segPatch = cv2.resize(segPatch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(segPatch,(0,0),(self.patch_size-1,self.patch_size-1),(0,0,0),1)

            img = cv2.cvtColor(segPatch,cv2.COLOR_BGR2GRAY)
            img = (img > 128).astype(np.uint8)
            dst, labels = cv2.distanceTransformWithLabels(img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
            index = np.copy(labels)
            index[img > 0] = 0
            place = np.argwhere(index > 0)
            nearCord = place[labels-1,:]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, self.patch_size, self.patch_size))
            nearPixel[0,:,:] = x
            nearPixel[1,:,:] = y
            grid = np.indices(img.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel
            dist = np.sqrt(np.sum(diff**2, axis = 0))

            direction = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            direction[0,img > 0] = np.divide(diff[0,img > 0], dist[img > 0])
            direction[1,img > 0] = np.divide(diff[1,img > 0], dist[img > 0])
            if hard == 0:
                direction[2,img > 0] = bboxi+1

            accumulation[0,img > 0] = 0
            accumulation[1,img > 0] = 0
            accumulation[2,img > 0] = 0
            accumulation = accumulation + direction
        vec = np.stack((accumulation[0], accumulation[1]))

        # compute weight
        weight = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        posRegion = accumulation[2]>0
        posCount = np.sum(posRegion)
        if posCount != 0:
            bboxRemain = 0
            for bboxi in range(len(bbox)):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                bboxRemain = bboxRemain+1
            bboxAve = float(posCount)/bboxRemain
            for bboxi in range(len(bbox)):
                overlap_bboxi = accumulation[2]==(bboxi+1)
                overlapCount_bboxi = np.sum(overlap_bboxi)
                if overlapCount_bboxi == 0: continue
                pixAve = bboxAve/overlapCount_bboxi
                weight = weight*(~overlap_bboxi) + pixAve*overlap_bboxi
        weight = weight[np.newaxis, ...]

        return image, vec, weight

class EuclideanLossLayerWithOHEM(caffe.Layer):

    def setup(self, bottom, top):
        # loss layer config
        params = eval(self.param_str)
        self.npRatio = int(params['npRatio'])
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # def dist and weight for backpropagation
        self.distL1 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.distL2 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightPos = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weightNeg = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # L1 and L2 distance
        self.distL1 = bottom[0].data - bottom[1].data
        self.distL2 = self.distL1**2
        # the amount of positive and negative pixels
        regionPos = (bottom[2].data>0)
        regionNeg = (bottom[2].data==0)
        sumPos = np.sum(regionPos)
        sumNeg = np.sum(regionNeg)
        # the amount of hard negative pixels
        sumhardNeg = min(self.npRatio*sumPos, sumNeg)
        # set loss on ~(top-sumhardNeg) negative pixels to 0
        lossNeg = (self.distL2[0][0]+self.distL2[0][1])*regionNeg
        lossFlat = lossNeg.flatten()
        lossFlat[np.argpartition(lossFlat,-sumhardNeg)[:-sumhardNeg]] = 0
        lossHard = lossFlat.reshape(lossNeg.shape)
        # weight for positive and negative pixels
        self.weightPos = np.concatenate((bottom[2].data,np.copy(bottom[2].data)), axis=1)
        self.weightNeg[0][0] = (lossHard!=0).astype(np.float32)#*sumPos/sumNeg
        self.weightNeg[0][1] = (lossHard!=0).astype(np.float32)#*sumPos/sumNeg
        # total loss
        top[0].data[...] = np.sum((self.distL1**2)*(self.weightPos + self.weightNeg)) / bottom[0].num / 2. / np.sum(self.weightPos + self.weightNeg)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.distL1*(self.weightPos + self.weightNeg) / bottom[0].num
        bottom[1].diff[...] = 0
        bottom[2].diff[...] = 0