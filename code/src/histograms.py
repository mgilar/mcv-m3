# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from configobj import ConfigObj
from math import floor
from tqdm import tqdm

HIST_NAMES = ["simple", "subimage", "pyramid", "pyramidFast"]

def accBackpropagationHistograms(keypoint_list, words, binNum, levels, h, w):
    def accBack(histsIndexed, level):
        hi = histsIndexed
        subLvlHists = dict()
        for x in range(0, level, 2):
            for y in range(0, level, 2):
                newHist = hi[(x,y)] + hi[(x,y+1)] + hi[(x+1,y)] + hi[(x+1,y+1)]
                subLvlHists[(int(x/2), int(y/2))] = newHist
        return subLvlHists
    
    def getOrderedHist(subHistsDict, lvl):
        orderedHists = list()
        for x in range(lvl):
            for y in range(lvl):
                orderedHists.append(subHistsDict[(x,y)])
        return orderedHists
        
#    w, h, _ = image.shape
    imageHist = list()
    hasFinished = False
    level = levels
    subHists = subImageHistogramsIndexed(keypoint_list, words, binNum, 2**levels, h, w)
    
    while(not hasFinished):
        orderedHistList = getOrderedHist(subHists, 2**level)
        imageHist.append(orderedHistList)
        subHists = accBack(subHists, 2**level)
        level-=1
        hasFinished = (len(subHists) == 1)
    imageHist = [subHists[(0,0)]] + imageHist
    return imageHist[::-1]
        
            
def subImageHistogramsIndexed(keypoints, words, binNum, divisions, h, w):
    # print(w,h)
    imageHist = dict()
    divisor_w = w/divisions
    divisor_h = h/divisions
    for word, kpt in zip(words,keypoints):
        (x,y) = kpt.pt
        i = floor(x/divisor_w)
        j = floor(h/divisor_h)
        if(not imageHist.haskey((i,j))):
            imageHist[(i, j)] = np.zeros(binNum)
        imageHist[(i, j)][word]+=1

    # for i in range(divisions):
        # x1 = floor(i*(w/divisions))
        # x2 = floor((i+1)*(w/divisions))
        # for j in range(divisions):
            # y1 = floor(j*(h/divisions))
            # y2 = floor((j+1)*(h/divisions))
            
#            insiders = [x if x.x > 10 and x.x < 100 and x.y < 10 and x.y > 100 for x in desc]
            # subImage = image[x1:x2,y1:y2,:]
            # hist = generateHistogram(subImage, binNum, colorSpace=colorSpace)

    return imageHist