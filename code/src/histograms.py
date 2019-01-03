# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from configobj import ConfigObj
from math import floor
from tqdm import tqdm


HIST_NAMES = ["simple", "subimage", "pyramid", "pyramidFast"]

def subImageHistogramsIndexed(keypoints, words, binNum, divisions, h, w):
    # print(w,h)
    imageHist = dict()
    for i in range(divisions):
        for j in range(divisions):
            imageHist[(i, j)] = np.zeros((1,binNum))
    divisor_w = w/divisions
    divisor_h = h/divisions
    for word, kpt in zip(words,keypoints):
        (x,y) = kpt.pt
        i = floor(x/divisor_w)
        j = floor(y/divisor_h)
        # if(not (i,j) in imageHist): # se puede convertir imageHist en una lista
            # imageHist[(i, j)] = np.zeros(binNum)
        imageHist[(i, j)][0,word]+=1


    return imageHist

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

    imageHist =  [np.concatenate(lh, axis=1) for lh in imageHist]   
    imageHist = imageHist + [subHists[(0,0)]] 

    return imageHist[::-1]
        
            