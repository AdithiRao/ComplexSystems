import random
from PIL import Image
import numpy as np

mutationrate = .05
populationsize = 100
bitmapx = 256
bitmapy = 256

class MRCM:
    def __init__(self, transforms):
        self.transforms = transforms
        #each transform is 6 parameters: [2x2 np array for linear transform, 2x1 np array for offset]

def gen_step(mrcm, old):
    new = np.zeros(old.shape, int)
    for transform in mrcm.transforms:
        for x in range(len(old)):
            for y in range(len(old)):
                if old[x,y] == 1: #draw in new
                    newcoord = np.dot(transform[0], np.array([[x],[y]])) + transform[1]*len(old) #tranform
                    newcoord = list((np.array(newcoord, int) % len(old)).flat) #cast to int and mod by the size
                    new[newcoord[0],newcoord[1]] = 1 #draw
    return new

def gen_image(mrcm, base):
    old = np.copy(base)
    new = gen_step(mrcm, old)
    steps = 1
    while (not (old == new).all()) and steps < 100: #do a bunch of gen_step
        old = new
        new = gen_step(mrcm, old)
        steps += 1
    return new

def fitness(mrcm, target):
    image = gen_image(mrcm, np.ones(target.shape))
    return sum((image*target - image*(1-target)).flat) / sum(target.flat) #1 point per correct pixel, -1 points per incorrect pixel; divide by total to get proportion correct
