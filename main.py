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
        #each transform is 6 parameters in a 2-element list: [2x2 np array for linear transform, 2x1 np array for offset]

def gen_step(mrcm, old):
    new = np.zeros(old.shape, int) #blank canvas
    size = len(old)
    for x in range(size):
        for y in range(size):
            if old[x,y] == 1: #need to draw in new
                for transform in mrcm.transforms:
                    newcoord = np.dot(transform[0], [[x],[y]]) + transform[1]*size
                    newcoord = list((newcoord % size).flat) #mod by the size, cast to list
                    new[int(newcoord[0]),int(newcoord[1])] = 1 #cast to int + draw
    return new

def gen_matrix(mrcm, base):
    old = np.copy(base)
    new = gen_step(mrcm, old)
    steps = 1
    while (not (old == new).all()) and steps < 20: #do a bunch of gen_step
        old = new
        new = gen_step(mrcm, old)
        steps += 1
    return new

def gen_image(mrcm, base, filename):
    matrix = gen_matrix(mrcm, base)
    im = Image.new('1', tuple(matrix.shape)) #white canvas
    im.putdata(list((1-matrix).flat)) #draw everything
    im.save(filename)

def fitness(mrcm, target):
    image = gen_matrix(mrcm, np.ones(target.shape))
    return sum((image*target - image*(1-target)).flat) / sum(target.flat) #1 point per correct pixel, -1 points per incorrect pixel; divide by total to get proportion correct

def random_mrcm(n):
    transforms = []
    for _ in range(n):
        transforms.append([np.random.rand(2,2), np.random.rand(1,2)])
    return MRCM(transforms)

def loadtarget(filename):
    im = Image.open(filename)
    raw = list(im.getdata(0))
    return 1 - np.array(raw, int).reshape(im.size) / 255
