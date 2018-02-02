import random
import threading
from PIL import Image
import numpy as np

mutationrate = .05
populationsize = 100
bitmapx = 256
bitmapy = 256
threadnum = 10

class MRCM:
    def __init__(self, transforms):
        self.transforms = transforms
        #each transform is 6 parameters in a 2-element list: [2x2 np array for linear transform, 2x1 np array for offset]
    def __hash__(self):
        return hash(tuple(self.transforms))
    def __add__(self, other):
        return MRCM([[a[0]+b[0], a[1]+b[1]] for a in self.transforms for b in other.transforms])
    def __mul__(self, n):
        return MRCM([[a[0]*n, a[1]*n] for a in self.transforms])
    def __repr__(self):
        return str(self.transforms)

def gen_sub(transform, size, old, ready, index, total):
    global INTERNAL
    for x in range(size*index//total, size*(index + 1)//total):
        for y in range(size):
            if old[x,y] == 1:
                newcoord = np.dot(transform[0], [[x],[y]]) + transform[1]*size
                newcoord = list((newcoord % size).flat) #mod by the size, cast to list
                INTERNAL[int(newcoord[0]),int(newcoord[1])] = 1 #cast to int + draw
    ready.set()
    return None

INTERNAL = None
def gen_step(mrcm, old):
    global INTERNAL
    INTERNAL = np.zeros(old.shape, int) #blank canvas
    size = len(old)
    readys = []
    for transform in mrcm.transforms:
        for index in range(threadnum): #divide and conquer
            ready = threading.Event()
            t = threading.Thread(target=gen_sub, args=(transform, size, old, ready, index, threadnum), daemon=True)
            readys.append(ready)
            t.start()
    for ready in readys: #wait for threads to finish
        ready.wait()
    return INTERNAL

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
    return sum((image*target + (1-image)*(1-target) - image*(1-target) - (1-image)*target).flat) / len(target.flat) #1 point per correct pixel, -1 points per incorrect pixel; divide by total to get proportion correct

def random_mrcm(n):
    transforms = []
    for _ in range(n):
        transforms.append([1-2*np.random.rand(2,2), 1-2*np.random.rand(1,2)])
    return MRCM(transforms)

def loadtarget(filename):
    im = Image.open(filename)
    raw = list(im.getdata(0))
    return 1 - np.array(raw, int).reshape(im.size) / 255
