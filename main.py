import random
import threading
from PIL import Image
import numpy as np
import copy

mutationrate = .05
populationsize = 20
threadnum = 10

class MRCM:
    def __init__(self, transforms):
        self.transforms = transforms
        #each transform is 6 parameters in a 2-element list: [2x2 np array for linear transform, 2x1 np array for offset]
    def __hash__(self):
        return hash(str(self))
    def __add__(self, other):
        return MRCM([[a[0]+b[0], a[1]+b[1]] for a in self.transforms for b in other.transforms])
    def __mul__(self, n):
        return MRCM([[a[0]*n, a[1]*n] for a in self.transforms])
    def __repr__(self):
        return str(self.transforms)
    def __str__(self):
        return str(self.transforms)
    def __eq__(self, other):
        return hash(self) == hash(other)

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

def gen_image(matrix, filename):
    #matrix = gen_matrix(mrcm, base)
    im = Image.new('1', tuple(matrix.shape)) #white canvas
    im.putdata(list((1-matrix).flat)) #draw everything
    im.save(filename)

def dist(point, m): #smallest pixel distance from a black pixel
    r = 0
    bestsquare = 2*m.shape[0]**2
    while r < bestsquare**.5: #check along a square border
        coords = []
        for i in range(r): #square border
            coords.append((point[0]+i, point[1]+r))
            coords.append((point[0]-i, point[1]+r))
            coords.append((point[0]+i, point[1]-r))
            coords.append((point[0]-i, point[1]-r))
            coords.append((point[0]+r, point[1]+i))
            coords.append((point[0]+r, point[1]-i))
            coords.append((point[0]-r, point[1]+i))
            coords.append((point[0]-r, point[1]-i))
        for coord in coords:
            try:
                if m[coord] == 1:
                    if ((coord[0]-point[0])**2 + (coord[1]-point[1])**2) < bestsquare:
                        bestsquare = (coord[0]-point[0])**2 + (coord[1]-point[1])**2
                    break
            except: #the coord is out of bounds
                pass
        r += 1
    return bestsquare**.5

def partialfitness(m1, m2):
    if sum(m1.flat) == 0:
        return 0
    fit = 0
    scale = m1.shape[0]*(2**.5)
    for x in range(m1.shape[0]):
        for y in range(m1.shape[1]):
            if m1[x,y] == 1:
                fit += 1 - dist([x,y], m2) / scale
    return fit / sum(m1.flat)

def precompute(m): #precomputed distances
    pre = np.zeros(m.shape)
    for x in range(pre.shape[0]):
        for y in range(pre.shape[1]):
            pre[x,y] = dist([x,y], m)
    return pre

def fitness(mrcm, targetm, tpre):
    image = gen_matrix(mrcm, np.ones(targetm.shape))
    fit1 = partialfitness(image, targetm)
    fit2 = sum((tpre*image).flat) / sum(targetm.flat)
    return min(fit1, fit2)

def random_mrcm(n):
    transforms = []
    for _ in range(n):
        transforms.append([1-2*np.random.rand(2,2), 1-2*np.random.rand(2,1)])
    return MRCM(transforms)

def loadtarget(filename):
    im = Image.open(filename)
    red = list(im.getdata(0)) #red channel
    blue = list(im.getdata(1)) #blue
    green = list(im.getdata(2)) #green
    redA = 255 - np.array(red, int).reshape(im.size) #convert to np array
    blueA = 255 - np.array(blue, int).reshape(im.size)
    greenA = 255 - np.array(green, int).reshape(im.size)
    imaverage = sum(((redA + blueA + greenA) // 3).flat) // (im.size[0]*im.size[1])
    return (redA + blueA + greenA) // (3*imaverage)

def pick(mrcmdict, n):
    v = np.array(list(mrcmdict.values()))
    shifted = v - min(v)
    probs = shifted / sum(shifted)
    return np.random.choice(list(mrcmdict.keys()), n, p=probs)

def crossover(mrcms):
    nex = MRCM([])
    for t in range(len(mrcms[0].transforms)):
        transform = []
        flattened = []
        for m in mrcms:
            flattened.append(list(m.transforms[t][0].flat) + list(m.transforms[t][1].flat))
        for i in range(6):
            choice = random.choice(flattened)
            transform.append(choice[i] + (random.random()-.5)*2*mutationrate)
        nex.transforms.append([np.array(transform[:4]).reshape([2,2]), np.array(transform[4:]).reshape([2,1])])
    return nex

def nexgen(mrcmdict):
    gen = []
    k = list(mrcmdict.keys())
    gen.append(max(k, key=lambda x: mrcmdict[x]))
    for _ in range(populationsize - 2):
        gen.append(crossover(pick(mrcmdict, 2)))
    gen.append(random_mrcm(len(gen[0].transforms)))
    return gen

def main(filename):
    bestgens = {}
    L = []
    t = loadtarget(filename)
    gen_image(t, 'real' + filename)
    for _ in range(populationsize):
        L.append(random_mrcm(5))
    i = 0
    prev = 0
    tpre = precompute(t)
    print('entering loop')
    while True:
        i += 1
        D = {}
        j = 0
        bestf = 0
        bestM = None
        for M in L:
            j += 1
            f = fitness(M, t, tpre)
            print('', j, f)
            D[M] = copy.copy(f)
            if f > bestf:
                bestf = f
                bestM = M
        bestgens[i] = copy.deepcopy(bestM)
        if bestf > prev:
            prev = bestf
            gen_image(gen_matrix(bestM, np.ones(t.shape)), 'gen' + str(i) + '.png')
        L = nexgen(D)
        print(i, max(D.values()), sum(D.values())/len(D.values()))
