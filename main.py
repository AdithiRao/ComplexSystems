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

def fitness(mrcm, targetm):
    image = gen_matrix(mrcm, np.ones(targetm.shape))
    truths = 1-np.bitwise_xor(image, targetm)
    return sum(truths.flat) / len(targetm.flat) #1 point per correct pixel, divide by total to get proportion correct

def random_mrcm(n):
    transforms = []
    for _ in range(n):
        transforms.append([1-2*np.random.rand(2,2), 1-2*np.random.rand(2,1)])
    return MRCM(transforms)

def loadtarget(filename):
    im = Image.open(filename)
    raw = list(im.getdata(0))
    return 1 - np.array(raw, int).reshape(im.size) // 255

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

bestgens = {}
L = []
t = loadtarget('IMAGENAME.png')
for _ in range(populationsize):
    L.append(random_mrcm(3))
i = 0
while True:
    i += 1
    D = {}
    j = 0
    bestf = 0
    bestM = None
    for M in L:
        j += 1
        f = fitness(M, t)
        D[M] = copy.copy(f)
        if f > bestf:
            bestf = f
            bestM = M
    bestgens[i] = copy.deepcopy(bestM)
    gen_image(gen_matrix(bestM, np.ones(t.shape)), 'gen' + str(i) + '.png')
    L = nexgen(D)
    print(i, max(D.values()), sum(D.values())/len(D.values()))
