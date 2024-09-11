"""
Calculation of the volumetric dice between 
outputs and actual magnetogram

It measures the superposition between 3D
representation of both files, with dimensions
[x,y,magnetic_levels], where values are one for 
each pixel if the magnetic field of that position 
is equal or lower in value to the range associated 
to that level, and of the same sign
"""

# Imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import h5py
import matplotlib.pyplot as plt

# Reading of HMI magnetograms
masks_HMI = []
for file in sorted(os.listdir('./masks/HMI/')):
    if file.endswith('HMI_9niveles.npy'):
        masks_HMI.append(file)

# Reading of SO magnetograms
masks_SO = []
for file in sorted(os.listdir('./masks/SO/')):
    if file.endswith('SO_9niveles.npy'):
        masks_SO.append(file)

# Reading of FarNet-II outputs
outputs = []
for file in sorted(os.listdir('./outputs/')):
    if file.endswith('.h5'):
        outputs.append(file)

# Selection of the item to represent from the list
i = int(input('Index for file selection (0 to 28) :'))

# Date of the volumetric dice to calculate
date = outputs[i][4:15]

# Loading of the magnetograms and outputs
HMI = np.load('./masks/HMI/'+masks_HMI[i])
SO = np.load('./masks/SO/'+masks_SO[i])
out = h5py.File('./outputs/'+outputs[i],'r')['magnetogram']

# The dice is calculated only for the central element of the firts
# sequence in each input, corresponding to the date in 'date'
out = out[0,5,:,:,:]

# Application of softmax to the output to interpret values
# as probabilities
out = out[np.newaxis,:,:,:]
b = torch.Tensor(out)
m = nn.Softmax2d()
b = m(b)
b = np.array(b)

# Production of 2D image
new_mag = np.zeros((144,120))
for x in range(len(new_mag[:,0])):
    for y in range(len(new_mag[0,:])):

        maximo = np.amax(b[0,:,x,y])
        lista = list(b[0,:,x,y])
        index = lista.index(maximo)

        # Inversion of polarities (outputs from one cycle after 
        # the one in which the network was trained)
        new_mag[x,y] = int((index-8)*-1)

# Volumetric dice function
def concent(true,logits,text):

    # Reading of the number of classes and zero level 
    num_classes = logits.shape[1]
    cero = int(num_classes/2)
    
    # Reading of the mask
    true = torch.LongTensor(true)

    # Vessels for the 3D representation of output and mask
    true_3D = np.zeros((9,true.shape[0],true.shape[1]))
    probas_3D = np.zeros((9,true.shape[0],true.shape[1]))

    # Selection of values 1 or 0, according to the volumetric dice criteria
    # for both output and mask
    for x in range(true.shape[0]):
        for y in range(true.shape[1]):

            if true[x,y] > cero:
                true_3D[cero:true[x,y]+1,x,y] = 1

            elif true[x,y] < cero:
                true_3D[true[x,y]:cero+1,x,y] = 1
            
            if new_mag[x,y] > cero:
                probas_3D[cero:int(new_mag[x,y])+1,x,y] = 1

            elif new_mag[x,y] < cero:
                probas_3D[int(new_mag[x,y]):cero+1,x,y] = 1
    
    # Graphic representation of volumetric shape of the mask
    zt, xt, yt = true_3D.nonzero()
    fig1 = plt.figure(figsize=(6,5))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.grid(False)
    ax1.set_title('a.')
    ax1.scatter(xt, yt, zt, c=zt, alpha=1)
    ax1.set_xlim3d(0,144)
    ax1.set_ylim3d(0,120)
    ax1.set_zlim3d(0,9)
    ax1.set_xlabel('Lat (deg)',labelpad=10)
    ax1.set_ylabel('Lon (deg)',labelpad=10)
    ax1.set_zticks([0,3,5,8])
    ax1.set_zticklabels([-4,-1,1,4])
    ax1.set_zlabel('Channel',labelpad=10)
    plt.savefig('./graphics/True'+date+text+'.pdf')

    # Graphic representation of volumetric shape of the output
    zp, xp, yp = probas_3D.nonzero()
    fig2 = plt.figure(figsize=(6,5))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_title('b.')
    ax2.scatter(xp, yp, zp, c=zp, alpha=1)
    ax2.grid(False)
    ax2.set_xlim3d(0,144)
    ax2.set_ylim3d(0,120)
    ax2.set_zlim3d(0,9)
    ax2.set_zticks([0,3,5,8])
    ax2.set_zticklabels([-4,-1,1,4])
    ax2.set_xlabel('Lat (deg)',labelpad=10)
    ax2.set_ylabel('Lon (deg)',labelpad=10)
    ax2.set_zlabel('Channel',labelpad=10)
    plt.savefig('./graphics/Probas'+date+text+'.pdf')
    
    # Smoothing value
    smooth = 0.001

    # Flattening of the output and the mask
    iflat = probas_3D.flatten()
    tflat = true_3D.flatten()

    # Intersection
    intersection = (iflat * tflat).sum()

    # 2D dice
    stuff = ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))

    # Return of the volumetric dice
    return stuff

# Calculation of the volumetric dices
Dice_HMI = concent(HMI,out,'HMI')
Dice_SO = concent(SO,out,'SO')

# Impresion of the volumetric dices
print(date,'SO',Dice_SO,'HMI',Dice_HMI)