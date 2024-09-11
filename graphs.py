"""
Production of images comparing outputs with 
magnetograms from HMI and SO
"""

# Imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

# Parameters for graphics
plt.rcParams.update({'font.size': 14})

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

# Date of the graphics data
date = outputs[i][4:15]

# Loading of the magnetograms and outputs
HMI = np.load('./masks/HMI/'+masks_HMI[i])
SO = np.load('./masks/SO/'+masks_SO[i])
out = h5py.File('./outputs/'+outputs[i],'r')['magnetogram']

# The image is produced only for the central element of the firts
# sequence in each input, corresponding to the date in 'date'
b = out[0,5,:,:,:]

# Application of softmax to the output to interpret values
# as probabilities
b = b[np.newaxis,:,:,:]
b = torch.Tensor(b)
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

# Figure creation
fig, ax= plt.subplots(1, 3, figsize=(20,6))
fig.suptitle(date)
a = ax[0].imshow(HMI,origin='lower')
ax[0].title.set_text('HMI')
b = ax[1].imshow(SO,origin='lower')
ax[1].title.set_text('SO')
c = ax[2].imshow(new_mag,origin='lower')
ax[2].title.set_text('FarNet-II')
plt.colorbar(c,ax=ax,fraction=0.02, pad=0.04)
plt.savefig('./graphics/graph_'+date+'.pdf')
#plt.show()


