#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:23:00 2018

@author: ll303
"""

import h5py
import seaborn as sns
import numpy as np
import pandas as pd

def readData(data):

    hybridSpikeMat = data['hybrid/spikeMatrix']
    rateSpikeMat = data['rate/spikeMatrix']
    ampSpikeMat = data['amp/spikeMatrix']

    hybridQuanta = data['hybrid/quanta']
    rateQuanta = data['rate/quanta']
    ampQuanta = data['amp/quanta']

    hybridFuckups = data['hybrid/fuckups']
    rateFuckups = data['rate/fuckups']
    ampFuckups = data['amp/fuckups']

    return hybridSpikeMat, rateSpikeMat, ampSpikeMat, hybridQuanta, rateQuanta, ampQuanta, hybridFuckups, rateFuckups, ampFuckups

#find out how to make proper dataframes
    
def getData(vRest, gLeak, vDev):
    name = 'data/modelData'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    data = h5py.File(name, 'r')
    hybridSpikeMat, rateSpikeMat, ampSpikeMat, hybridQuanta, rateQuanta, ampQuanta, hybridFuckups, rateFuckups, ampFuckups = readData(data)
    nS, nGen = hybridQuanta.shape
    means = np.zeros((nS, 3))
    variances = np.zeros((nS, 3))
    gData = np.zeros((nS*nGen*3, 3))
    gVar = np.zeros((3*nS, 3))
    gVxM = np.zeros((3*nS, 3))
    gFuckups = np.zeros((3*nS, 3))

    gSpikes = np.zeros((3*nS, 3))
    for i in range(nS):
        means[i, 0] = np.mean(hybridQuanta[i])
        means[i, 1] = np.mean(rateQuanta[i])
        means[i, 2] = np.mean(ampQuanta[i])

        variances[i, 0] = np.var(hybridQuanta[i])
        variances[i, 1] = np.var(rateQuanta[i])
        variances[i, 2] = np.var(ampQuanta[i])

        gVar[i, 0] = i
        gVar[nS+i, 0] = i
        gVar[nS*2+i, 0] = i
        gVar[i, 1] = variances[(i, 0)]
        gVar[nS+i, 1] = variances[(i, 1)]
        gVar[nS*2+i, 1] = variances[(i, 2)]
        gVar[i, 2] = 0
        gVar[nS+i, 2] = 1
        gVar[2*nS+i, 2] = 2

        gVxM[i, 0] = means[(i, 0)]
        gVxM[nS+i, 0] = means[(i, 1)]
        gVxM[nS*2+i, 0] = means[(i, 2)]
        gVxM[i, 1] = variances[(i, 0)]
        gVxM[nS+i, 1] = variances[(i, 1)]
        gVxM[nS*2+i, 1] = variances[(i, 2)]
        gVxM[i, 2] = 0
        gVxM[nS+i, 2] = 1
        gVxM[2*nS+i, 2] = 2

        gFuckups[i, 0] = i
        gFuckups[nS+i, 0] = i
        gFuckups[nS*2+i, 0] = i
        gFuckups[i, 1] = hybridFuckups[i]
        gFuckups[nS+i, 1] = rateFuckups[i]
        gFuckups[nS*2+i, 1] = ampFuckups[i]
        gFuckups[i, 2] = 0
        gFuckups[nS+i, 2] = 1
        gFuckups[2*nS+i, 2] = 2

        gSpikes[i, 0] = i
        gSpikes[nS+i, 0] = i
        gSpikes[2*nS+i, 0] = i
        gSpikes[i, 2] = 0
        gSpikes[nS+i, 2] = 1
        gSpikes[2*nS+i, 2] = 2

        for j in range(nGen):   
            gData[i*nGen+j,0] = i
            gData[i*nGen+j,1] = hybridQuanta[(i, j)]
            gData[i*nGen+j,2] = 0
            gData[nS*nGen+i*nGen+j,0] = i
            gData[nS*nGen+i*nGen+j,1] = rateQuanta[(i, j)]
            gData[nS*nGen+i*nGen+j,2] = 1
            gData[2*nS*nGen+i*nGen+j,0] = i
            gData[2*nS*nGen+i*nGen+j,1] = ampQuanta[(i, j)]
            gData[2*nS*nGen+i*nGen+j,2] = 2
            gSpikes[i, 1] += hybridSpikeMat[i, j].size/nGen
            gSpikes[nS+i, 1] += rateSpikeMat[i, j].size/nGen
            gSpikes[2*nS+i, 1] += ampSpikeMat[i, j].size/nGen
    """
    for i in range(20):
        gData[i+20][0] = i
        gData[i+20][1] = np.mean(rateQuanta[i])
        gData[i+20][2] = 1
    for i in range(20):
        gData[i+40][0] = i
        gData[i+40][1] = np.mean(ampQuanta[i])
        gData[i+40][2] = 2"""
    data.close()
    return gData, gVar, gVxM, gFuckups, gSpikes,means

def plotData(gData, gVar, gVxM, gFuckups, gSpikes):
    # avg. quanta per contrast
    df = pd.DataFrame(data=np.array(gData), columns=['contrast', 'quanta', 'input'])
    sns.set()
    sns.relplot(x="contrast", y="quanta", style="input", kind="line", data=df)

    # quanta variance per contrast
    df = pd.DataFrame(data=np.array(gVar), columns=['contrast', 'variance', 'input'])
    sns.set()
    sns.relplot(x="contrast", y="variance", style="input", kind="line", data=df)

    #quanta variance vs mean
    df = pd.DataFrame(data=np.array(gVxM), columns=['mean', 'variance', 'input'])
    sns.set()
    sns.relplot(x="mean", y="variance", style="input", kind="line", data=df)

    #total fuckups per contrast
    df = pd.DataFrame(data=np.array(gFuckups), columns=['contrast', 'fuckups(total)', 'input'])
    sns.set()
    sns.relplot(x="contrast", y="fuckups(total)", style="input", kind="line", data=df)

    #average number of spikes per contrast
    df = pd.DataFrame(data=np.array(gSpikes), columns=['contrast', 'number of spikes', 'input'])
    sns.set()
    sns.relplot(x="contrast", y="number of spikes", style="input", kind="line", data=df)
    
    #ave number spikes per ave quanta

    
    
        
def graphData(vRest, gLeak, vDev):
    gData, gVar, gVxM, gFuckups, gSpikes,means = getData(vRest, gLeak, vDev)
    plotData(gData, gVar, gVxM, gFuckups, gSpikes)
    return gSpikes, gData

"""data = h5py.File('samples/samples_0.25_50.0_20.0.hdf5', 'r')

hybridV = data['hybrid/voltage']
hybridC = data['hybrid/current']
rateV = data['rate/voltage']
rateC = data['rate/current']
ampV = data['amp/voltage']
ampC = data['amp/current']

nT = hybridV.size
mData = np.zeros((6*nT, 4))

for i in range(0, nT):
    mData[i, 0] = i
    mData[nT+i, 0] = i
    mData[2*nT+i, 0] = i
    mData[3*nT+i, 0] = i
    mData[4*nT+i, 0] = i
    mData[5*nT+i, 0] = i
    
    mData[i, 1] = hybridV[i]
    mData[nT+i, 1] = rateV[i]
    mData[2*nT+i, 1] = ampV[i]
    mData[3*nT+i, 1] = hybridC[i]
    mData[4*nT+i, 1] = rateC[i]
    mData[5*nT+i, 1] = ampC[i]
    
    mData[i, 2] = 0
    mData[nT+i, 2] = 0
    mData[2*nT+i, 2] = 0
    mData[3*nT+i, 2] = 1
    mData[4*nT+i, 2] = 1
    mData[5*nT+i, 2] = 1
    
    mData[i, 3] = 0
    mData[nT+i, 3] = 1
    mData[2*nT+i, 3] = 2
    mData[3*nT+i, 3] = 0
    mData[4*nT+i, 3] = 1
    mData[5*nT+i, 3] = 2
    
data.close()

#voltage and currrent
df = pd.DataFrame(data=np.array(mData), columns=['time', 'value', 'type', 'input'])
sns.set()
sns.relplot(x="time", y="value", col="input", row='type', kind="line", data=df)"""