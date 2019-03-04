#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:23:00 2018

@author: ll303
"""

import csv
import h5py
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def readData(data):
    metadata = data['metadata']
    
    hybridSpikeMat = data['hybrid/spikeMatrix']
    rateSpikeMat = data['rate/spikeMatrix']
    ampSpikeMat = data['amp/spikeMatrix']

    hybridQuanta = data['hybrid/quanta']
    rateQuanta = data['rate/quanta']
    ampQuanta = data['amp/quanta']

    hybridFuckups = data['hybrid/fuckups']
    rateFuckups = data['rate/fuckups']
    ampFuckups = data['amp/fuckups']
    
    hybridVs = data['hybrid/meanV']
    rateVs = data['rate/meanV']
    ampVs = data['amp/meanV']
    
    hybridFree = data['hybrid/freeV']
    rateFree = data['rate/freeV']
    ampFree = data['amp/freeV']
    
    hybridCumQuanta = data['hybrid/cumQuanta']
    rateCumQuanta = data['rate/cumQuanta']
    ampCumQuanta = data['amp/cumQuanta']
    
    hybridVarQuanta = data['hybrid/varQuanta']
    rateVarQuanta = data['rate/varQuanta']
    ampVarQuanta = data['amp/varQuanta']
    
    hybridPhaseVar = data['hybrid/phaseVar']
    ratePhaseVar = data['rate/phaseVar']
    ampPhaseVar = data['amp/phaseVar']
    
    hybridLambda = data['hybrid/lambda']
    rateLambda = data['rate/lambda']
    ampLambda = data['amp/lambda']

    hybridExVar = data['hybrid/expPhaseVar']
    rateExVar = data['rate/expPhaseVar']
    ampExVar = data['amp/expPhaseVar']
    
    hybridFirst = data['hybrid/firstSpikes']
    rateFirst = data['rate/firstSpikes']
    ampFirst = data['amp/firstSpikes']
    
    print(metadata.value)
    
    return hybridSpikeMat, rateSpikeMat, ampSpikeMat, hybridQuanta, rateQuanta, ampQuanta, hybridFuckups, rateFuckups, ampFuckups, hybridVs, rateVs, ampVs, hybridFree, rateFree, ampFree, hybridCumQuanta, rateCumQuanta, ampCumQuanta, hybridVarQuanta, rateVarQuanta, ampVarQuanta, hybridPhaseVar, ratePhaseVar, ampPhaseVar, hybridLambda, rateLambda, ampLambda, hybridExVar, rateExVar, ampExVar, hybridFirst, rateFirst, ampFirst

def getSamples(data):
    
    hybridSample = data['hybrid/Vs']
    rateSample = data['rate/Vs']
    ampSample = data['amp/Vs']
    
    hybridNTSample = data['hybrid/NTVs']
    rateNTSample = data['rate/NTVs']
    ampNTSample = data['amp/NTVs']
    
    return hybridSample, rateSample, ampSample, hybridNTSample, rateNTSample, ampNTSample
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
    gFuckups = np.zeros((3*nS, 3))

    gSpikes = np.zeros((3*nS, 3))
    for i in range(nS):
        means[i, 0] = np.mean(hybridQuanta[i])
        means[i, 1] = np.mean(rateQuanta[i])
        means[i, 2] = np.mean(ampQuanta[i])

        variances[i, 0] = np.var(hybridQuanta[i])
        variances[i, 1] = np.var(rateQuanta[i])
        variances[i, 2] = np.var(ampQuanta[i])

        gVar[i, 0] = means[i, 0]
        gVar[nS+i, 0] = means[i, 1]
        gVar[nS*2+i, 0] = means[i, 2]
        gVar[i, 1] = variances[(i, 0)]
        gVar[nS+i, 1] = variances[(i, 1)]
        gVar[nS*2+i, 1] = variances[(i, 2)]
        gVar[i, 2] = 0
        gVar[nS+i, 2] = 1
        gVar[2*nS+i, 2] = 2


        gFuckups[i, 0] = means[i, 0]
        gFuckups[nS+i, 0] = means[i, 1]
        gFuckups[nS*2+i, 0] = means[i, 2]
        gFuckups[i, 1] = hybridFuckups[i]
        gFuckups[nS+i, 1] = rateFuckups[i]
        gFuckups[nS*2+i, 1] = ampFuckups[i]
        gFuckups[i, 2] = 0
        gFuckups[nS+i, 2] = 1
        gFuckups[2*nS+i, 2] = 2

        gSpikes[i, 0] = means[i, 0]
        gSpikes[nS+i, 0] = means[i, 1]
        gSpikes[2*nS+i, 0] = means[i, 2]
        gSpikes[i, 2] = 0
        gSpikes[nS+i, 2] = 1
        gSpikes[2*nS+i, 2] = 2

        for j in range(nGen):   
            gData[i*nGen+j,0] = means[i, 0]
            gData[i*nGen+j,1] = hybridQuanta[(i, j)]
            gData[i*nGen+j,2] = 0
            gData[nS*nGen+i*nGen+j,0] = means[i, 1]
            gData[nS*nGen+i*nGen+j,1] = rateQuanta[(i, j)]
            gData[nS*nGen+i*nGen+j,2] = 1
            gData[2*nS*nGen+i*nGen+j,0] = means[i, 2]
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
    return gData, gVar, gVxM, gFuckups, gSpikes

def plotData(gData, gVar, gVxM, gFuckups, gSpikes):
    # avg. quanta per contrast
    df = pd.DataFrame(data=np.array(gData), columns=['contrast', 'quanta', 'input'])
    sns.set()
    sns.relplot(x="contrast", y="quanta", style="input", kind="line", data=df)

    # quanta variance per contrast
    df = pd.DataFrame(data=np.array(gVar), columns=['quanta rate', 'variance', 'input'])
    sns.set()
    sns.relplot(x="quanta rate", y="variance", style="input", kind="line", data=df)

    #total fuckups per contrast
    df = pd.DataFrame(data=np.array(gFuckups), columns=['quanta rate', 'fuckups(total)', 'input'])
    sns.set()
    sns.relplot(x="quanta rate", y="fuckups(total)", style="input", kind="line", data=df)

    #average number of spikes per contrast
    df = pd.DataFrame(data=np.array(gSpikes), columns=['quanta rate', 'number of spikes', 'input'])
    sns.set()
    sns.relplot(x="quanta rate", y="number of spikes", style="input", kind="line", data=df)
    
def graphData(vRest, gLeak, vDev):
    gData, gVar, gVxM, gFuckups, gSpikes = getData(vRest, gLeak, vDev)
    plotData(gData, gVar, gVxM, gFuckups, gSpikes)
    

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

def getPhase(phase, evTime, dt, nP): # - getPhase[t] / getPhase[iT*dt]
    return phase[int(evTime/dt%nP)]

def vectorStrength(spikeMat, f):
    VS=[]
    for contrast in spikeMat:
        cos = 0
        sin = 0
        n=0
        for gen in contrast:
            for spike in gen:
                cos += np.cos(2*np.pi*f*(spike%(1/f)))
                sin += np.sin(2*np.pi*f*(spike%(1/f)))
                n+=1
                 
        cos = cos**2
        sin = sin**2
        VS.append(np.sqrt(cos+sin)/n)
    
    return VS

def getVS(vRest, gLeak, vDev, f):
    name = 'data/modelData'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    data = h5py.File(name, 'r')
    hybridSpikeMat, rateSpikeMat, ampSpikeMat, hybridQuanta, rateQuanta, ampQuanta, hybridFuckups, rateFuckups, ampFuckups = readData(data)
    hVS = vectorStrength(hybridSpikeMat, f)
    rVS = vectorStrength(rateSpikeMat, f)
    aVS = vectorStrength(ampSpikeMat, f)
    
    nS, nGen = hybridQuanta.shape
    means = np.zeros((nS, 3))
    for i in range(nS):
        means[i, 0] = np.mean(hybridQuanta[i])
        means[i, 1] = np.mean(rateQuanta[i])
        means[i, 2] = np.mean(ampQuanta[i])
        
    data.close()
    return hVS, rVS, aVS, means

def graphVS(vRest, gLeak, vDev, f):
    hVS, rVS, aVS, means = getVS(vRest, gLeak, vDev, f)
    nS = len(hVS)
    gData = np.zeros((3*nS, 3))
    for i in range(nS):
        gData[i, 0] = means[i, 0]
        gData[i+nS, 0] = means[i, 1]
        gData[i+2*nS, 0] = means[i, 2]
        
        gData[i, 1] = hVS[i]
        gData[i+nS, 1] = rVS[i]
        gData[i+2*nS, 1] = aVS[i]
        
        gData[i, 2] = 0
        gData[i+nS, 2] = 1
        gData[i+2*nS, 2] = 2
        
    df = pd.DataFrame(data=np.array(gData), columns=['quanta rate', 'vector strength', 'input'])
    sns.set()
    sns.relplot(x="quanta rate", y="vector strength", style="input", kind="line", data=df)
    
def getPhases(spikeMat, f, phase, dt, nP):
    PDMat = []
    for contrast in spikeMat:
        PD = []
        for gen in contrast:
            for spike in gen:
                PD.append(getPhase(phase, spike, dt, nP))
        PDMat.append(PD)
    return PDMat
    
def phaseDist(vRest, gLeak, vDev, f, phase, dt, nP):
    name = 'data/modelData'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    data = h5py.File(name, 'r')
    hybridSpikeMat, rateSpikeMat, ampSpikeMat, hybridQuanta, rateQuanta, ampQuanta, hybridFuckups, rateFuckups, ampFuckups = readData(data)
    hPD = getPhases(hybridSpikeMat, f, phase, dt, nP)
    rPD = getPhases(rateSpikeMat, f, phase, dt, nP)
    aPD = getPhases(ampSpikeMat, f, phase, dt, nP)
    
    return hPD, rPD, aPD

def getFirstSpikes(vRest, gLeak, vDev, f, recLen, dt):
    
    name = 'data/modelData'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    data = h5py.File(name, 'r')
    hybridSpikeMat, rateSpikeMat, ampSpikeMat, hybridQuanta, rateQuanta, ampQuanta, hybridFuckups, rateFuckups, ampFuckups, hybridVs, rateVs, ampVs, hybridFree, rateFree, ampFree, hybridCumQuanta, rateCumQuanta, ampCumQuanta, hybridVarQuanta, rateVarQuanta, ampVarQuanta, hybridPhaseVar, ratePhaseVar, ampPhaseVar = readData(data)
    nS, nGen = hybridQuanta.shape
    
    t = np.arange(0,recLen,dt)
    lambdaI = np.cos(2* np.pi * f * t)

    firstSpikesA = []
    firstSpikesR = []
    
    k=0
    l=0
    gData = np.zeros((nGen)) 
    gData2 = np.zeros((nGen)) 
    for i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        
        for j in range(nGen):
            
            if (ampSpikeMat[(i ,j)].size>0):
                gData[k] = ampSpikeMat[(i, j)][0]
                k+=1
                
            if (rateSpikeMat[(i ,j)].size>0):
                gData2[l] = rateSpikeMat[(i, j)][0]
                l+=1
                
                
    
    df = pd.DataFrame(data=np.array(gData), columns=['amp'])
    df2 = pd.DataFrame(data=np.array(gData2), columns=['rate'])
    
    '''sns.distplot(firstSpikesA[0], hist=False, rug=False, label='A code 1')
    sns.distplot(firstSpikesA[1], hist=False, rug=False, label='A code 2')
    sns.distplot(firstSpikesA[2], hist=False, rug=False, label='A code 3')
    sns.distplot(firstSpikesA[3], hist=False, rug=False, label='A code 4')'''
    
    '''g = sns.FacetGrid(df, hue="input")
    g.map(sns.distplot, "time", hist=False, rug=False)
    g.add_legend()'''
    
    df.to_csv('dist.csv', encoding='utf-8', index=False)
    df2.to_csv('dist2.csv', encoding='utf-8', index=False)
    
    #sns.distplot(firstSpikesR, hist=False, rug=False, label='R code')
    #plt.legend()
    
def plotVs(vRest, gLeak, vDev, contrastN, f, dt):
    name = 'data/modelData'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    data = h5py.File(name, 'r')
    hybridSpikeMat, rateSpikeMat, ampSpikeMat, hybridQuanta, rateQuanta, ampQuanta, hybridFuckups, rateFuckups, ampFuckups, hybridVs, rateVs, ampVs, hybridFree, rateFree, ampFree, hybridCumQuanta, rateCumQuanta, ampCumQuanta, hybridVarQuanta, rateVarQuanta, ampVarQuanta, hybridPhaseVar, ratePhaseVar, ampPhaseVar = readData(data)

    nS, nT = hybridVs.shape
    
    gData = np.zeros((9*nT, 4)) #nT, value, input, datatype
                                #inputs: h, r, a
                                #data: v, no thresh, lambda
    for i in range(0, nT):
        gData[i, 0] = i*dt
        gData[nT+i, 0] = i*dt
        gData[2*nT+i, 0] = i*dt
        gData[3*nT+i, 0] = i*dt
        gData[4*nT+i, 0] = i*dt
        gData[5*nT+i, 0] = i*dt
        gData[6*nT+i, 0] = i*dt
        gData[7*nT+i, 0] = i*dt
        gData[8*nT+i, 0] = i*dt
        
        gData[i, 1] = hybridVs[contrastN, i]
        gData[nT+i, 1] = hybridFree[contrastN, i]
        gData[2*nT+i, 1] = hybridCumQuanta[i, contrastN]
        gData[3*nT+i, 1] = rateVs[contrastN, i]
        gData[4*nT+i, 1] = rateFree[contrastN, i]
        gData[5*nT+i, 1] = rateCumQuanta[i, contrastN]
        gData[6*nT+i, 1] = ampVs[contrastN, i]
        gData[7*nT+i, 1] = ampFree[contrastN, i]
        gData[8*nT+i, 1] = ampCumQuanta[i, contrastN]
        
        gData[i, 2] = 0
        gData[nT+i, 2] = 0
        gData[2*nT+i, 2] = 0
        gData[3*nT+i, 2] = 1
        gData[4*nT+i, 2] = 1
        gData[5*nT+i, 2] = 1
        gData[6*nT+i, 2] = 2
        gData[7*nT+i, 2] = 2
        gData[8*nT+i, 2] = 2
        
        gData[i, 3] = 0
        gData[nT+i, 3] = 1
        gData[2*nT+i, 3] = 2
        gData[3*nT+i, 3] = 0
        gData[4*nT+i, 3] = 1
        gData[5*nT+i, 3] = 2
        gData[6*nT+i, 3] = 0
        gData[7*nT+i, 3] = 1
        gData[8*nT+i, 3] = 2
        
    df = pd.DataFrame(data=np.array(gData), columns=['time', 'value', 'input', 'data'])
    df["input"] = df["input"].map({0: "Hybrid Code", 1: "Rate Code", 2: "Amplitude Code"})
    with sns.axes_style("white"):
        g = sns.FacetGrid(df, row="data", col="input", margin_titles=True, height=5, sharey='row', xlim=(0, 1))
    g.map(plt.plot, "time", "value", color="#334488");
    g.axes[0,0].set_ylabel('Mean Voltage')
    g.axes[1,0].set_ylabel('Mean Free Voltage')
    g.axes[2,0].set_ylabel('Cumulative Vesicle Release')
    g.set_titles(col_template = '{col_name}')
    for row in g.axes:
        row[-1].texts = []
    g.fig.subplots_adjust(wspace=.2, hspace=.02);
    
def getInstFreq(spikeMat, vRest, gLeak, vDev, contrastN, recLen, f, dt, nGen, nT):
    freqs = np.zeros((nT))
    for j in range(nGen):
        n = len(spikeMat[contrastN, j])
        if n>=1:
            nextSpike = spikeMat[contrastN, j][0]
            spikes = 1
            freq = 1/nextSpike
            for i in range(0, int(recLen/dt)):
                if (i*dt<=nextSpike or nextSpike==-1):
                    freqs[i] += freq
                elif(nextSpike == spikeMat[contrastN, j][-1]):
                    freq = 1/(recLen-nextSpike)
                    nextSpike=-1
                    freqs[i] += freq
                else:
                    freq = 1/(spikeMat[contrastN, j][spikes]-nextSpike)
                    nextSpike = spikeMat[contrastN, j][spikes]
                    spikes+=1
                    freqs[i] += freq
                   
        
    freqs/=nGen
    
    return freqs

def plotInstFreq(vRest, gLeak, vDev, contrastN, recLen, f, dt):
    name = 'data/modelData'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    data = h5py.File(name, 'r')
    hybridSpikeMat, rateSpikeMat, ampSpikeMat, hybridQuanta, rateQuanta, ampQuanta, hybridFuckups, rateFuckups, ampFuckups, hybridVs, rateVs, ampVs, hybridFree, rateFree, ampFree, hybridCumQuanta, rateCumQuanta, ampCumQuanta, hybridVarQuanta, rateVarQuanta, ampVarQuanta, hybridPhaseVar, ratePhaseVar, ampPhaseVar = readData(data)
    
    nS, nT = hybridVs.shape
    nS, nGen = hybridQuanta.shape
    
    hInst = getInstFreq(hybridSpikeMat, vRest, gLeak, vDev, contrastN, recLen, f, dt, nGen, nT)
    rInst = getInstFreq(rateSpikeMat, vRest, gLeak, vDev, contrastN, recLen, f, dt, nGen, nT)
    aInst = getInstFreq(ampSpikeMat, vRest, gLeak, vDev, contrastN, recLen, f, dt, nGen, nT)
    
    gData = np.zeros((3*nT, 3))
    
    for i in range(0, nT):
        gData[i, 0] = i*dt
        gData[nT+i, 0] = i*dt
        gData[2*nT+i, 0] = i*dt
        
        gData[i, 1] = hInst[i]
        gData[nT+i, 1] = rInst[i]
        gData[2*nT+i, 1] = aInst[i]
        
        gData[i, 2] = 0
        gData[nT+i, 2] = 1
        gData[2*nT+i, 2] = 2
        
    df = pd.DataFrame(data=np.array(gData), columns=['time', 'freq', 'input'])
    df["input"] = df["input"].map({0: "Hybrid Code", 1: "Rate Code", 2: "Amplitude Code"})
    with sns.axes_style("white"):
        g = sns.FacetGrid(df, col="input", margin_titles=True, height=5, sharey='row', xlim=(0, recLen))
    g.map(plt.plot, "time", "freq", color="#334488");
    g.axes[0,0].set_ylabel('Inst. Frequency')
    g.set_titles(col_template = '{col_name}')
    for row in g.axes:
        row[-1].texts = []
    g.fig.subplots_adjust(wspace=.2, hspace=.02);