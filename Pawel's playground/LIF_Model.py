import numpy as np
from math import factorial
import time
from bisect import bisect_left
import h5py
import matplotlib.pyplot as plt

def expVes(lambdaI,nS,maxQ):
    expVesT = np.zeros((nS))
    for i in range(0,nS):
        for j in range(0,maxQ):
            expVesT[i]=expVesT[i] + (lambdaI[-1,i,j] * (j+1))
    return expVesT

def expVesFull(lambdaI,nT,nS,maxQ):
    expVesT = np.zeros((nT, nS))
    for i in range(0,nS):
        for j in range(0,maxQ):
            for k in range(0, nT):
                expVesT[k, i]=expVesT[k, i] + (lambdaI[k,i,j] * (j+1))
    return expVesT

def hybridInput(nT, nS, maxQ, aVals, aTheta, offSet, f, t, dt, inputs):
    lambdaInstH = np.zeros((nT,nS))
    pInT = np.zeros((nT,nS))
    piI = np.zeros((nT,nS,maxQ+1))
    lambdaHybrid = np.zeros((nT,nS,maxQ+1))
    lambdaInstHybrid = np.zeros((nT,nS,maxQ+1))

    for j in range(0,nS):
        lambdaInstH[:,j] = offSet +  aVals[j] * (.5 * np.cos(2* np.pi * f * t)+.5)
        pInT[:,j] = aTheta[j] * (.5 * np.cos(2*np.pi * f * t)+.5)

        for k in range(0,maxQ+1):
            piI[:,j,k] = factorial(maxQ)/(factorial(maxQ-k)*factorial(k)) * pInT[:,j]**k * (1-pInT[:,j])**(maxQ-k)
            lambdaInstHybrid[:,j,k]= piI[:,j,k]* lambdaInstH[:,j]
            lambdaHybrid[:,j,k]= np.cumsum(lambdaInstHybrid[:,j,k]) * dt
            
    lambdaHybrid*=inputs
    exHybrid = expVes(lambdaHybrid,nS,maxQ+1)
    
    nP = int(1/dt/f)
    phases = int(nT/nP)
    bins = 16
    nBin = int(nP/bins) #bin size
    lambdaInt = np.zeros((bins, nS))
    
    for k in range(0, maxQ+1):
        for j in range(0, phases):
            for l in range(0, bins):
                for i in range(1, nBin):
                    for m in range(0, nS):
                        lambdaInt[l, m]+= (lambdaHybrid[j*nP+l*nBin+i, m, k]-lambdaHybrid[j*nP+l*nBin, m, k])*dt*(k+1)**2
                        
    lambdaInt = slideHalfPhase(lambdaInt, nS, bins)
                
    return lambdaHybrid, exHybrid, lambdaInt

def rateInput(nT, nS, maxQ, exHybrid, offSet, f, t, dt, lambdaAmp, inputs):
    aRate = np.zeros((nS))
    lambdaInstRate = np.zeros((nT, nS))
    lambdaRate = np.zeros((nT, nS, maxQ+1))
    a=-0.41028
    b=-0.026691
    
    
    for j in range(0, nS):
        aRate[j] = 2*(exHybrid[j]-offSet*t[-1])/(t[-1]+(np.sin(2*np.pi*f*t[-1]))/(2*np.pi*f))
        lambdaInstRate[:,j] = offSet +  aRate[j] * (.5 * np.cos(2* np.pi * f * t)+.5)
        lambdaRate[:,j,0]= np.cumsum(lambdaInstRate[:, j]) * dt# + b*t+a*np.sin(2*np.pi*f*t)
    
    
    exRate = expVesFull(lambdaRate,nT,nS,maxQ+1)
    exAmp = expVesFull(lambdaAmp,nT,nS,maxQ+1)
    
    diff = exRate-exAmp
    
    lambdaRate[:, :, 0] -= diff[:, :]
    
    nP = int(1/dt/f)
    phases = int(nT/nP)
    bins = 16
    nBin = int(nP/bins) #bin size
    lambdaInt = np.zeros((bins, nS))
    for j in range(0, phases):
        for l in range(0, bins):
            for i in range(1, nBin):
                for m in range(0, nS):
                    lambdaInt[l, m]+= (lambdaRate[j*nP+l*nBin+i, m, 0]-lambdaRate[j*nP+l*nBin, m, 0])*dt
                    
    lambdaInt = slideHalfPhase(lambdaInt, nS, bins)
    
    return lambdaRate, lambdaInt#, exRate

def ampInput(nT, nS, maxQ, exHybrid, aTheta, offSet, f, t, dt, inputs):
    lambdaInstHAmp = np.zeros((nT, nS))
    lambdaInstAmp = np.zeros((nT, nS, maxQ+1))
    lambdaAmp = np.zeros((nT, nS, maxQ+1))
    pInT = np.zeros((nT,nS))
    piI = np.zeros((nT,nS,maxQ+1))

    for j in range(0, nS):
        lambdaInstHAmp[:, j] = 2*exHybrid[j]/((aTheta[j]*maxQ+2)*t[-1])
        pInT[:,j] = aTheta[j] * (.5 * np.cos(2*np.pi * f * t)+.5)

        for k in range(0,maxQ+1):
            piI[:,j,k] = factorial(maxQ)/(factorial(maxQ-k)*factorial(k)) * pInT[:,j]**k * (1-pInT[:,j])**(maxQ-k)
            lambdaInstAmp[:,j,k]= piI[:,j,k]* lambdaInstHAmp[:,j]
            lambdaAmp[:,j,k]= np.cumsum(lambdaInstAmp[:,j,k]) * dt
            
    #exAmp = expVes(lambdaAmp,nS,maxQ+1)
    
    
    nP = int(1/dt/f)
    phases = int(nT/nP)
    bins = 16
    nBin = int(nP/bins) #bin size
    lambdaInt = np.zeros((bins, nS))
    
    for k in range(0, maxQ+1):
        for j in range(0, phases):
            for l in range(0, bins):
                for i in range(1, nBin):
                    for m in range(0, nS):
                        lambdaInt[l, m]+= (lambdaAmp[j*nP+l*nBin+i, m, k]-lambdaAmp[j*nP+l*nBin, m, k])*dt*(k+1)**2
    
    lambdaInt = slideHalfPhase(lambdaInt, nS, bins)
    
    return lambdaAmp, lambdaInt#, exAmp

def getLambdas(nT, nS, maxQ, maxAVal, maxTheta, offSet, f, t, dt, inputs):
    aVals = np.linspace(maxAVal/nS,maxAVal,num=nS)
    aTheta = np.linspace(maxTheta/nS,maxTheta,num=nS)
    
    lambdaHybrid, exHybrid, exHybridVar = hybridInput(nT, nS, maxQ, aVals, aTheta, offSet, f, t, dt, inputs)
    lambdaAmp, exAmpVar = ampInput(nT, nS, maxQ, exHybrid, aTheta, offSet, f, t, dt, inputs)
    lambdaRate, exRateVar = rateInput(nT, nS, maxQ, exHybrid, offSet, f, t, dt, lambdaAmp, inputs)
    
    return lambdaHybrid, lambdaRate, lambdaAmp, exHybridVar, exRateVar, exAmpVar

def sampleEvents(lambdaI, nT, nS, maxQ, dt, nGen, fuckups): 
    eventMat = []
    for k in range(0, maxQ+1):
        eventRow = []
        for j in range(0, nGen):
            events = []
            u = 0
            i = 0
            counter = 0
            iNext = 0
            maxU = lambdaI[-1, nS, k]
            while u<maxU:
                rand = np.random.uniform(0, 1) 
                u -= np.log(rand)
                u = np.round(u, int(np.ceil(-np.log10(dt))))
                if u<maxU:
                    iNext = bisect_left(lambdaI[:, nS, k], u, lo=i, hi=nT-1)-1 #binsearching the i value of the infimum
                    if iNext >= nT-2:
                        break
                    if iNext<=0:
                        break
                    counter+=1
                    if i>=iNext:
                        continue
                    i=iNext
                    event = np.round(i*dt, int(np.ceil(-np.log10(dt))))
                    events.append(event)
            if not (len(events)==counter):
                fuckups[nS] += (counter-len(events))*(k+1)
            eventRow.append(events)
        eventMat.append(eventRow)
    return eventMat

def makeTimeSeries(eventMat, quantaMat, nT, nS, maxQ, dt, nGen):
    timeSeries = np.zeros((nT, nGen))
    for k, eventRow in enumerate(eventMat[nS]):
        for j, events in enumerate(eventRow):
            for event in events:
                timeSeries[int(event/dt), j] += k+1
                quantaMat[nS, j] += k+1
    return timeSeries

def getCumQuanta(timeSeries, nT, nS, nGen): # timeSeries[contrastN, tN, genN]
    cumQuanta = np.zeros((nT, nS))
    varQuanta = np.zeros((nT, nS))
    for i in range(0, nT):
        for j in range(nS):
            if (i>0):
                cumQuanta[i, j] = cumQuanta[i-1, j]
            for k in range(nGen):
                cumQuanta[i, j] += timeSeries[j, i, k]/nGen
                    
            varQuanta[i, j] = np.var(timeSeries[j, i, :])

    return cumQuanta, varQuanta

def getPhaseVar(timeSeries, nT, nGen, f, dt, bins, nS):
    nP = int(1/dt/f) #phase size
    nBin = int(nP/bins) #bin size
    genPhaseVar = np.zeros((bins, nGen, nS)) #distribution for each gen
    numPhases = int(nT/nP) #number of phases
    phaseVar = np.zeros((bins, nS)) #distribution for contrast
    
    for i in range(0, numPhases):
        for j in range(0, bins):
            for k in range(0, nGen):
                for l in range(0, nS):
                    genPhaseVar[j, k, l] += np.var(timeSeries[l, (i*nP+j*nBin):(i*nP+(j+1)*nBin), k])
    
    for i in range(0, bins):
        for j in range(0, nS):
            phaseVar[i, j] = np.mean(genPhaseVar[i, :, j])/f**2/dt/500
            
    phaseVar = slideHalfPhase(phaseVar, nS, bins)
        
    return phaseVar

def synAlpha (timeSeries, tau1, tau2, nT, nGen, dt):
    nF = int(tau1/200/dt)
    filt = np.zeros((nF))
    for i in range(0, nF):
        filt[i] = np.exp(-i*dt*1000/tau1)-np.exp(-i*dt*1000/tau2) #ampa/ka synaptic alpha function
        
    result = []
    for i in range(nGen):
        result.append(np.convolve(timeSeries[:, i], filt[:]))
    return result

def lifModelCurrVec(curr, spikeMat, nT, nS, vDev, vThresh, vRest, vReset, gLeak, res, dt, nGen, ns, inputT, save): # for faster spike counting
    #tVec = time.time()
    V = np.zeros((nT, nGen))
    V[0, :] = vRest
    
    freeV = np.zeros((nT, nGen))
    freeV[0, :] = vRest
    #xScale = np.arange(0, recLen, dt)
    
    meanV = np.zeros((nT))
    meanFreeV = np.zeros((nT))
    meanV[0] = vRest*nGen
    meanFreeV[0] = vRest*nGen
    
    for i in range(0, nT-1):
        V[i+1, :] = V[i,:] + (-(V[i,:] - vRest)*gLeak + res*curr[nS, i, :] + ns[i, nS, :])*dt #vectorized additive noise
        freeV[i+1, :] = freeV[i,:] + (-(freeV[i,:] - vRest)*gLeak + res*curr[nS, i, :] + ns[i, nS, :])*dt
        
        for j in range(0, nGen):
            if V[i+1, j]>=vThresh:
                V[i+1, j] = vReset
                spikeMat.append([round((i+1)*dt, -int(np.ceil(np.log10(dt)))), nS, j])
           
            meanV[i+1] += V[i+1, j]
            meanFreeV[i+1] += freeV[i+1, j]
            
    meanV/=nGen
    meanFreeV/=nGen
       
    if(save==True):
        output=V
        output2=freeV
    else:
        output = 0
        output2 = 0
        
    return meanV, meanFreeV, output, output2
    #tVecEnd = time.time()
    
def lifModelCondVec(curr, spikeMat, nT, nS, vDev, vThresh, vRest, vReset, gLeak, gExc, dt, nGen, ns, inputT, save, vExc): # for faster spike counting
    #tVec = time.time()
    V = np.zeros((nT, nGen))
    V[0, :] = vRest
    
    freeV = np.zeros((nT, nGen))
    freeV[0, :] = vRest
    #xScale = np.arange(0, recLen, dt)
    
    meanV = np.zeros((nT))
    meanFreeV = np.zeros((nT))
    meanV[0] = vRest*nGen
    meanFreeV[0] = vRest*nGen
    
    for i in range(0, nT-1):
        V[i+1, :] = V[i,:] + (-(V[i,:] - vRest)*gLeak + gExc*(vExc-V[i, :])*curr[nS, i, :] + ns[i, nS, :])*dt #vectorized additive noise
        freeV[i+1, :] = freeV[i,:] + (-(freeV[i,:] - vRest)*gLeak + gExc*(vExc-V[i, :])*curr[nS, i, :] + ns[i, nS, :])*dt
        
        for j in range(0, nGen):
            if V[i+1, j]>=vThresh:
                V[i+1, j] = vReset
                spikeMat.append([round((i+1)*dt, -int(np.ceil(np.log10(dt)))), nS, j])
           
            meanV[i+1] += V[i+1, j]
            meanFreeV[i+1] += freeV[i+1, j]
            
    meanV/=nGen
    meanFreeV/=nGen
       
    if(save==True):
        output=V
        output2=freeV
    else:
        output = 0
        output2 = 0
        
    return meanV, meanFreeV, output, output2
    #tVecEnd = time.time()
    
def countSpikes(spikeMat, nS, nGen, dt): 
    
    tSpikesStart = time.time()
    spikeTimeMat = []
    maxSpikes = 0
    
    n = len(spikeMat)
    
    if n<1:
        spikeTimeMat = np.full((nS, nGen, 2), np.nan)
        return spikeTimeMat, 2
    
    i=0
    for j in range (0, nS):
        contrastSpikeMat = []
        while(i<n and spikeMat[i][1]==j):
            contrastSpikeMat.append([spikeMat[i][0], spikeMat[i][2]])
            i+=1
        
        nk = len(contrastSpikeMat)
        if (nk<1):
            spikeTimeMat.append([])
        else:
            spikeTimeRow=[]
            contrastSpikeMat=sorted(contrastSpikeMat, key=lambda x: x[1])
            k = 0
            genN = 0
            while (genN<nGen):
                spikeTimes=[]
                spikes = 0
                while(k<nk and contrastSpikeMat[k][1]==genN):
                    spikeTimes.append(contrastSpikeMat[k][0])
                    k+=1
                    spikes += 1
                maxSpikes = max(maxSpikes, spikes)
                spikeTimeRow.append(spikeTimes)
                genN+=1
            spikeTimeMat.append(spikeTimeRow)
    
    tSpikesEnd = time.time()
    print("counting spikes - %fs" % (tSpikesEnd-tSpikesStart))
    
    return spikeTimeMat, maxSpikes

def createInput (lambdaI, eventMat, timeSeries, quantaMat, curr, nT, nS, nGen, maxQ, dt, tau1, tau2):
    tPreStart = time.time()
    fuckups = np.zeros((nS))
    
    for i in range (0, nS):
        eventMat.append(sampleEvents(lambdaI, nT, i, maxQ, dt, nGen, fuckups))
    
    for i in range (0, nS):
        timeSeries.append(makeTimeSeries(eventMat, quantaMat, nT, i, maxQ, dt, nGen)) 
    
    for i in range (0, nS):
        curr.append(np.transpose(synAlpha(timeSeries[i], tau1, tau2, nT, nGen, dt)))
    
    tPreEnd = time.time()
    print("pre - %fs" % (tPreEnd-tPreStart))
    return fuckups

def runContrasts(lambdaI, nT, nS, nGen, maxQ, dt, vDev, vThresh, vRest, vReset, gLeak, res, tau1, tau2, inputT, save, f, gExc, vExc, doCond, saveQuantaStats, savePhaseStats, doCurr):

    modelTimes = np.zeros((nS))
    spikeTrueMat = []
    ns = np.random.normal(0,vDev,[nT, nS, nGen])
    
    
    
    ############ CREATE INPUTS ###################################
    
    eventMat = []
    timeSeries = []
    curr = []
    quantaMat = np.zeros((nS, nGen))
    
    fuckups = createInput(lambdaI, eventMat, timeSeries, quantaMat, curr, nT, nS, nGen, maxQ, dt, tau1, tau2)
    
    eventMat = np.array(eventMat) # eventMat[contrastN, tN, genN]
    timeSeries = np.array(timeSeries) # timeSeries[contrastN, tN, genN]
    curr = np.array(curr) # curr[contrastN, tN, genN]
    
    
    ###################### MODEL RUN ###############################
    
    spikeMat = []
    allVs = []
    allFreeVs = []
    meanVs = []
    meanFreeVs = []
    spikeCondMat = []
    allConds = []
    allFreeConds = []
    meanConds = []
    meanFreeConds = []
    for i in range (0, nS):
        tContrastStart = time.time()
        if (doCurr == True):
            meanV, meanFreeV, Vs, freeVs = lifModelCurrVec(curr, spikeMat, nT, i, vDev, vThresh, vRest, vReset, gLeak, res, dt, nGen, ns, inputT, save)
            meanVs.append(meanV)
            meanFreeVs.append(meanFreeV)
            allVs.append(Vs)
            allFreeVs.append(freeVs)
        
        if (doCond == True):
            meanCond, meanFreeCond, conds, freeConds = lifModelCondVec(curr, spikeCondMat, nT, i, vDev, vThresh, vRest, vReset, gLeak, gExc, dt, nGen, ns, inputT, save, vExc)
            meanConds.append(meanCond)
            meanFreeConds.append(meanFreeCond)
            allConds.append(conds)
            allFreeConds.append(freeConds)
        tContrastEnd = time.time()
        
        modelTimes[i] = tContrastEnd-tContrastStart
    
    
    
    if (saveQuantaStats == True):
        cumQuanta, varQuanta = getCumQuanta(timeSeries, nT, nS, nGen)
    if (savePhaseStats == True):
        phaseVar = getPhaseVar(timeSeries, nT, nGen, f, dt, 16, nS)
    
    timeSum = np.zeros((nS))
    timeSum[:] = np.cumsum(modelTimes[:])
    print("%d runs - %fs" % (nGen*nS, timeSum[-1]))
    ################## COUNTING SPIKES ##############################
    cleanSpikeTimes = []
    maxSpikes = 0
    maxSpikesCond = 0
    if (doCurr == True):
        spikeTimeMat, maxSpikes = countSpikes(spikeMat, nS, nGen, dt)
        cleanSpikeTimes = cleanSpikeMat(spikeTimeMat, nS, nGen, maxSpikes)
    
    cleanCondTimes = []
    if (doCond == True):
        condTimeMat, maxSpikesCond = countSpikes(spikeCondMat, nS, nGen, dt)
        cleanCondTimes = cleanSpikeMat(condTimeMat, nS, nGen, maxSpikesCond)
    
    return cleanSpikeTimes, quantaMat, fuckups, meanVs, meanFreeVs, cumQuanta, varQuanta, allVs, allFreeVs, phaseVar, cleanCondTimes, meanConds, meanFreeConds, allConds, allFreeConds, maxSpikes, maxSpikesCond


def getFirstSpikes(spikeMatrix, nS, nGen):
    
    firstSpikes = np.zeros((nS, nGen))
    
    for j in range(0, nS):
        for k in range(0, nGen):
            if (len(spikeMatrix[j][k])>0):
                firstSpikes[j, k] = spikeMatrix[j][k][0]
            else:
                firstSpikes[j, k] = np.nan
                
    return firstSpikes


def cleanSpikeMat(spikeMatrix, nS, nGen, maxSpikes):
    cleanSpikes = np.full((nS, nGen, max(maxSpikes, 1)), np.nan)
    for j in range(0, nS):
        for k in range(0, nGen):
            for i in range(0, len(spikeMatrix[j][k])):
                cleanSpikes[j, k, i] = spikeMatrix[j][k][i]
    return cleanSpikes


def slideHalfPhase(arr, nS, bins):
    slidedArr = np.zeros((bins, nS))
    slide = int(bins/2)
    for i in range(bins):
        for j in range(nS):
            slidedArr[i, j] = arr[(i+slide)%bins, j]
            
    return slidedArr


def runAll(recLen, dt, f, nS, nGen, offSet, maxAVal, maxTheta, maxQ, vRest, vReset, vThresh, gLeak, res, vDev, tau1, tau2, save, inputs, gExc, vExc, doCond, doHybrid, doRate, doAmp, saveMeanV, saveMeanFreeV, saveQuantaStats, savePhaseStats, saveFirstSpikes, doCurr):
    t = np.arange(0,recLen,dt)
    nT = len(t)
    
    lambdaHybrid, lambdaRate, lambdaAmp, exHybridVar, exRateVar, exAmpVar = getLambdas(nT, nS, maxQ, maxAVal, maxTheta, offSet, f, t, dt, inputs)
        
    if (doHybrid == True):
        print('A/R/NHPP: ')
        hybridSpikeMat, hybridQuanta, hybridFuckups, hybridMean, hybridFree, hybridInput, hybridVar, hybridVs, hybridFreeVs, hybridPhaseVar, hybridCondTimes, hybridMeanCond, hybridMeanFreeCond, hybridAllCond, hybridAllFreeCond, hybridSpikes, hybridSpikesCond = runContrasts(lambdaHybrid, nT, nS, nGen, maxQ, dt, vDev, vThresh, vRest, vReset, gLeak, res, tau1, tau2, 'hybrid', save, f, gExc, vExc, doCond, saveQuantaStats, savePhaseStats, doCurr)
        lambdaH = expVesFull(lambdaHybrid,nT,nS,maxQ+1)
        if (saveFirstSpikes == True):
            if (doCurr == True):
                hybridFirst = getFirstSpikes(hybridSpikeMat, nS, nGen)
            if (doCond == True):
                hybridFirstCond = getFirstSpikes(hybridCondTimes, nS, nGen)
        
    if (doRate == True):
        print('R/NHPP: ')
        rateSpikeMat, rateQuanta, rateFuckups, rateMean, rateFree, rateInput, rateVar, rateVs, rateFreeVs, ratePhaseVar, rateCondTimes, rateMeanCond, rateMeanFreeCond, rateAllCond, rateAllFreeCond, rateSpikes, rateSpikesCond = runContrasts(lambdaRate, nT, nS, nGen, maxQ, dt, vDev, vThresh, vRest, vReset, gLeak, res, tau1, tau2, 'rate', save, f, gExc, vExc, doCond, saveQuantaStats, savePhaseStats, doCurr)
        lambdaR = expVesFull(lambdaRate,nT,nS,maxQ+1)
        if (saveFirstSpikes == True):
            if (doCurr == True):
                rateFirst = getFirstSpikes(rateSpikeMat, nS, nGen)
            if (doCond == True):
                rateFirstCond = getFirstSpikes(rateCondTimes, nS, nGen)
        
    if (doAmp == True):
        print('A/HPP: ')
        ampSpikeMat, ampQuanta, ampFuckups, ampMean, ampFree, ampInput, ampVar, ampVs, ampFreeVs, ampPhaseVar, ampCondTimes, ampMeanCond, ampMeanFreeCond, ampAllCond, ampAllFreeCond, ampSpikes, ampSpikesCond = runContrasts(lambdaAmp, nT, nS, nGen, maxQ, dt, vDev, vThresh, vRest, vReset, gLeak, res, tau1, tau2, 'amp', save, f, gExc, vExc, doCond, saveQuantaStats, savePhaseStats, doCurr)
        lambdaA = expVesFull(lambdaAmp,nT,nS,maxQ+1)
        if (saveFirstSpikes == True):
            if (doCurr == True):
                ampFirst = getFirstSpikes(ampSpikeMat, nS, nGen)
            if (doCond == True):
                ampFirstCond = getFirstSpikes(ampCondTimes, nS, nGen)
    
    # data saving
    
    print('saving data: ')
    tData = time.time()
    dFloat = h5py.special_dtype(vlen=np.dtype('float32'))
    
    name = 'data/modelData'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'_'+str(inputs)+'.h5'
    metadata = 'Model parameters: vRest = '+str(vRest)+'; gLeak = '+str(gLeak)+', vDev = '+str(vDev)+', dt = '+str(dt)+', '+str(inputs)+' inputs used at frequency of '+str(f)+'Hz; '+str(nGen)+' simulations for each.'
    with h5py.File(name, 'w') as data:
        
        data.create_dataset('metadata', data=metadata)
        
        #######HYBRID SAVING##############
        if (doHybrid == True):
            
            if (doCurr == True):
                data.create_dataset('hybrid/current/spikeMatrix', (nS, nGen, hybridSpikes), data=hybridSpikeMat) 
                if (saveMeanV == True): 
                    data.create_dataset('hybrid/current/meanV', (nS, nT), data=hybridMean)
                if (saveMeanFreeV == True):
                    data.create_dataset('hybrid/current/freeV', (nS, nT), data=hybridFree)
                if (saveFirstSpikes == True):
                    data.create_dataset('hybrid/current/firstSpikes', data=hybridFirst)
                if (save==True):
                    data.create_dataset('hybrid/current/VSamples', (nS, nT, nGen), data = hybridVs)
                    data.create_dataset('hybrid/current/freeVSamples', (nS, nT, nGen), data = hybridFreeVs)
                    
            if (doCond == True):
                data.create_dataset('hybrid/conductance/spikeMatrix', (nS, nGen, hybridSpikesCond), data=hybridCondTimes)
                if (saveMeanV == True):
                    data.create_dataset('hybrid/conductance/meanV', (nS, nT), data=hybridMeanCond)
                if (saveMeanFreeV == True):
                    data.create_dataset('hybrid/conductance/meanFreeV', (nS, nT), data=hybridMeanFreeCond)
                if (saveFirstSpikes == True):
                    data.create_dataset('hybrid/conductance/firstSpikes', data=hybridFirstCond)
                if (save==True):
                    data.create_dataset('hybrid/conductance/VSamples', (nS, nT, nGen), data = hybridAllCond)
                    data.create_dataset('hybrid/conductance/freeVSamples', (nS, nT, nGen), data = hybridAllFreeCond)
                    
            if (saveQuantaStats == True):
                data.create_dataset('hybrid/cumQuanta', (nT, nS), data = hybridInput)
                data.create_dataset('hybrid/lambda', (nT, nS), data = lambdaH)
                data.create_dataset('hybrid/varQuanta', (nT, nS), data = hybridVar)
                data.create_dataset('hybrid/quanta', (nS, nGen), data=hybridQuanta, dtype=int)
                data.create_dataset('hybrid/fuckups', dtype=int, data=hybridFuckups)
                
            if (savePhaseStats == True):
                data.create_dataset('hybrid/phaseVar', data=hybridPhaseVar)
                data.create_dataset('hybrid/expPhaseVar', data=exHybridVar)
            
        #######RATE SAVING##############
        if (doRate == True):
            
            if (doCurr == True):
                data.create_dataset('rate/current/spikeMatrix', (nS, nGen, rateSpikes), data=rateSpikeMat)
                if (saveMeanV == True):
                    data.create_dataset('rate/current/meanV', (nS, nT), data=rateMean)
                if (saveMeanFreeV == True):
                    data.create_dataset('rate/current/freeV', (nS, nT), data=rateFree)
                if (saveFirstSpikes == True):
                    data.create_dataset('rate/current/firstSpikes', data=rateFirst)
                if (save==True):
                    data.create_dataset('rate/current/VSamples', (nS, nT, nGen), data = rateVs)
                    data.create_dataset('rate/current/freeVSamples', (nS, nT, nGen), data = rateFreeVs)
                
            if (doCond == True):
                data.create_dataset('rate/conductance/spikeMatrix', (nS, nGen, rateSpikesCond), data=rateCondTimes)
                if (saveMeanV == True):
                    data.create_dataset('rate/conductance/meanV', (nS, nT), data=rateMeanCond)
                if (saveMeanFreeV == True):
                    data.create_dataset('rate/conductance/freeV', (nS, nT), data=rateMeanFreeCond)
                if (saveFirstSpikes == True):
                    data.create_dataset('rate/conductance/firstSpikes', data=rateFirstCond)
                if (save==True):
                    data.create_dataset('rate/conductance/VSamples', (nS, nT, nGen), data = rateAllCond)
                    data.create_dataset('rate/conductance/freeVSamples', (nS, nT, nGen), data = rateAllFreeCond)
                
            if (saveQuantaStats == True):
                data.create_dataset('rate/cumQuanta', (nT, nS), data = rateInput)
                data.create_dataset('rate/lambda', (nT, nS), data = lambdaR)
                data.create_dataset('rate/varQuanta', (nT, nS), data = rateVar)
                data.create_dataset('rate/quanta', (nS, nGen), dtype=int, data=rateQuanta)
                data.create_dataset('rate/fuckups', dtype=int, data=rateFuckups)
                
            if (savePhaseStats == True):
                data.create_dataset('rate/phaseVar', data=ratePhaseVar)
                data.create_dataset('rate/expPhaseVar', data=exRateVar)
            
                    
                
        #######AMP SAVING##############
        if (doAmp == True):
            
            if (doCurr == True):
                data.create_dataset('amp/current/spikeMatrix', (nS, nGen, ampSpikes), data=ampSpikeMat)
                if (saveMeanV == True):
                    data.create_dataset('amp/current/meanV', (nS, nT), data=ampMean)
                if (saveMeanFreeV == True):
                    data.create_dataset('amp/current/freeV', (nS, nT), data=ampFree)
                if (saveFirstSpikes == True):
                    data.create_dataset('amp/current/firstSpikes', data=ampFirst)
                if (save==True):
                    data.create_dataset('amp/current/VSamples', (nS, nT, nGen), data = ampVs)
                    data.create_dataset('amp/current/freeVSamples', (nS, nT, nGen), data = ampFreeVs)
            
            if (doCond == True):
                data.create_dataset('amp/conductance/spikeMatrix', (nS, nGen,ampSpikesCond), data=ampCondTimes)
                if (saveMeanV == True):
                    data.create_dataset('amp/conductance/meanV', (nS, nT), data=ampMeanCond)
                if (saveMeanFreeV == True):
                    data.create_dataset('amp/conductance/freeV', (nS, nT), data=ampMeanFreeCond)
                if (saveFirstSpikes == True):
                    data.create_dataset('amp/conductance/firstSpikes', data=ampFirstCond)
                if (save==True):
                    data.create_dataset('amp/conductance/VSamples', (nS, nT, nGen), data = ampAllCond)
                    data.create_dataset('amp/conductance/freeVSamples', (nS, nT, nGen), data = ampAllFreeCond)
                
            if (saveQuantaStats == True):
                data.create_dataset('amp/cumQuanta', (nT, nS), data = ampInput)
                data.create_dataset('amp/lambda', (nT, nS), data = lambdaA)
                data.create_dataset('amp/varQuanta', (nT, nS), data = ampVar)
                data.create_dataset('amp/quanta', (nS, nGen), dtype=int, data=ampQuanta)
                data.create_dataset('amp/fuckups', dtype=int, data=ampFuckups)
                
            if (savePhaseStats == True):
                data.create_dataset('amp/phaseVar', data=ampPhaseVar)
                data.create_dataset('amp/expPhaseVar', data=exAmpVar)
        
        data.close()
    tDataEnd = time.time()
    print('%fs' % (tDataEnd-tData))
    
def getPhase(phase, evTime, dt, nP): # - getPhase[t] / getPhase[iT*dt]
    return phase[int(evTime/dt%nP)]