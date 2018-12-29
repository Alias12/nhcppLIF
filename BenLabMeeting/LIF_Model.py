import numpy as np
from math import factorial
import time
from bisect import bisect_left
import h5py

def expVes(lambdaI,nS,maxQ):
    expVesT = np.zeros((nS))
    for i in range(0,nS):
        for j in range(0,maxQ):
            expVesT[i]=expVesT[i] + (lambdaI[-1,i,j] * (j+1))
    return expVesT

def hybridInput(nT, nS, maxQ, aVals, aTheta, offSet, f, t, dt):
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
            
    exHybrid = expVes(lambdaHybrid,nS,maxQ+1)
    
    return lambdaHybrid, exHybrid

def rateInput(nT, nS, maxQ, exHybrid, offSet, f, t, dt):
    aRate = np.zeros((nS))
    lambdaInstRate = np.zeros((nT, nS))
    lambdaRate = np.zeros((nT, nS, maxQ+1))

    for j in range(0, nS):
        aRate[j] = 2*(exHybrid[j]-offSet*t[-1])/(t[-1]+(np.sin(2*np.pi*f*t[-1]))/(2*np.pi*f))
        lambdaInstRate[:,j] = offSet +  aRate[j] * (.5 * np.cos(2* np.pi * f * t)+.5)
        lambdaRate[:,j,0]= np.cumsum(lambdaInstRate[:, j]) * dt
        
    #exRate = expVes(lambdaRate,nS,maxQ+1)
    
    return lambdaRate#, exRate

def ampInput(nT, nS, maxQ, exHybrid, aTheta, offSet, f, t, dt):
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
    
    return lambdaAmp#, exAmp

def getLambdas(nT, nS, maxQ, maxAVal, maxTheta, offSet, f, t, dt):
    aVals = np.arange(0,maxAVal,maxAVal/nS)
    aTheta = np.arange(0,maxTheta,maxTheta/nS)
    
    lambdaHybrid, exHybrid = hybridInput(nT, nS, maxQ, aVals, aTheta, offSet, f, t, dt)
    lambdaRate = rateInput(nT, nS, maxQ, exHybrid, offSet, f, t, dt)
    lambdaAmp = ampInput(nT, nS, maxQ, exHybrid, aTheta, offSet, f, t, dt)
    
    return lambdaHybrid, lambdaRate, lambdaAmp

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
                fuckups[nS] +=1
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

def synAlpha (timeSeries, tau1, tau2, nT, nGen, dt):
    nF = int(0.01/dt)
    filt = np.zeros((nF))
    for i in range(0, nF):
        filt[i] = np.exp(-i*dt*1000*tau2)-np.exp(-i*dt*1000*tau1) #ampa/ka synaptic alpha function
        
    result = []
    for i in range(nGen):
        result.append(np.convolve(timeSeries[:, i], filt[:]))
    return result

def lifModelCurrVec(curr, nT, nS, vDev, vThresh, vRest, vReset, gLeak, res, dt, nGen, ns, inputT): # for faster spike counting
    #tVec = time.time()
    V = np.zeros((nT, nGen))
    V[0, :] = vRest
    #xScale = np.arange(0, recLen, dt)
    spikesMat = np.zeros((nT+1, nGen))
    
    lastSpike = np.zeros((nGen)) 
    
    for i in range(0, nT-1):
        V[i+1, :] = V[i,:] + (-(V[i,:] - vRest)*gLeak + res*curr[nS, i, :] + ns[i, nS, :])*dt #vectorized additive noise
        for j in range(0, nGen):
            if V[i+1, j]>=vThresh:
                V[i+1, j] = vReset
                spikesMat[i+1, j] = lastSpike[j] #saving last spike time for faster counting
                lastSpike[j] = i+1
    spikesMat[-1, :] = lastSpike[:]
    """name = 'samples/samples_'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    with h5py.File(name, 'r+') as data:
        name1 = inputT+'/voltage'
        name2 = inputT+'/current'
        dset1 = data[name1]
        dset2 = data[name2]
        dset1[:] = V[:, 0]
        dset2[:] = curr[nS, :, 0]
        data.close()"""
        
    #tVecEnd = time.time()
    return(spikesMat)

def countSpikes(spikeTrueMat, nT, nS, nGen, dt): 
    
    tSpikesStart = time.time()
    spikeTimeMat = []
    for j in range(0, nS):
        spikeTimeRow = []
        for k in range(0, nGen):
            spikeTimes = []
            i = nT
            while spikeTrueMat[j, i, k]>0:
                spikeTime = round(spikeTrueMat[j, i, k]*dt, -int(np.ceil(np.log10(dt))))
                spikeTimes.append(spikeTime)
                if i==int(spikeTrueMat[j, i, k]):
                    print('uhhh')
                    data = h5py.File('uhhh.hdf5', 'w')
                    fuckup = spikeTrueMat[j, :, k]
                    data.create_dataset('fuckup', data=fuckup)
                    data.close()
                    break
                i = int(spikeTrueMat[j, i, k])
                
            spikeTimeRow.append(np.flip(spikeTimes,0)) # spikeTimes is in reverse order, need to flip
        spikeTimeMat.append(spikeTimeRow)
    tSpikesEnd = time.time()
    
    print("counting spikes - %fs" % (tSpikesEnd-tSpikesStart))
    
    return spikeTimeMat

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

def runContrasts(lambdaI, nT, nS, nGen, maxQ, dt, vDev, vThresh, vRest, vReset, gLeak, res, tau1, tau2, inputT):

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
    
    for i in range (0, nS):
        tContrastStart = time.time()
        spikeTrueMat.append(lifModelCurrVec(curr, nT, i, vDev, vThresh, vRest, vReset, gLeak, res, dt, nGen, ns, inputT)) #run model 
        tContrastEnd = time.time()
        
        modelTimes[i] = tContrastEnd-tContrastStart
    
    spikeTrueMat = np.array(spikeTrueMat)
    timeSum = np.zeros((nS))
    timeSum[:] = np.cumsum(modelTimes[:])
    print("%d runs - %fs" % (nGen*nS, timeSum[-1]))
    
    
    ################## COUNTING SPIKES ##############################
    
    spikeTimeMat = countSpikes(spikeTrueMat, nT, nS, nGen, dt) 
    return spikeTimeMat, quantaMat, fuckups



def runAll(recLen, dt, f, nS, nGen, offSet, maxAVal, maxTheta, maxQ, vRest, vReset, vThresh, gLeak, res, vDev, tau1, tau2):
    t = np.arange(0,recLen,dt)
    nT = len(t)
    
    lambdaHybrid, lambdaRate, lambdaAmp = getLambdas(nT, nS, maxQ, maxAVal, maxTheta, offSet, f, t, dt)
    
    """name = 'samples/samples_'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    with h5py.File(name, 'w') as data:
        data.create_dataset('hybrid/voltage', data=np.zeros((nT)))
        data.create_dataset('hybrid/current', data=np.zeros((nT)))
        data.create_dataset('rate/voltage', data=np.zeros((nT)))
        data.create_dataset('rate/current', data=np.zeros((nT)))
        data.create_dataset('amp/voltage', data=np.zeros((nT)))
        data.create_dataset('amp/current', data=np.zeros((nT)))
        data.close()"""
        
    print('A/R/NHPP: ')
    hybridSpikeMat, hybridQuanta, hybridFuckups = runContrasts(lambdaHybrid, nT, nS, nGen, maxQ, dt, vDev, vThresh, vRest, vReset, gLeak, res, tau1, tau2, 'hybrid')
    print('R/NHPP: ')
    rateSpikeMat, rateQuanta, rateFuckups = runContrasts(lambdaRate, nT, nS, nGen, maxQ, dt, vDev, vThresh, vRest, vReset, gLeak, res, tau1, tau2, 'rate')
    print('A/HPP: ')
    ampSpikeMat, ampQuanta, ampFuckups = runContrasts(lambdaAmp, nT, nS, nGen, maxQ, dt, vDev, vThresh, vRest, vReset, gLeak, res, tau1, tau2, 'amp')
    
    # data saving
    
    print('saving data: ')
    tData = time.time()
    dFloat = h5py.special_dtype(vlen=np.dtype('float32'))
    #dInt = h5py.special_dtype(vlen=np.dtype('int32'))
    
    name = 'data/modelData'+str(vRest)+'_'+str(gLeak)+'_'+str(vDev)+'.hdf5'
    with h5py.File(name, 'w') as data:

        dSet1 = data.create_dataset('hybrid/spikeMatrix', (nS, nGen,), dtype=dFloat)
        dSet2 = data.create_dataset('hybrid/quanta', (nS, nGen,), dtype=int)
        data.create_dataset('hybrid/fuckups', dtype=int, data=hybridFuckups)

        dSet3 = data.create_dataset('rate/spikeMatrix', (nS, nGen,), dtype=dFloat)
        dSet4 = data.create_dataset('rate/quanta', (nS, nGen,), dtype=int)
        data.create_dataset('rate/fuckups', dtype=int, data=rateFuckups)

        dSet5 = data.create_dataset('amp/spikeMatrix', (nS, nGen,), dtype=dFloat)
        dSet6 = data.create_dataset('amp/quanta', (nS, nGen,), dtype=int)
        data.create_dataset('amp/fuckups', dtype=int, data=ampFuckups)
        
        for j in range(0, nS):
            dSet1[j] = hybridSpikeMat[j] 
            dSet2[j] = hybridQuanta[j] 
            dSet3[j] = rateSpikeMat[j] 
            dSet4[j] = rateQuanta[j] 
            dSet5[j] = ampSpikeMat[j] 
            dSet6[j] = ampQuanta[j] 
        
        data.close()
    
    tDataEnd = time.time()
    print('%fs' % (tDataEnd-tData))