# -*- coding: utf-8 -*-
"""

//#############################################################################
//  File:      MainOffshoreWave.py
//  Function:  Generates the plots for 'Testing Resilience Aspects of Operation 
//             Options for Offshore Wind Farms beyond the End-of-Life' by 
//             Corinna Köpke, Jennifer Mielniczek and Alexander Stolz 
//  Author:    Corinna Köpke
//  Date:      June-23
//  Copyright: (c) Fraunhofer Institute for High-Speed-Dynamics EMI
//#############################################################################

"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import calendar
import seaborn as sns
import copy
import matplotlib as mpl

#https://stackoverflow.com/questions/4130922/how-to-increment-datetime-by-custom-months-in-python-without-using-library
def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)

###############################################################################
# choose EOL option
Mode = 'Decom'#'Decom', #'Exten', 'Repow'

##############################################################################
print('Generate synthetic wave heights')

nummonths = 60*12 #60 years
waveFrequency = 10 #sec
loadChanges = 6*60*24*30

base = datetime.date(2015,1,1)
now = datetime.date.today()
eol = datetime.date(2040,1,1)
date_list = [add_months(base, x) for x in range(nummonths)]
t = date_list

severeWeatherSeasons = [11, 12, 1, 2, 3]
Hsig, Hmax = [],[]

def rayleighX(x,sigma):
    f = (x/sigma**2)*np.exp(-(x**2)/(2*sigma**2))
    return f

rs1 = 1.5
rs2 = 0.7
Hmin = 8 #smallest wave height to consider
waveHeightNum = []
hNum = []
for i in t:
    if i.month in severeWeatherSeasons:
        rs = rs1
    else:
        rs = rs2
    hsig = np.random.rayleigh(scale=rs, size=1) 
    plus = np.random.uniform(low=0, high=hsig*(0.07*(i.year-base.year)+3))
    if abs(plus) > 20:
        print('Large:',plus)
        plus = np.array(20)
    hmax = hsig+abs(plus)
    Hsig.append(hsig)
    Hmax.append(hmax)
    
    whn = [] #estimate the number of load changes for each discrete wave height:
    h = []
    for j in range(Hmin,int(hmax)):
        h.append(j)
        r = rayleighX(j,rs)*loadChanges
        if r > 1:
            whn.append(r)
        else:
            whn.append(1)
    waveHeightNum.append(whn)
    hNum.append(h)
    
load = []
for w in waveHeightNum:
    if w == []:
        load.append(0)
    else:
        load.append(sum(w))

            
##############################################################################
FS = 30
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)

plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.plot(t, Hmax,color='k',label='H maximal')
plt.plot(t, Hsig,color='red',label='H significant')

plt.xlabel('Time', fontsize=FS)
plt.ylabel('Wave height H', fontsize=FS)

plt.legend(fontsize=FS)

name = 'WaveHeight.png'
fig.savefig(name, dpi=100)

##############################################################################
s = 1000000
test = np.random.rayleigh(scale=np.mean([rs1,rs2]), size=s)


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)

ax.set_xlim(0, 12) 

plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
sns.distplot(a=test, hist=False, color='k',kde_kws={'linestyle':'--'},label='Rayleigh distribution')
sns.distplot(a=Hsig, bins=20, hist=True, color='k',label='H significant')
sns.distplot(a=Hmax, bins=100, hist=True, color='grey',label='H maximal')

plt.xlabel('Wave height H', fontsize=FS)
plt.ylabel('Probability density', fontsize=FS)

plt.legend(fontsize=FS)

name = 'WaveHist.png'
fig.savefig(name, dpi=100)

##############################################################################
print('Estimate the expected load')
        

fig = plt.figure(figsize=(15,10)) #width, height
ax = fig.add_subplot(111)

plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.plot(t, load, color='k',label='Load')
plt.xlabel('Time', fontsize=FS)
plt.ylabel('Load changes', fontsize=FS)


name = 'Load.png'
fig.savefig(name, dpi=100)
pmax = 0.5

#See: https://www.fino3.de/de/standort/hydrologie.html
#https://www.fino1.de/de/standort/wellen-stroemung.html

def loadFailure(minH,maxH,xtest):
    ytest = []
    if xtest == 0:
        return 10000 
        
    if -0.5*np.log(xtest)+maxH > minH:
        ytest = -0.5*np.log(xtest)+maxH
    else:
        ytest = minH

    return ytest


x = np.logspace(0,10,num=100,base=10)
y, yDe = [],[]
for i in x:
    y.append(loadFailure(8,17,i))
    yDe.append(loadFailure(10,19,i))

##############################################################################       
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)

plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.semilogx(x, y, color='k',label='Existing structures') #semilogx
plt.semilogx(x, yDe, color='r',label='New structures') #semilogx
plt.xlabel('Acceptable load changes', fontsize=FS)
plt.ylabel('Wave height H', fontsize=FS)

# build a rectangle in axes coords
left, width = .15, .5
bottom, height = .15, .5
right = left + width
top = bottom + height

props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
plt.text(right, top, 'p = 0.5', transform=ax.transAxes, fontsize=FS,
        verticalalignment='bottom', bbox=props)

plt.text(left, bottom, 'p = 0.1', transform=ax.transAxes, fontsize=FS,
        verticalalignment='bottom', bbox=props)

plt.legend(fontsize=FS)


name = 'LoadWaves.png'
fig.savefig(name, dpi=100)

#############################################################################
#Decommissioning:
decomDate = datetime.date(2042,1,1) #date when decom is finished
Rebuild = datetime.date(2050,1,1)
dDT = date_list.index(Rebuild) 
#save in WaveTime all load changes for certain waves higher than Hmin
WaveTime, WaveTimeCumSum, WaveTimeCumSumDe = [],[],[]
for i in range(0, len(max(waveHeightNum))):
    temp,temp2,temp3,temp4 = [],[],[],[]
    for t in range(0,len(date_list)):
        if waveHeightNum[t] == []:
            temp.append(0)
            temp4.append(0)
        elif i < len(waveHeightNum[t]):
            temp.append(waveHeightNum[t][i])
            temp4.append(waveHeightNum[t][i])
        
        
        if date_list[t] > decomDate and date_list[t] <= Rebuild:
            temp4 = [] #set temp4 to empty array to start from zero
            temp3.append(0)
        else:
            temp3.append(sum(temp4))
        temp2.append(sum(temp))
    WaveTime.append(temp)
    WaveTimeCumSum.append(temp2)
    WaveTimeCumSumDe.append(temp3)

#############################################################################
print('Estimate the time when design specifications are exceeded')
phigh = 0.5
plow = 0.1

def degradationTime(WTlist, wmin, wmax, phigh, plow):
    dt =-1
    #loop over time:
    prop = []
    for t in date_list:
        #loop over significant waves:
        dt = dt+1
        dw=-1
        p = []
        for w in WTlist:
            dw = dw+1
            if max(hNum)[dw] > loadFailure(wmin,wmax,w[dt]):
                p.append('exceed')
            else:
                p.append('lower')
        if 'exceed' in p:
            prop.append(phigh)
        else:
            prop.append(plow)
    tdeg = prop.index(0.5)
    return tdeg, prop
    
wmin = 8
wmax = 17
WTlist = WaveTimeCumSum
tdeg, prop = degradationTime(WTlist, wmin, wmax, phigh, plow)

wmin = 10
wmax = 19
WTlist = WaveTimeCumSumDe
tdegDe, propDe = degradationTime(WTlist, wmin, wmax, phigh, plow)

            
#################################################################
def plotCumSum(WTlist, t, name):
    
    cmap = []
    N = len(WTlist)
    cmap = plt.get_cmap('viridis', N)

    fig = plt.figure(figsize=(15,10))
    plt.xticks(fontsize=FS)
    plt.yticks(fontsize=FS)
    
    w=-1
    for i in WTlist:
        w = w+1
        plt.semilogy(date_list, i, color=cmap(w),label=name) 
    plt.plot([date_list[t],date_list[t]], [min(WTlist[0]),max(WTlist[0])],color='k',label='H maximal')
    plt.plot([eol, eol], [min(WTlist[0]),max(WTlist[0])], color='k',linestyle=':')
    plt.xlabel('Time', fontsize=FS)
    plt.ylabel('Cumulative sum of load changes', fontsize=FS)
    
    minC = Hmin
    maxC = max(hNum)[-1]
    norm = mpl.colors.Normalize(vmin=minC, vmax=maxC)
      
    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
      
    cb = plt.colorbar(sm, ticks=np.linspace(minC, maxC, N))
    cb.ax.tick_params(labelsize=FS)
    fig.savefig(name, dpi=100)

plotCumSum(WaveTimeCumSum, tdeg, 'CumulativeTime.png')
plotCumSum(WaveTimeCumSumDe, tdegDe, 'CumulativeTimeDecom.png')

#############################################################################
print('Generate simple 3 OSS network')
RT = 1
RO = 2
numTurbinesPerOSS = 60

#Hard coded 3 OSS: 
TO = [i for i in range(3*numTurbinesPerOSS+3)]
TT = ['Turbine']*(numTurbinesPerOSS*3)
TT.append('OSS1')
TT.append('OSS2')
TT.append('OSS3')
#https://www.wind-energie.de/fileadmin/redaktion/dokumente/publikationen-oeffentlich/themen/06-zahlen-und-fakten/20230116_Status_des_Offshore-Windenergieausbaus_Jahr_2022.pdf

counters = np.zeros(len(TO))
broken = np.zeros(len(TO))

OSS = {
       'OSS1':[i for i in range(0,1*numTurbinesPerOSS)],
       'OSS2':[i for i in range(1*numTurbinesPerOSS,2*numTurbinesPerOSS)],
       'OSS3':[i for i in range(2*numTurbinesPerOSS,3*numTurbinesPerOSS)]
       }

#############################################################################
print('Destroy something')
tt = -1
Broken = []
for t in date_list:
    tt = tt+1
    if Mode == 'Decom' and t > Rebuild:
        propUse = propDe
    else:
        propUse = prop
    for i in TO:
        counters[i] = counters[i]-1
        smalln = np.random.uniform(0,1,size=1)
        if smalln <= propUse[tt] and Hmax[tt] > 15 and counters[i] < 0: #consider waves larger 15
            counters[i] = 2 #fixed repair time
            broken[i] = 1
        if counters[i] == 0:
            #repair
            broken[i] = 0
    Broken.append(copy.deepcopy(broken))

Status = [1-(np.sum(i)/len(Broken[0])) for i in Broken]

EOut,M = [],[]
eout = np.zeros(len(TO))
tt=-1
minOut = 4 #MW
maxOut = 6 #MW Mittelwert bis 2022: 5.3MW, danach 9.5-15MW
#Repowering
minOutRe = 4
maxOutRe = 6
#Decomissioning
minOutDe = 4
maxOutDe = 6


for t in date_list:
    if t > eol:
        minOutRe = min(minOutRe+0.05,6)
        maxOutRe = min(maxOutRe+0.05,9)
    if t > eol and t <= Rebuild:
        minOutDe = max(minOutDe-0.15,0)
        maxOutDe = max(maxOutDe-0.15,0)
    if t > Rebuild:
        minOutDe = min(minOutDe+0.15,7)
        maxOutDe = min(maxOutDe+0.15,10)

    tt = tt+1
    for i in TO:
        if Broken[tt][i] == 0: #läuft
            if TT[i] == 'Turbine':
                if TO[i] < numTurbinesPerOSS:
                    if Mode == 'Exten':
                    #Extension, Repowering and Decommissioning: Select here:
                    #Extension:
                        eout[i] = np.random.uniform(minOut,maxOut,size=1) 
                        tdegPlot = tdeg
                    #Repowering:
                    elif Mode == 'Repow':
                        eout[i] = np.random.uniform(minOutRe,maxOutRe,size=1) 
                        tdegPlot = tdeg
                    #Decommissioning:
                    else:
                        eout[i] = np.random.uniform(minOutDe,maxOutDe,size=1) 
                        tdegPlot = tdegDe
                else:
                    eout[i] = np.random.uniform(minOut,maxOut,size=1) 
        if Broken[tt][i] == 1: #broken
            if TT[i] == 'Turbine':
                eout[i] = 0
            if TT[i] == 'OSS1':
                eout[OSS['OSS1']] = 0
            elif TT[i] == 'OSS2':
                eout[OSS['OSS2']] = 0
            else: 
                eout[OSS['OSS3']] = 0
    EOut.append(copy.deepcopy(eout))
    M.append([minOutRe,maxOutRe])
                
Energy = [np.sum(i) for i in EOut]        
    
#############################################################################

if Mode == 'Decom':
    t = tdegDe
else:
    t = tdeg

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)

plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.plot(date_list, Status, color='k',label='Load') #semilogx
plt.plot([eol,eol],[min(Status),max(Status)],color='grey',label='H maximal')
plt.plot([date_list[t],date_list[t]], [min(Status),max(Status)],color='k',label='H maximal')
plt.xlabel('Time', fontsize=FS)
plt.ylabel('Status', fontsize=FS)

name = 'Status.png'
fig.savefig(name, dpi=100)



fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)

plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.plot(date_list, Energy, color='k',label='Load') #semilogx
plt.plot([eol,eol],[min(Energy),max(Energy)],color='grey',label='H maximal')
plt.plot([date_list[t],date_list[t]], [min(Energy),max(Energy)],color='k',label='H maximal')
plt.xlabel('Time', fontsize=FS)
plt.ylabel('Energy [MW]', fontsize=FS)

name = 'Energy.png'
fig.savefig(name, dpi=100)

AreaE = np.trapz(Energy)
print('Area: ', AreaE)
