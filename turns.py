# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 02:51:25 2025

@author: iremo
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb

import pywt

fs = 4000
def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")

recortada = emgMyo[0:int(16.75*fs)]
recorte2 = emgMyo[int(17*fs):int(26.5*fs)]

# Concatenarlos
pegada = np.concatenate((recortada, recorte2))
trec = np.arange(len(pegada)) / fs


from scipy.signal import butter, filtfilt

# Filtro pasa banda 20-450 Hz con orden 4
b, a = butter(N=4, Wn=[15, 450], btype='bandpass', fs=fs)

emgMyo = filtfilt(b, a, pegada)
emgNeuro = filtfilt(b, a, emgNeuro)
emgHealthy = filtfilt(b, a, emgHealthy)

def amplitud(senal, segmento = 250):
    largo = int(segmento * fs / 1000)
    valores = []
    
    for inicio in range(0, len(senal) - largo, largo):
        seg = senal[inicio:inicio + largo]
        turns = []
        for i in range(1,len(seg)-1):
            if (seg[i] > seg[i-1] and seg[i] > seg[i+1]) or \
                (seg[i] < seg[i-1] and seg[i] < seg[i+1]):
                turns.append(i)
                
        amp = [abs(seg[turns[i+1]] - seg[turns[i]])
                for i in range(len(turns) - 1)]
    
        if len(amp) > 0:
            valores.append([len(turns), np.mean(amp)])
        else:
            valores.append([len(turns), 0])

    return np.array(valores)   

TurnsAH = amplitud(emgHealthy) 
TurnsAM = amplitud(emgMyo)  
TurnsAN = amplitud(emgNeuro) 

#segmentos iniciales

emgH1 = emgHealthy[0:int(3*fs)]
emgH2 = emgHealthy[int(3.2*fs): int(4.5*fs)]
emgH3 = emgHealthy[int(5*fs): int(12*fs)]

emgM1 = emgMyo[int(2*fs):int(12.5*fs)]
emgM2 = emgMyo[int(12.5*fs): int(15*fs)]
emgM3 = pegada[int(15*fs): int(22*fs)]

emgN1 = emgNeuro[0:int(10*fs)]
emgN2 = emgNeuro[int(12.5*fs): int(17*fs)]
emgN3 = emgNeuro[int(20*fs): int(27*fs)]

turnsH1 = amplitud(emgH1)
turnsH2 = amplitud(emgH2)
turnsH3 = amplitud(emgH3)

turnsM1 = amplitud(emgM1)
turnsM2 = amplitud(emgM2)
turnsM3 = amplitud(emgM3)

turnsN1 = amplitud(emgN1)
turnsN2 = amplitud(emgN2)
turnsN3 = amplitud(emgN3)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

# Contraccion debil
axes[0].scatter(turnsH1[:, 0], turnsH1[:, 1], label='Sano', alpha=0.7)
axes[0].scatter(turnsN1[:, 0], turnsN1[:, 1], label='Neuropatía', alpha=0.7)
axes[0].scatter(turnsM1[:, 0], turnsM1[:, 1], label='Miopatía', alpha=0.7)
axes[0].set_title('Periodo 1: Contracción débil')
axes[0].set_xlabel('Turns')
axes[0].set_ylabel('Amplitud (µV)')
axes[0].grid(True)

# contraccion fuerte
axes[1].scatter(turnsH2[:, 0], turnsH2[:, 1], alpha=0.7)
axes[1].scatter(turnsN2[:, 0], turnsN2[:, 1], alpha=0.7)
axes[1].scatter(turnsM2[:, 0], turnsM2[:, 1], alpha=0.7)
axes[1].set_title('Segmento 2: Contracción fuerte')
axes[1].set_xlabel('Turns')
axes[1].grid(True)

# relajacion
axes[2].scatter(turnsH3[:, 0], turnsH3[:, 1], alpha=0.7)
axes[2].scatter(turnsN3[:, 0], turnsN3[:, 1], alpha=0.7)
axes[2].scatter(turnsM3[:, 0], turnsM3[:, 1], alpha=0.7)
axes[2].set_title('Segmento 3: Relajación')
axes[2].set_xlabel('Turns')
axes[2].grid(True)

axes[0].legend()
plt.tight_layout()
plt.show()

     








