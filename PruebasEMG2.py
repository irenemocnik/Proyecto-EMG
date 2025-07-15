# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 21:37:34 2025

@author: iremo

Las señales son un EMG intramuscular del músculo tibialis anterior.
El procedimiento consiste en la dorsiflexión suave del pie contra resistencia, seguido de relajación.

----------------------------------------------------------------------------------------------------

Datos EMG - Healthy
1 canal,
fs = 4000 Hz,
cantMuestras = 50860 (12 segundos aprox)
El paciente no tiene antecedentes de enfermedades neuromusculares.

----------------------------------------------------------------------------------------------------

Datos EMG - Neuropathy
1 canal, 
fs = 4000 Hz, 
cantMuestras = 147858 (aprox 35 segundos)
El paciente en cuestión tiene neuropatía por radiculopatía L5.

----------------------------------------------------------------------------------------------------

Datos EMG – Myopathy
1 canal, 
fs = 4000 Hz, 
cantMuestras = 110337 (aprox 27 segundos aprox)
El paciente tiene diagnostico de miopatía por una historia de polimiositis prolongada,
esta bajo tratamiento con esteroides y metotrexato en baja dosis.


"""

#%%


import wfdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.signal import sosfiltfilt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")

fs = 4000 # Hz
nyq = fs // 2
ws1 = 1
wp1 = 20
wp2 = 480
ws2 = 600
ripple = 1
atenuacion = 40 

frecs = np.array([0.0, ws1, wp1, wp2, ws2, nyq]) / nyq
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains / 20)

bpSosButter = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq, ws=np.array([ws1, ws2]) / nyq, gpass = 1.0, gstop = 40., analog=False, ftype = 'butter', output = 'sos')

emgiltfilt = sosfiltfilt(bpSosButter, emgHealthy)
emgH = emgHealthy
emgN = sosfiltfilt(bpSosButter, emgNeuro)
emgM = sosfiltfilt(bpSosButter, emgMyo)

from scipy.signal import medfilt

# Filtro no lineal (mediana)
tamaño_ventana = int(0.003 * fs)  # 5 ms de ventana
if tamaño_ventana % 2 == 0:
    tamaño_ventana += 1  # Debe ser impar

emgH = medfilt(emgHealthy, kernel_size=tamaño_ventana)


def envRMS(senal, fs, largoVentana):
    N = int(fs * largoVentana / 1000)
    senal = senal**2 #Energía
    rms = np.convolve(senal, np.ones(N)/N, mode = 'same') #Convolución entre una ventana rectangular y la senal
    return np.sqrt(rms)

envolvente = envRMS(emgH, fs = fs, largoVentana = 50)
envolventeN = envRMS(emgN, fs = fs, largoVentana = 50) 
envolventeM = envRMS(emgM, fs = fs, largoVentana = 50) 

#Zonas de energía baja

tantes = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04,0.045,0.05, 0.055] #aprox 30ms antes de cada pico
dt = 0.015

#altura = 0.5 * np.max(emgH)
altura = 0.15
picos, _ = find_peaks(emgH, height=altura, distance=int(0.05*fs))

tiemposLB = []
valoresLB = []
valoresLB2 = []

for pico in picos:
    for t in tantes:
        medio = int(pico - t * fs)
        inicio = medio - int(dt * fs)
        final = medio + int(dt * fs)
    
        if inicio < 0 or final > len(emgH):
            continue
        promedio = np.mean(emgH[inicio:final])
        mediana = np.median(emgH[inicio:final])
    
        tiemposLB.append(medio)
        valoresLB.append(promedio)
        valoresLB2.append(mediana)
    
spline = interp1d(tiemposLB, valoresLB, kind='linear', fill_value='extrapolate')
tiempoT = np.arange(len(emgH))
spline2 = interp1d(tiemposLB, valoresLB2, kind='linear', fill_value='extrapolate')
lineaBase = spline(tiempoT)
lineaBase2 =spline2(tiempoT)

emgFiltrada = emgHealthy - lineaBase
emgFiltrada2 = emgHealthy - lineaBase2

t = np.arange(len(emgHealthy)) / fs
plt.figure(figsize=(12, 5))
plt.plot(t, emgHealthy, label='EMG original', alpha=0.5)
#plt.plot(t, emgH, label='EMG original', alpha=0.5)
#plt.plot(t, lineaBase, label='Línea de base estimada', color='green')
#plt.plot(t, lineaBase2, label='Línea de base estimada2', color='blue')
plt.plot(t, emgFiltrada, label='EMG corregida', color='black')
plt.plot(t, emgFiltrada2, label='EMG corregida2', color='grey')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Estimación de línea de base desde zonas isoeléctricas previas a PAUMs")
plt.legend()
plt.xlim(3.25,4.2)
plt.ylim(-1,1)
plt.grid(True)
plt.tight_layout()
plt.show()