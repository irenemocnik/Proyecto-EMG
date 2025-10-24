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

emgN = sosfiltfilt(bpSosButter, emgNeuro)
emgM = sosfiltfilt(bpSosButter, emgMyo)

from scipy.signal import medfilt

# Filtro no lineal (mediana)
tamaño_ventana = 201  # 5 ms de ventana
bajas = medfilt(emgHealthy, kernel_size=tamaño_ventana)
emgH = emgHealthy - bajas

# def envRMS(senal, fs, largoVentana):
#     N = int(fs * largoVentana / 1000)
#     senal = senal**2 #Energía
#     rms = np.convolve(senal, np.ones(N)/N, mode = 'same') #Convolución entre una ventana rectangular y la senal
#     return np.sqrt(rms)

# envolvente = envRMS(emgH, fs = fs, largoVentana = 50)
# envolventeN = envRMS(emgN, fs = fs, largoVentana = 50) 
# envolventeM = envRMS(emgM, fs = fs, largoVentana = 50) 

#Zonas de energía baja



#altura = 0.5 * np.max(emgH)
altura = 0.25
picos, _ = find_peaks(emgHealthy, height=altura, distance=int(0.01*fs))



tLB = []
vLB = []
vLB2 = []
vLB3 = []

for i in range(1, len(picos)):
    pico = picos[i]
    picoPrev = picos[i - 1]
    distancia = pico - picoPrev
    
    punto = int(picoPrev + 0.3 * distancia)
    mediaVentana = int(0.2 * distancia)
    inicio = punto - mediaVentana
    final = punto + mediaVentana

  
    if inicio < 0 or final > len(emgH):
            continue

  
    ventana = emgH[inicio:final]
    promedio = np.mean(ventana)
    mediana = np.median(ventana)

    tLB.append(punto)
    vLB.append(promedio)
    vLB2.append(mediana)
    vLB3.append(punto)
    
tiemposLB = []
valoresLB = []
valoresLB2 = []
valoresLB3 = []

for i in range(1, len(picos)):
    pico = picos[i]
    picoPrev = picos[i - 1]
    frac = [0.2, 0.5, 0.8]
    distancia = pico - picoPrev
        
    for p in frac:
        punto = int(picoPrev + p * distancia)
        mediaVentana = int(0.3 * distancia)
        inicio = punto - mediaVentana
        final = punto + mediaVentana

      
        if inicio < 0 or final > len(emgHealthy):
            continue

      
        ventana = emgHealthy[inicio:final]
        promedio = np.mean(ventana)
        mediana = np.median(ventana)

        tiemposLB.append(punto)
        valoresLB.append(promedio)
        valoresLB2.append(mediana)
        valoresLB3.append(punto)  


tiempoT = np.arange(len(emgH))
sProm = interp1d(tLB, vLB, kind='cubic', fill_value='extrapolate')
sMed = interp1d(tLB, vLB2, kind='cubic', fill_value='extrapolate')

lBase = sProm(tiempoT)
lBase2 = sMed(tiempoT)

splineProm = interp1d(tiemposLB, valoresLB, kind='cubic', fill_value='extrapolate')
splineMed = interp1d(tiemposLB, valoresLB2, kind='cubic', fill_value='extrapolate')

lineaBase = splineProm(tiempoT)
lineaBase2 = splineMed(tiempoT)

# Corrección de la EMG
# emgFiltrada = emgHealthy - lineaBase #usa el prom
emgInter = emgHealthy - lineaBase2 #usa la mediana

t = np.arange(len(emgHealthy)) / fs

# picos + señal
plt.figure(figsize=(12, 5))
plt.plot(t[picos], emgHealthy[picos], 'rx', label='Picos detectados', markersize=8)
plt.plot(t, emgHealthy, label='EMG original', alpha=0.5)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Detección de picos en señal EMG saludable")
plt.legend()
plt.xlim(3.4, 4.5)
plt.ylim(-1, 2)
plt.grid(True)
plt.tight_layout()
plt.show()


#picos + señal + puntos 3 ms antes de cada pico

plt.figure(figsize=(12, 5))
plt.plot(t[picos], emgHealthy[picos], 'rx', label='Picos detectados', markersize=8)
plt.plot(t, emgHealthy, label='EMG original', alpha=0.5)
plt.plot(t[vLB3], emgHealthy[vLB3], 'mx', label='Puntos 3 ms previos a cada pico', markersize=8)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Detección de picos en señal EMG saludable")
plt.legend()
plt.xlim(3.4, 4.5)
plt.ylim(-1, 2)
plt.grid(True)
plt.tight_layout()
plt.show()

#picos, señal, puntos, lbmed, lbmean
plt.figure(figsize=(12, 5))
plt.plot(t[picos], emgHealthy[picos], 'rx', label='Picos detectados', markersize=8)
plt.plot(t, emgHealthy, label='EMG original', alpha=0.5)
plt.plot(t[vLB3], emgHealthy[vLB3], 'mx', label='Puntos 3 ms previos a los picos', markersize=8)
plt.plot(t, lBase, label='Línea de base (media)', color='green')
plt.plot(t, lBase2, label='Línea de base (mediana)', color='blue')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Línea de base por interpolación de media y mediana")
plt.legend()
plt.xlim(3.4, 4.5)
plt.ylim(-1, 2)
plt.grid(True)
plt.tight_layout()
plt.show()

#picos, señal, mas puntos, mejor lbmed
plt.figure(figsize=(12, 5))
plt.plot(t[picos], emgHealthy[picos], 'rx', label='Picos detectados', markersize=8)
plt.plot(t, emgHealthy, label='EMG original', alpha=0.5)
plt.plot(t[valoresLB3], emgHealthy[valoresLB3], 'mx', label='Puntos 2, 5 y 8 ms previos a los picos', markersize=8)
plt.plot(t, lineaBase2, label='Línea de base (mediana)', color='blue')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Linea de Base por Interpolación de Mediana")
plt.legend()
plt.xlim(3.4, 4.5)
plt.ylim(-1, 2)
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
plt.plot(t, emgHealthy, label='EMG original', alpha=0.5)
plt.plot(t, emgH, label='EMG filtrado de mediana', alpha=0.5)
# plt.plot(t, emgFiltrada, label='EMG corregida (MEDIA)', color='black')
plt.plot(t, emgInter, label='EMG filtrado por Interpolación', color='grey')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Filtro de mediana y filtro por interpolación")
plt.legend()
plt.xlim(3.8, 4.4)
plt.ylim(-1, 2)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(t, emgHealthy, label='EMG original', alpha=0.5)
plt.plot(t, emgH, label='EMG filtrado de mediana', alpha=0.5)
# plt.plot(t, emgFiltrada, label='EMG corregida (MEDIA)', color='black')
plt.plot(t, emgInter, label='EMG filtrado por Interpolación', color='grey')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Filtro de mediana y filtro por interpolación")
plt.legend()
plt.xlim(11.2, 12.0)
plt.ylim(-1, 2)
plt.grid(True)
plt.tight_layout()
plt.show()

# from scipy.signal import welch

# nperseg = len(emgHealthy) // 20
# foriginal, welchOriginal = welch(emgHealthy, fs=fs, window='blackman',
#                      nperseg= nperseg, noverlap=nperseg // 2, scaling='density')
# finter, welchInterpolado = welch(emgInter, fs=fs, window='blackman',
#                          nperseg= nperseg, noverlap= nperseg // 2, scaling='density')
# fmed,welchMed = welch(emgH, fs=fs, window='blackman',
#                          nperseg= nperseg, noverlap= nperseg // 2, scaling='density')


# # fOrigN, welchOrigN, fFiltN, welchFiltN = PSDwelch(emgNeuro, emgFiltN)
# # fOrigM, welchOrigM, fFiltM, welchFiltM = PSDwelch(emgMyo, emgFiltM)

# nyq = fs // 2
# ws1 = 1
# wp1 = 20
# wp2 = 480
# ws2 = 600
# ripple = 1
# atenuacion = 40 
# frecs = np.array([0.0, ws1, wp1, wp2, ws2, nyq]) / nyq
# gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
# gains = 10**(gains / 20)

# bpSosButter = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq, ws=np.array([ws1, ws2]) / nyq, gpass = 1.0, gstop = 40., analog=False, ftype = 'butter', output = 'sos')
# emgIIR = sosfiltfilt(bpSosButter, emgHealthy)
# fIIR, welchIIR = welch(emgIIR, fs=fs, window='blackman',
#                          nperseg= nperseg, noverlap= nperseg // 2, scaling='density')
# import pywt
# wavelet = 'dmey'
# niveles = 7
# coeffs = pywt.wavedec(emgHealthy, wavelet, level=niveles)
# # coeffs = [cA6, cD6, cD5, ..., cD1]

# coeffs[0] = np.zeros_like(coeffs[0])      # cA6 aprox hasta 15
# coeffs[-1] = np.zeros_like(coeffs[-1])    # cD1 (2000–1000 Hz)
# coeffs[-2] = np.zeros_like(coeffs[-2])    # cD2 (1000–500 Hz)

# filtroWavelet = pywt.waverec(coeffs, wavelet)
# filtroWavelet = filtroWavelet[:len(emgHealthy)] #saca el padding interno
# fWavelet, welchWavelet = welch(filtroWavelet, fs=fs, window='blackman',
#                          nperseg= nperseg, noverlap= nperseg // 2, scaling='density')

# plt.figure(figsize=(12, 4))
# plt.plot(foriginal, 10 * np.log10(welchOriginal), label='Original', alpha=0.6)
# plt.plot(fmed, 10 * np.log10(welchMed), label='Filtrada (Mediana)')
# plt.plot(finter, 10 * np.log10(welchInterpolado),  label='Filtrada (Interpolación)')
# plt.plot(fIIR, 10 * np.log10(welchIIR), label='Filtrada (LINEAL)')
# # plt.plot(fWavelet, 10 * np.log10(welchWavelet), label='Filtrada (Wavelets)')

# plt.title('Comparativa de filtros - EMG saludable')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [V²/Hz]')
# plt.xlim([0, 800])
# plt.ylim([-140, 0])
# plt.grid(True, which='both')
# plt.legend()
# plt.tight_layout()
# plt.show()
