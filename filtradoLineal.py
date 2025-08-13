# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:48:57 2025

@author: iremo
"""

#%% Carga de señales
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from pytc2.sistemas_lineales import plot_plantilla
from scipy.signal import iirnotch, tf2sos
from scipy.signal import sosfiltfilt

def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")

fs = 4000

#%%Diseño de filtros lineales
nyq = fs // 2
ws1 = 1
wp1 = 20
wp2 = 480
ws2 = 600
ripple = 1
atenuacion = 30

frecs = np.array([0.0, ws1, wp1, wp2, ws2, nyq]) / nyq
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains / 20)

bpSosButter = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq, ws=np.array([ws1, ws2]) / nyq, gpass = 1.0, gstop =40., analog=False, ftype = 'butter', output = 'sos')
bpSosCheby = sig.iirdesign(wp=np.array([wp1, wp2]) / nyq, ws=np.array([ws1, ws2]) / nyq, gpass=1.0, gstop=40., analog=False, ftype='cheby1', output='sos')


f0 = 60
Q = 200
sos2 = tf2sos(*iirnotch(f0, Q, fs=fs))
sos4 = np.concatenate((sos2, sos2), axis=0)  #orden 4
sos8 = np.concatenate((sos4, sos4), axis=0) #orden 8
sos16 = np.concatenate((sos8, sos8), axis=0) #orden 16
sos32 = np.concatenate((sos16, sos16), axis=0) #orden 32

NN = 1024
wRad  = np.append(np.logspace(-2, 0.8, NN//4), np.logspace(0.9, 1.6, NN//4) ) #mas detalle en frecs bajas
wRad  = np.append(wRad, np.linspace(40, nyq, NN//2, endpoint=True) ) / nyq * np.pi 
wRadZoom = np.linspace(50, 70, 10000) * 2 * np.pi / fs #para hacer el zoom en el notch
fZoom = wRadZoom * fs / (2 * np.pi)

w, hButter = sig.sosfreqz(bpSosButter, worN=wRad)
_, hCheby = sig.sosfreqz(bpSosCheby, worN=wRad)
_, hNotch2 = sig.sosfreqz(sos2, worN=wRadZoom)
_, hNotch8 = sig.sosfreqz(sos8, worN=wRadZoom)
_, hNotch32 = sig.sosfreqz(sos32, worN=wRadZoom)

f = w * fs / (2 * np.pi)

# plt.figure(figsize=(10, 5))
# plt.plot(f, 20 * np.log10(np.abs(hButter) + 1e-10), label=f'Butter (orden {bpSosButter.shape[0]})')
# plt.plot(f, 20 * np.log10(np.abs(hCheby) + 1e-10), label=f'Cheby1 (orden {bpSosCheby.shape[0]})')
# plt.title('Filtros Butter y Cheby')
# plt.xlabel('Frequencia [Hz]')
# plt.ylabel('Modulo [dB]')
# plt.grid()
# plt.axis([0, 1000, -60, 5 ]);
# plot_plantilla(filter_type = 'bandpass', fpass=[wp1, wp2], ripple=ripple, fstop=[ws1, ws2], attenuation=atenuacion, fs=fs)

# plt.figure(figsize=(10, 5))
# plt.plot(fZoom, 20 * np.log10(np.abs(hNotch2) + 1e-10), label='Orden 2', linewidth=1.5)
# plt.plot(fZoom, 20 * np.log10(np.abs(hNotch8) + 1e-10), label='Orden 8', linewidth=1.5)
# plt.plot(fZoom, 20 * np.log10(np.abs(hNotch32) + 1e-10), label='Orden 32', linewidth=1.5)
# plt.title('Zoom en Notch 60 Hz')
# plt.xlabel('Frequencia [Hz]')
# plt.ylabel('Modulo [dB]')
# plt.grid()
# plt.axis([55, 65, -80, 5 ]);
# plot_plantilla(filter_type = 'bandstop', fpass=[58, 62], ripple= 0, fstop=[59.99, 60.01], attenuation= 40, fs=fs)

sosButterNotch = np.concatenate((sos2, bpSosButter), axis=0) 

# wTotal = np.unique(np.concatenate((wRad, wRadZoom))) #unique ordena y elimina duplicados entre 50 y 70
# fTotal = wTotal * fs / (2 * np.pi)
# w, hFinal = sig.sosfreqz(sosButterNotch, worN=wTotal)
# plt.figure(figsize=(10, 5))
# plt.plot(fTotal, 20 * np.log10(np.abs(hFinal) + 1e-10))
# plt.title('Filtro combinado: pasabanda + notch 60 Hz')
# plt.xlabel('Frequencia [Hz]')
# plt.ylabel('Modulo [dB]')
# plt.grid()
# plt.xlim([0, 800])
# plt.ylim([-50, 5])
# plt.tight_layout()
# plot_plantilla(filter_type = 'bandpass', fpass=[wp1, wp2], ripple=ripple, fstop=[ws1, ws2], attenuation=atenuacion, fs=fs)

#%%Filtrado

emgFiltH = sosfiltfilt(sosButterNotch, emgHealthy)

# plt.figure(figsize=(12, 4))
# plt.plot(emgHealthy, label='Original', alpha=0.5)
# plt.plot(emgFiltH, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.title('EMG saludable: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([4500, 6000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(emgHealthy, label='Original', alpha=0.5)
# plt.plot(emgFiltH, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.title('EMG saludable: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([39000, 41000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(emgHealthy, label='Original', alpha=0.5)
# plt.plot(emgFiltH, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.title('EMG saludable: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([47000, 48000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

emgFiltN = sosfiltfilt(sosButterNotch, emgNeuro)

# plt.figure(figsize=(12, 4))
# plt.plot(emgNeuro, label='Original', alpha=0.5)
# plt.plot(emgFiltN, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.title('EMG con neuropatía: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([45000, 48000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(emgNeuro, label='Original', alpha=0.5)
# plt.plot(emgFiltN, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.title('EMG con neuropatía: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([48000, 51000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(emgNeuro, label='Original', alpha=0.5)
# plt.plot(emgFiltN, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.title('EMG con neuropatía: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([76000, 80000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

emgFiltM = sosfiltfilt(sosButterNotch, emgMyo)

# plt.figure(figsize=(12, 4))
# plt.plot(emgMyo, label='Original', alpha=0.5)
# plt.plot(emgFiltM, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.title('EMG con miopatía: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([71000,72000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(emgMyo, label='Original', alpha=0.5)
# plt.plot(emgFiltM, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.title('EMG con miopatía: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([105000,107000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

from scipy.signal import welch
def PSDwelch(senal, senalFiltrada):
    foriginal, welchOriginal = welch(senal, fs=fs, window='blackman',
                         nperseg=1024, noverlap=512, scaling='density')
    fFiltrado, welchFiltrado = welch(senalFiltrada, fs=fs, window='blackman',
                         nperseg=1024, noverlap=512, scaling='density')
    return foriginal, welchOriginal, fFiltrado, welchFiltrado

fOrigH, welchOrigH, fFiltH, welchFiltH = PSDwelch(emgHealthy, emgFiltH)
fOrigN, welchOrigN, fFiltN, welchFiltN = PSDwelch(emgNeuro, emgFiltN)
fOrigM, welchOrigM, fFiltM, welchFiltM = PSDwelch(emgMyo, emgFiltM)

# plt.figure(figsize=(12, 4))
# plt.semilogy(fOrigH, welchOrigH, label='Original', alpha=0.6)
# plt.semilogy(fFiltH, welchFiltH, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz')
# plt.title('Densidad espectral de potencia - EMG saludable')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [V²/Hz]')
# plt.xlim([0, 800])
# plt.grid(True, which='both')
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.semilogy(fOrigN, welchOrigN, label='Original', alpha=0.6)
# plt.semilogy(fFiltN, welchFiltN, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz')
# plt.title('Densidad espectral de potencia - EMG con neuropatía')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [V²/Hz]')
# plt.xlim([0, 800])
# plt.grid(True, which='both')
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.semilogy(fOrigM, welchOrigM, label='Original', alpha=0.6)
# plt.semilogy(fFiltM, welchFiltM, label='Filtrada (Butter + Notch)', linewidth=1.2)
# plt.axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz')
# plt.title('Densidad espectral de potencia - EMG con miopatía')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [V²/Hz]')
# plt.xlim([0, 800])
# plt.grid(True, which='both')
# plt.legend()
# plt.tight_layout()
# plt.show()

emgH = sosfiltfilt(bpSosButter, emgHealthy) #filtrado sin el notch
emgN = sosfiltfilt(bpSosButter, emgNeuro)
emgM = sosfiltfilt(bpSosButter, emgMyo)

# plt.figure(figsize=(12, 4))
# plt.plot(emgNeuro, label='Original', alpha=0.5)
# plt.plot(emgN, label='Filtrada (Butter)', linewidth=1.2)
# plt.title('EMG con neuropatía: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([76000, 80000])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

from scipy.signal import butter, filtfilt

emg = emgMyo
# Filtro pasa banda 20-450 Hz con orden 4
b, a = butter(N=4, Wn=[15, 450], btype='bandpass', fs=fs)
emgButter = filtfilt(b, a, emgMyo)
nperseg = len(emgHealthy) // 20
ff, welchb = welch(emgButter, fs=fs, window='blackman',
                          nperseg= nperseg, noverlap= nperseg // 2, scaling='density')
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(emgButter)) / fs, emgButter, label='Filtrada (Butter orden 4)', linewidth=1.2)
plt.show()


# plt.figure(figsize=(12, 4))
# plt.plot(emgHealthy, label='Original', alpha=0.5)
# plt.plot(emgButter, label='Filtrada (Butter orden 4)', linewidth=1.2)
# plt.plot(emgH, label='Filtrada (Butter diseño previo)', linewidth=1.2)
# plt.title('EMG saludable: antes y después del filtrado')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud (mV)')
# plt.xlim([39000, 39750])
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(12, 4))

# plt.plot(ff, 10 * np.log10(welchb), label='Filtrada (Mediana)')

# plt.title('Comparativa de filtros - EMG saludable')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [V²/Hz]')
# plt.xlim([0, 800])
# plt.ylim([-140, 0])
# plt.grid(True, which='both')
# plt.legend()
# plt.tight_layout()
# plt.show()