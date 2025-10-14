# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:42:51 2025

@author: iremo
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import wfdb

def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")


from scipy.signal import butter, filtfilt
fs = 4000
# Filtro pasa banda 20-450 Hz con orden 4
b, a = butter(N=4, Wn=[15, 450], btype='bandpass', fs=fs)

emgMyo = filtfilt(b, a, emgMyo)
emgNeuro = filtfilt(b, a, emgNeuro)
emgHealthy = filtfilt(b, a, emgHealthy)

def psd(sig, fs):
    N = len(sig)
    T = N / fs
    df = 1 / T
    ventana = signal.windows.hamming(N)
    xxventana = sig*ventana
    desvio=np.std(ventana)
    xxventananormalizada=xxventana/desvio
    
    ft = 1/N*np.fft.fft(sig)
    ft_XVENTANA = 1/N*np.fft.fft( xxventananormalizada)
    
    ff = np.linspace(0, (N-1)*df, N)
    bfrec = ff <= fs/2 #hasta nyquist, por la simetría de la fft, lo demás es redundante.
    return ff, bfrec, ft_XVENTANA

def recortar(sig, fs, t0, t1=None):
    i0 = int(round(t0 * fs))
    i1 = int(round(t1 * fs))
    recortada = sig[i0:i1]
    return recortada

emgHealthy1 = recortar(emgHealthy, fs, 0.0, 3.0)    
emgHealthy2 = recortar(emgHealthy, fs, 3.0, 4.5)
emgHealthy3 = recortar(emgHealthy, fs, 4.5, 12.0)   

emgNeuro1 = recortar(emgNeuro, fs, 0.0, 11.0)
emgNeuro2 = recortar(emgNeuro, fs, 11.0, 18.0)
emgNeuro3 = recortar(emgNeuro, fs, 19.0, 30.0) 
emgNeuro4 = recortar(emgNeuro, fs, 19.0, 35.0)

emgMyo1 = recortar(emgMyo, fs, 0.0, 12.5)
emgMyo2 = recortar(emgMyo, fs, 12.5, 15.0)     
emgMyo3 = recortar(emgMyo, fs, 15.0, 25.0)  

ffsano1, bfrecsano1, ft_SanoV1 = psd(emgHealthy1, fs)
ffmio1,   bfrecmio1,   ft_MioV1   = psd(emgMyo1, fs)
ffneuro1, bfrecneuro1, ft_NeuroV1 = psd(emgNeuro1, fs)

ffsano2, bfrecsano2, ft_SanoV2 = psd(emgHealthy2, fs)
ffmio2,   bfrecmio2,   ft_MioV2   = psd(emgMyo2, fs)
ffneuro2, bfrecneuro2, ft_NeuroV2 = psd(emgNeuro2,fs)

ffsano3, bfrecsano3, ft_SanoV3 = psd(emgHealthy3, fs)
ffmio3,   bfrecmio3,   ft_MioV3   = psd(emgMyo3,fs)
ffneuro3, bfrecneuro3, ft_NeuroV3 = psd(emgNeuro3,fs)



fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axs[0].plot(ffsano1[bfrecsano1],  10*np.log10(2*np.abs(ft_SanoV1[bfrecsano1])**2 + 1e-12),
            color='steelblue', lw=1.5)
axs[0].set_title('Paciente Sano')
axs[0].set_xlabel('Frecuencia [Hz]')
axs[0].set_ylabel('Densidad de Potencia [dB]')
axs[0].grid(True, alpha=0.3)

axs[1].plot(ffmio1[bfrecmio1],    10*np.log10(2*np.abs(ft_MioV1[bfrecmio1])**2 + 1e-12),
            color='darkorange', lw=1.5)
axs[1].set_title('Miopatía')
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].grid(True, alpha=0.3)

axs[2].plot(ffneuro1[bfrecneuro1], 10*np.log10(2*np.abs(ft_NeuroV1[bfrecneuro1])**2 + 1e-12),
            color='seagreen', lw=1.5)
axs[2].set_title('Neuropatía')
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].grid(True, alpha=0.3)

fig.suptitle('Periodograma - Actividad Baja', fontsize=12)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axs[0].plot(ffsano2[bfrecsano2],  10*np.log10(2*np.abs(ft_SanoV2[bfrecsano2])**2 + 1e-12),
            color='steelblue', lw=1.5)
axs[0].set_title('Paciente Sano')
axs[0].set_xlabel('Frecuencia [Hz]')
axs[0].set_ylabel('Densidad de Potencia [dB]')
axs[0].grid(True, alpha=0.3)

axs[1].plot(ffmio2[bfrecmio2],10*np.log10(2*np.abs(ft_MioV2[bfrecmio2])**2 + 1e-12),
            color='darkorange', lw=1.5)
axs[1].set_title('Miopatía')
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].grid(True, alpha=0.3)

axs[2].plot(ffneuro2[bfrecneuro2], 10*np.log10(2*np.abs(ft_NeuroV2[bfrecneuro2])**2 + 1e-12),
            color='seagreen', lw=1.5)
axs[2].set_title('Neuropatía')
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].grid(True, alpha=0.3)

fig.suptitle('Periodograma - Actividad intensa', fontsize=12)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axs[0].plot(ffsano3[bfrecsano3],  10*np.log10(2*np.abs(ft_SanoV3[bfrecsano3])**2 + 1e-12),
            color='steelblue', lw=1.5)
axs[0].set_title('Paciente Sano')
axs[0].set_xlabel('Frecuencia [Hz]')
axs[0].set_ylabel('Densidad de Potencia [dB]')
axs[0].grid(True, alpha=0.3)

axs[1].plot(ffmio3[bfrecmio3],10*np.log10(2*np.abs(ft_MioV3[bfrecmio3])**2 + 1e-12),
            color='darkorange', lw=1.5)
axs[1].set_title('Miopatía')
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].grid(True, alpha=0.3)

axs[2].plot(ffneuro3[bfrecneuro3], 10*np.log10(2*np.abs(ft_NeuroV3[bfrecneuro3])**2 + 1e-12),
            color='seagreen', lw=1.5)
axs[2].set_title('Neuropatía')
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].grid(True, alpha=0.3)

fig.suptitle('Periodograma - Reposo', fontsize=12)
plt.tight_layout()
plt.show()

# def metricas_periodograma(ff, bfrec, ft, band=(20, 450)):
    
#     f = ff[bfrec]
#     P = 2.0 * np.abs(ft[bfrec])**2                   
#     m = (f >= band[0]) & (f <= band[1])
#     f, P = f[m], P[m]
#     A = np.trapezoid(P, f) 
#     Pn = P / (A + 1e-12)

    
#     df = f[1] - f[0] if len(f) > 1 else 1.0
#     cdf = np.cumsum(Pn) * df
#     MNF = float(np.trapezoid(f * Pn, f))
#     MDF = float(f[np.searchsorted(cdf, 0.50)])
#     f_peak = float(f[np.argmax(Pn)])
#     F5  = float(f[np.searchsorted(cdf, 0.05)])
#     F95 = float(f[np.searchsorted(cdf, 0.95)])
#     bandwidth = float(F95 - F5)
   

#     bands = np.array([[20, 60], [60, 150], [150, 350]])
#     bp = np.array([np.trapezoid(Pn[(f>=a)&(f<=b)], f[(f>=a)&(f<=b)]) for a, b in bands])
#     ratios = {"B2/B1": float(bp[1]/(bp[0]+1e-12)), "B3/B2": float(bp[2]/(bp[1]+1e-12))}

#     return {
#         "MNF": MNF, "MDF": MDF, "f_peak": f_peak,
#         "F5": F5, "F95": F95, "bandwidth": bandwidth,
#         "bands": bands, "band_powers": bp, "ratios": ratios
#     }


# met_s = metricas_periodograma(ffsano1,  bfrecsano1,  ft_SanoV1,  band=(20, 450))
# met_m = metricas_periodograma(ffmio1,   bfrecmio1,   ft_MioV1,   band=(20, 450))
# met_n = metricas_periodograma(ffneuro1, bfrecneuro1, ft_NeuroV1, band=(20, 450))

# print("Sano      :", met_s)
# print("Miopatía  :", met_m)
# print("Neuropatía:", met_n)
