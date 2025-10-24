# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal.windows import dpss
import pandas as pd
from numpy.fft import rfft, rfftfreq

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
 
    
def psdMultitaper(sig, fs, anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5):
    #promedio de tapers y epocas temporales
    
    N = sig.size
    L = int(round(largoEpoch * fs))
    H = int(round(epochPaso * fs))
    inicios = np.arange(0, N - L + 1, H, dtype = int)
    
    NW = float(anchotiempo)
    K = int(np.floor(2*NW - 1)) #nro de tapers validos, saca extremos
    
    tapers = dpss(L, NW = NW, Kmax = K, sym = False) #dpss devuelve matriz K x N
    Etaper = np.sum(tapers**2, axis = 1)
    
    n_fft = 1 << (L - 1).bit_length() #redondea a potencia de 2
    freqs = rfftfreq(n_fft, d=1.0/fs)
    #normalizar por energia del taper y fs
    
    acumulaPSD = np.zeros(freqs.size, dtype = float)
    nepoch = 0
    
    for i in inicios:
        epoch = sig[i:i+L]
        epoch = epoch - np.mean(epoch)
        epW = tapers * epoch
        X = rfft(epW, n = n_fft, axis = 1)
        S = (np.abs(X)**2) / (fs * Etaper[:, None]) #matriz K x nfreqs
        Sprom = S.mean(axis=0)
        acumulaPSD = acumulaPSD + Sprom
        nepoch = nepoch + 1
        
    promPSD = acumulaPSD / nepoch
    return freqs, promPSD

    
    



def recortar(sig, fs, t0, t1=None):
    i0 = int(round(t0 * fs))
    i1 = int(round(t1 * fs))
    recortada = sig[i0:i1]
    return recortada

emgHealthy1 = recortar(emgHealthy, fs, 0.0, 3.0)    
emgHealthy2 = recortar(emgHealthy, fs, 3.4, 4.6)
emgHealthy3 = recortar(emgHealthy, fs, 5.0, 12.0)   

emgNeuro1 = recortar(emgNeuro, fs, 5.0, 8.0)
emgNeuro2 = recortar(emgNeuro, fs, 11.9, 13.1)
emgNeuro3 = recortar(emgNeuro, fs, 20.0, 27.0) 


emgMyo1 = recortar(emgMyo, fs, 6.5, 9.5)
emgMyo2 = recortar(emgMyo, fs, 12.4, 13.6)     
emgMyo3 = recortar(emgMyo, fs, 15.0, 22.0)  


f_s1, P_s1 = psdMultitaper(emgHealthy1, fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)
f_m1, P_m1= psdMultitaper(emgMyo1,     fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)
f_n1, P_n1= psdMultitaper(emgNeuro1,   fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)

f_s2, P_s2 = psdMultitaper(emgHealthy2, fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)
f_m2, P_m2 = psdMultitaper(emgMyo2,     fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)
f_n2, P_n2 = psdMultitaper(emgNeuro2,   fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)

f_s3, P_s3 = psdMultitaper(emgHealthy3, fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)
f_m3, P_m3 = psdMultitaper(emgMyo3,     fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)
f_n3, P_n3 = psdMultitaper(emgNeuro3,   fs,  anchotiempo = 3.5, largoEpoch = 1.0, epochPaso = 0.5)



    

plt.figure(figsize=(8,6))
plt.plot(f_s1, 10*np.log10(P_s1 + 1e-12), label='Sano')
plt.plot(f_m1, 10*np.log10(P_m1 + 1e-12), label='Miopatía')
plt.plot(f_n1, 10*np.log10(P_n1 + 1e-12), label='Neuropatía')
plt.xlabel('Frecuencia [Hz]'); plt.ylabel('PSD [dB re W/Hz]')
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axs[0].plot(f_s1, 10*np.log10(P_s1 + 1e-12), label='Sano', color='steelblue', lw=1.5)
axs[0].set_title('Paciente Sano')
axs[0].set_xlabel('Frecuencia [Hz]')
axs[0].set_ylabel('Densidad de Potencia [dB]')
axs[0].grid(True, alpha=0.3)

axs[1].plot(f_m1, 10*np.log10(P_m1 + 1e-12), label='Miopatía', color='darkorange', lw=1.5)
axs[1].set_title('Miopatía')
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].grid(True, alpha=0.3)

axs[2].plot(f_n1, 10*np.log10(P_n1 + 1e-12), label='Neuropatía',color='seagreen', lw=1.5)
axs[2].set_title('Neuropatía')
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].grid(True, alpha=0.3)

fig.suptitle('Mutitaper - Actividad Baja', fontsize=12)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axs[0].plot(f_s2, 10*np.log10(P_s2 + 1e-12), label='Sano',color='steelblue', lw=1.5)
axs[0].set_title('Paciente Sano')
axs[0].set_xlabel('Frecuencia [Hz]')
axs[0].set_ylabel('Densidad de Potencia [dB]')
axs[0].grid(True, alpha=0.3)

axs[1].plot(f_m2, 10*np.log10(P_m2 + 1e-12), label='Miopatía', color='darkorange', lw=1.5)
axs[1].set_title('Miopatía')
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].grid(True, alpha=0.3)

axs[2].plot(f_n2, 10*np.log10(P_n2 + 1e-12), label='Neuropatía',color='seagreen', lw=1.5)
axs[2].set_title('Neuropatía')
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].grid(True, alpha=0.3)

fig.suptitle('Mutitaper - Actividad intensa', fontsize=12)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axs[0].plot(f_s3, 10*np.log10(P_s3 + 1e-12), label='Sano', color='steelblue', lw=1.5)
axs[0].set_title('Paciente Sano')
axs[0].set_xlabel('Frecuencia [Hz]')
axs[0].set_ylabel('Densidad de Potencia [dB]')
axs[0].grid(True, alpha=0.3)

axs[1].plot(f_m3, 10*np.log10(P_m3 + 1e-12), label='Miopatía', color='darkorange', lw=1.5)
axs[1].set_title('Miopatía')
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].grid(True, alpha=0.3)

axs[2].plot(f_n3, 10*np.log10(P_n3 + 1e-12), label='Neuropatía',color='seagreen', lw=1.5)
axs[2].set_title('Neuropatía')
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].grid(True, alpha=0.3)

fig.suptitle('Mutitaper - Reposo',fontsize=12)
plt.tight_layout()
plt.show()


def MDF(f,psd):
    suma=np.cumsum(psd)
    medio = suma[-1] / 2.0
    imedio = np.where(suma >= medio)[0][0]
    f0,f1 = f[imedio-1], f[imedio]
    s0,s1 = suma[imedio-1],suma[imedio]
    w = (medio-s0) / (s1 - s0 + 1e-15)
    return float(f0+w*(f1-f0)) #interpolacion lineal

def metricas(psd, f, fs):
    df = fs / len(psd)
    areatot = np.sum(psd)
    psdnormalizado = psd / areatot
    areacum = np.cumsum(psdnormalizado)
    
    indice_95 = np.where(areacum >= 0.95)[0][0]
    freq95 = f[indice_95]
    
    fmean = np.sum(f * psd) / areatot
    fmediana = MDF(f, psd)
    energiatotal = areatot * df # V2/HZ a V2
    
    return fmean, areacum, fmediana, freq95, energiatotal

fmeanS1, eacumS1, fmedS1, freq95S1, ES1 = metricas(P_s1, f_s1, fs)
fmeanS2, eacumS2, fmedS2,  freq95S2, ES2 = metricas(P_s2, f_s2, fs)
fmeanS3, eacumS3, fmedS3,  freq95S3, ES3 = metricas(P_s3, f_s3, fs)
fmeanN1, eacumN1, fmedN1,  freq95N1, EN1 = metricas(P_n1, f_n1, fs)
fmeanN2,eacumN2, fmedN2,  freq95N2, EN2 = metricas(P_n2, f_n2, fs)
fmeanN3,eacumN3, fmedN3,  freq95N3, EN3 = metricas(P_n3, f_n3, fs)
fmeanM1,eacumM1, fmedM1, freq95M1, EM1 = metricas(P_m1, f_m1, fs)
fmeanM2,eacumM2, fmedM2, freq95M2, EM2 = metricas(P_m2, f_m2, fs)
fmeanM3, eacumM3,fmedM3, freq95M3, EM3 = metricas(P_m3, f_m3, fs)

def show_table_figure(df: pd.DataFrame, title: str, fontsize=12):
    fig, ax = plt.subplots(figsize=(10, 2 + 0.35*len(df)))
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=14)
    tbl = ax.table(cellText=np.round(df.values, 4),
                    colLabels=df.columns,
                    rowLabels=df.index,
                    loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.3)
    fig.tight_layout()
    plt.show()

#Actividad leve
df_leve = pd.DataFrame(
    {
        "Fmean [Hz]":  [float(fmeanM1), float(fmeanS1),  float(fmeanN1)],
        "Fmedian [Hz]":[float(fmedM1), float(fmedS1),   float(fmedN1)],
        "freq95 [Hz]": [float(freq95M1),float(freq95S1), float(freq95N1)],
        
        "E_total":     [float(EM1), float(ES1), float(EN1)],
    },
    index=[ "Miopatía (M1)","Sano (S1)", "Neuropatía (N1)"]
)

# Actividad intensa
df_intensa = pd.DataFrame(
    {
        "Fmean [Hz]":  [float(fmeanM2),  float(fmeanS2),  float(fmeanN2)],
        "Fmedian [Hz]":[float(fmedM2),   float(fmedS2),   float(fmedN2)],
        "freq95 [Hz]": [float(freq95M2), float(freq95S2), float(freq95N2)],
        
        "E_total":     [float(EM2),float(ES2),  float(EN2)],
    },
    index=[ "Miopatía (M2)","Sano (S2)",  "Neuropatía (N2)"]
)

# 3) Reposo
df_reposo = pd.DataFrame(
    {
        "Fmean [Hz]":  [float(fmeanM3), float(fmeanS3), float(fmeanN3)],
        "Fmedian [Hz]":[float(fmedM3),   float(fmedS3),   float(fmedN3)],
        "freq95 [Hz]": [float(freq95M3), float(freq95S3), float(freq95N3)],
        "E_total":     [float(EM3),      float(ES3),      float(EN3)],
    },
    index=["Miopatía (M3)","Sano (S3)", "Neuropatía (N3)"]
)

show_table_figure(df_leve,   "Métricas PSD — Actividad leve")
show_table_figure(df_intensa,"Métricas PSD — Actividad intensa")
show_table_figure(df_reposo, "Métricas PSD — Reposo")