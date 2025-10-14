import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal.windows import dpss



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
 
largoVentana = 1
def ventanas(sig, fs, largoVentana = 1, paso = 0.5):
    N = len(sig)
    Nv = int(largoVentana * fs)#muestras x ventana
    step = int(round(paso*largoVentana*fs))
    inicios = np.arange(0, N - Nv + 1, step) 
    W = sig[inicios[:, None] + np.arange(Nv)[None, :]] #Nv muestras seguidas despues de cadaa inicio
    W = W - W.mean(axis = 1, keepdims = True)
    return W, Nv, inicios

#aplicamos tapers DPSS en cada ventana, luego fft, potencia, promedio entre tapers, promedio entrre ventanas

    
    
def psdMultitaper(sig, fs, largoVentana = 1.0, paso = 0.5, df = 6 ):
    W, nV, inicios = ventanas(sig, fs, largoVentana, paso) 
    Nv = W.shape[1]
    T = Nv / fs
    
    TW = 0.5 * T * df
    K = max(1, int(np.floor(2*TW - 1)))  # cant de tapers buenos
    tapers, avas = dpss(Nv, TW, Kmax=K, return_ratios=True)
    taperPot = np.sum(tapers**2, axis=1)
    wk = avas / avas.sum()  # pesos que suman 1
    
    # aplicar los tapers a las ventanas
    Wtap = W[:, None, :] * tapers[None, :, :]
    
    # FFT -> potencia (densidad dos caras)
    ft = np.abs(np.fft.fft(Wtap, axis=-1))**2 / (fs * taperPot[None, :, None])
    
    # promedio entre tapers -> PSD por ventana (nVentanas, Nv)
    sumaProm = np.sum(ft * wk[None, :, None], axis=1)
    
    # promedio entre ventanas -> PSD global
    Px = np.sum(sumaProm, axis=0) / sumaProm.shape[0]
    
    
    grilla = fs / Nv
    ff = np.linspace(0.0, (Nv-1)*grilla, Nv)
    bfrec = (ff <= fs/2)
    ff = ff[bfrec]
    Px = Px[bfrec]
    Svent = sumaProm[:, bfrec]                     # matriz (nVentanas, nFreq) para espectrograma
    tms = inicios / fs + (Nv / (2*fs))             # tiempos centrales de cada ventana (s)
    
    return ff, Px, Svent, tms 
    
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


f_s1, P_s1, S_s1, t_s1 = psdMultitaper(emgHealthy1, fs, largoVentana=1.0, paso=0.5, df=6.0)
f_m1, P_m1, S_m1, t_m1 = psdMultitaper(emgMyo1,     fs, largoVentana=1.0, paso=0.5, df=6.0)
f_n1, P_n1, S_n1, t_n1 = psdMultitaper(emgNeuro1,   fs, largoVentana=1.0, paso=0.5, df=6.0)

f_s2, P_s2, S_s2, t_s2 = psdMultitaper(emgHealthy2, fs, largoVentana=1.0, paso=0.5, df=6.0)
f_m2, P_m2, S_m2, t_m2 = psdMultitaper(emgMyo2,     fs, largoVentana=1.0, paso=0.5, df=6.0)
f_n2, P_n2, S_n2, t_n2 = psdMultitaper(emgNeuro2,   fs, largoVentana=1.0, paso=0.5, df=6.0)

f_s3, P_s3, S_s3, t_s3 = psdMultitaper(emgHealthy3, fs, largoVentana=1.0, paso=0.5, df=6.0)
f_m3, P_m3, S_m3, t_m3 = psdMultitaper(emgMyo3,     fs, largoVentana=1.0, paso=0.5, df=6.0)
f_n3, P_n3, S_n3, t_n3 = psdMultitaper(emgNeuro3,   fs, largoVentana=1.0, paso=0.5, df=6.0)



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

# plt.figure(figsize=(9,4))
# plt.pcolormesh(t_s, f_s, 10*np.log10(S_s.T + 1e-12), shading='auto')
# plt.title('Healthy — Espectrograma multitaper')
# plt.xlabel('Tiempo [s]'); plt.ylabel('Frecuencia [Hz]')
# plt.ylim(0, 600)
# cbar = plt.colorbar(); cbar.set_label('PSD [dB re W/Hz]')
# plt.tight_layout(); plt.show()

# plt.figure(figsize=(9,4))
# plt.pcolormesh(t_m, f_m, 10*np.log10(S_m.T + 1e-12), shading='auto')
# plt.title('Miopatía — Espectrograma multitaper')
# plt.xlabel('Tiempo [s]'); plt.ylabel('Frecuencia [Hz]')
# plt.ylim(0, 600)
# cbar = plt.colorbar(); cbar.set_label('PSD [dB re W/Hz]')
# plt.tight_layout(); plt.show()

# plt.figure(figsize=(9,4))
# plt.pcolormesh(t_n, f_n, 10*np.log10(S_n.T + 1e-12), shading='auto')
# plt.title('Neuropatía — Espectrograma multitaper')
# plt.xlabel('Tiempo [s]'); plt.ylabel('Frecuencia [Hz]')
# plt.ylim(0, 600)
# cbar = plt.colorbar(); cbar.set_label('PSD [dB re W/Hz]')
# plt.tight_layout(); plt.show()