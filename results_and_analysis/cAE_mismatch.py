import os, sys, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import RobustScaler

tfk  = tf.keras
tfkl = tf.keras.layers

import keras_tuner as kt
import numpy as np
import pycbc.waveform
from pycbc import waveform
import time
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from scipy.signal import hilbert


def cal_cycles(signal):
    analytic_signal = hilbert(signal)
    phase = np.angle(analytic_signal)
    unwrapped_phase = np.unwrap(phase)
    delta_phi = unwrapped_phase[-1] - unwrapped_phase[0]
    num_cycles = delta_phi / (2*np.pi)
    return num_cycles

def cal_hphc_from_amp_ph(amplitude, phase):
    hp = amplitude * np.cos(phase)
    hc = amplitude * np.sin(phase)
    return hp, hc

def cal_overlap_freq(h1, h2, dt=1.0, psd=1.0, fmin=None, fmax=None, eps=1e-30):
    def _build_psd_on_grid(fgrid, psd_spec, eps_):
        if psd_spec is None or (np.isscalar(psd_spec) and float(psd_spec) == 1.0):
            Sn = np.ones_like(fgrid, dtype=float)
        elif isinstance(psd_spec, str) and psd_spec.lower() in ("aligo_o4high","o4","o4high"):
            try:
                f_file, asd = np.loadtxt("data/aligo_O4high.txt", unpack=True, comments="#", dtype=float)
            except Exception as e:
                raise FileNotFoundError(f"The ASD file cannot be read data/aligo_O4high.txt：{e}")
            Sn_file = asd**2
            Sn = np.interp(fgrid, f_file, Sn_file, left=np.inf, right=np.inf)
        elif callable(psd_spec):
            Sn = np.asarray(psd_spec(fgrid), dtype=float)
            if Sn.shape != fgrid.shape: Sn = np.broadcast_to(Sn, fgrid.shape).astype(float)
        elif isinstance(psd_spec, (tuple, list)) and len(psd_spec) == 2:
            f_psd, Sn_psd = np.asarray(psd_spec[0], float), np.asarray(psd_spec[1], float)
            Sn = np.interp(fgrid, f_psd, Sn_psd, left=np.inf, right=np.inf)
        else:
            Sn = np.asarray(psd_spec, dtype=float)
            if Sn.shape != fgrid.shape:
                raise ValueError(f"psd array shape {Sn.shape} != freq shape {fgrid.shape}")
        return np.maximum(Sn, eps_)
    h1, h2 = np.asarray(h1, float), np.asarray(h2, float)
    n1, n2 = len(h1), len(h2)
    if n1 != n2:
        nmin = min(n1, n2)
        if abs(n1 - n2) <= 5:
            h1, h2 = h1[:nmin], h2[:nmin]
        else:
            raise ValueError(f"Length mismatch >5: len(h1)={n1}, len(h2)={n2}")
    n = len(h1)
    if n < 4: raise ValueError("Signal too short.")
    H1 = np.fft.rfft(h1) * dt
    H2 = np.fft.rfft(h2) * dt
    f  = np.fft.rfftfreq(n, d=dt)
    if f.size <= 1: raise ValueError("No positive frequency bins.")
    H1, H2, f = H1[1:], H2[1:], f[1:]
    df = f[1] - f[0] if f.size > 1 else 1.0/(n*dt)
    Sn = _build_psd_on_grid(f, psd, eps)
    m = np.isfinite(Sn)
    if fmin is not None: m &= (f >= float(fmin))
    if fmax is not None: m &= (f <= float(fmax))
    if not np.any(m): raise ValueError("Empty frequency band after masking.")
    f, H1, H2, Sn = f[m], H1[m], H2[m], Sn[m]
    weight = np.ones_like(f)
    if n % 2 == 0 and f.size > 0 and np.isclose(f[-1], 0.5/dt): weight[-1] = 0.5
    w = weight / Sn
    inner_12 = 4.0 * np.real(np.sum(H1 * np.conjugate(H2) * w)) * df
    inner_11 = 4.0 * np.real(np.sum(H1 * np.conjugate(H1) * w)) * df
    inner_22 = 4.0 * np.real(np.sum(H2 * np.conjugate(H2) * w)) * df
    denom = np.sqrt(max(inner_11, eps) * max(inner_22, eps))
    overlap  = float(inner_12 / denom)
    mismatch = 1.0 - overlap
    return mismatch, overlap





dataset_path = '/path/to/your/data/BNS_test_set.npz'
dataset_dict = np.load(dataset_path)
X_test       = dataset_dict['X_test'].astype('float32')                 
y_amplitude       = dataset_dict['y_amplitude'].astype('float32')         
y_phase       = dataset_dict['y_phase'].astype('float32')  

scaler_X = MinMaxScaler()
X_test_normalized = scaler_X.fit_transform(X_test)
scaler_y_amplitude = StandardScaler()
scaler_y_phase = StandardScaler()
y_amplitude_normalized = scaler_y_amplitude.fit_transform(y_amplitude)
y_phase_normalized = scaler_y_phase.fit_transform(y_phase)

cAE_encoder_phase_conditional = tf.keras.models.load_model(
    '/path/to/your/model/cAE_encoder_phase_conditional'
)
cAE_decoder_phase = tf.keras.models.load_model(
    '/path/to/your/model/cAE_decoder_phase'
)

cAE_encoder_amplitude_conditional = tf.keras.models.load_model(
    '/path/to/your/model/cAE_encoder_amplitude_conditional'
)
cAE_decoder_amplitude = tf.keras.models.load_model(
    '/path/to/your/model/cAE_decoder_amplitude'
)

n   = X_test.shape[0]      
fs  = 4096
dt  = 1.0 / fs
               

phase_z_norm   = cAE_encoder_phase_conditional   .predict(y_phase_normalized, verbose=0)  
phase_rec_norm = cAE_decoder_phase   .predict(phase_z_norm, verbose=0) 
amp_z_norm     = cAE_encoder_amplitude_conditional.predict(y_amplitude_normalized, verbose=0)  
amp_rec_norm   = cAE_decoder_amplitude.predict(amp_z_norm, verbose=0) 

phase_rec = scaler_y_phase    .inverse_transform(phase_rec_norm)   
amp_rec   = scaler_y_amplitude.inverse_transform(amp_rec_norm)     

mismatch_hp      = np.empty(n, dtype=np.float32)
overlap_hp       = np.empty(n, dtype=np.float32)
mismatch_hc      = np.empty(n, dtype=np.float32)
overlap_hc       = np.empty(n, dtype=np.float32)
cycles_hp_true   = np.empty(n, dtype=np.float32)  

for i in tqdm(range(n), desc="Compute hp/hc metrics"):
    hp_pred, hc_pred = cal_hphc_from_amp_ph(amp_rec[i],    phase_rec[i])
    hp_true, hc_true = cal_hphc_from_amp_ph(y_amplitude[i], y_phase[i])

    # mismatch & overlap
    m1, o1 = cal_overlap_freq(hp_true, hp_pred, dt)
    m2, o2 = cal_overlap_freq(hc_true, hc_pred, dt)
    mismatch_hp[i], overlap_hp[i] = m1, o1
    mismatch_hc[i], overlap_hc[i] = m2, o2

    cycles_hp_true[i] = cal_cycles(hp_true)
out_path = ('/path/to/your/data/cae_hp_hc_metrics.npz')
np.savez(
    out_path,
    x_test         = X_test,
    mismatch_hp     = mismatch_hp,
    overlap_hp      = overlap_hp,
    mismatch_hc     = mismatch_hc,
    overlap_hc      = overlap_hc,
    cycles_hp_true  = cycles_hp_true      
)























