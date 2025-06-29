
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras_tuner as kt
from sklearn.preprocessing import RobustScaler

tfk  = tf.keras
tfkl = tf.keras.layers

import pycbc.waveform
from pycbc import waveform
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set(style="ticks")
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'Arial',
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 16,
})
palette = sns.color_palette("muted", 6)



from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler




def cal_hphc_from_amp_ph(amplitude, phase):
    hp = amplitude * np.cos(phase)
    hc = amplitude * np.sin(phase)
    return hp, hc


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


amp_model   = tfk.models.load_model('/path/to/your/model/res_mlp_cnn_amplitude')
phase_model = tfk.models.load_model('/path/to/your/model/res_mlp_cnn_phase')




batch_sizes = [1, 10, 50, 100, 500, 1000]
times = {}

for bs in batch_sizes:
    idx0 = 0 
    X_batch    = X_test_normalized[idx0:idx0+bs]
    t0 = time.time()
    
    rec_ph = phase_model.predict(X_batch, verbose=0)
    
    rec_am = amp_model.predict(X_batch, verbose=0)

    rec_ph_orig = scaler_y_phase.inverse_transform(rec_ph)
    rec_am_orig = scaler_y_amplitude.inverse_transform(rec_am)
    hp,hc=cal_hphc_from_amp_ph(rec_am_orig,rec_ph_orig)
    t1 = time.time()
    times[f"batch_{bs}"] = t1 - t0
    print(f"Batch size {bs:5d} → time = {times[f'batch_{bs}']:.4f} s")

np.savez_compressed("/path/to/your/data/res_mlp_cnn_times_cpu.npz", **times)













